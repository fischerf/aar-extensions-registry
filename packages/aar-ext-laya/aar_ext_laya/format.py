"""Render Laya's answer objects as text for the model and for slash-commands.

Laya returns one answer per question key, shaped by the question's type::

    choice  {"type": "choice", "choice": str, "probabilities": {opt: p},
             "confidence": float, "action": {"act_probability": float}}
    score   {"type": "score", "score": float, "legend": {idx: level},
             "probabilities": {idx: p}, "confidence": float, ...}
    noul    {"type": "noul", "noul": float, "confidence": float, "action": {...}}

Shapes verified against laya 0.1.6.  Note that the model repo's own
``rl_agent_api.py`` names the last key ``rl_agent`` while the pip package emits
``action``; nothing here reads it, but do not rely on either name.

Every accessor here is defensive: laya is a 0.1.x package and the extension
must degrade to something readable rather than raise on a shape change.
"""

from __future__ import annotations

from typing import Any

# Qualitative bands for a calibrated P(true). The model reads these, so they
# must not overstate: Laya is calibrated but still a 0.4B classifier.
BANDS = (
    (0.85, "very likely"),
    (0.60, "likely"),
    (0.40, "uncertain"),
    (0.15, "unlikely"),
)


def band(p: float) -> str:
    for threshold, label in BANDS:
        if p >= threshold:
            return label
    return "very unlikely"


def answers_of(result: Any) -> dict[str, Any]:
    """Pull the ``answers`` mapping out of a /predict payload."""
    if isinstance(result, dict):
        answers = result.get("answers")
        if isinstance(answers, dict):
            return answers
    return {}


def noul_of(answers: dict[str, Any], key: str) -> float | None:
    """Return the calibrated P(true) for a noul question, or None."""
    answer = answers.get(key)
    if isinstance(answer, dict):
        value = answer.get("noul")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _legend(answer: dict[str, Any]) -> dict[str, str]:
    """Map a score answer's probability keys to level names.

    Laya keys score probabilities by ordinal index ("0", "1", …) and ships the
    names separately in ``legend``, so without this the output reads
    "3 0.67" instead of "critical 0.67".
    """
    legend = answer.get("legend")
    if isinstance(legend, dict):
        return {str(k): str(v) for k, v in legend.items()}
    if isinstance(legend, list):
        return {str(i): str(v) for i, v in enumerate(legend)}
    return {}


def _probs(answer: dict[str, Any], limit: int = 6) -> str:
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return ""
    names = _legend(answer)
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return " · ".join(f"{names.get(str(k), k)} {float(v):.2f}" for k, v in items)


def _confidence(answer: dict[str, Any]) -> str:
    value = answer.get("confidence")
    return f"confidence {float(value):.2f}" if isinstance(value, (int, float)) else ""


def _join(*parts: str) -> str:
    kept = [p for p in parts if p]
    return f" ({'; '.join(kept)})" if kept else ""


def format_answer(key: str, answer: Any) -> str:
    """One line per answer, type-aware."""
    if not isinstance(answer, dict):
        return f"{key}: (no answer)"

    qtype = answer.get("type")

    if qtype == "choice" or "choice" in answer:
        choice = answer.get("choice", "?")
        return f"{key}: {choice}{_join(_confidence(answer), _probs(answer))}"

    if qtype == "score" or "score" in answer:
        score = answer.get("score")
        shown = f"{float(score):.2f}" if isinstance(score, (int, float)) else "?"
        names = _legend(answer)
        scale = ""
        if names:
            ordered = [names[k] for k in sorted(names, key=lambda k: (len(k), k))]
            scale = f"scale: {', '.join(ordered)}"
        return f"{key}: {shown}{_join(_confidence(answer), scale, _probs(answer))}"

    if qtype == "noul" or "noul" in answer:
        value = answer.get("noul")
        if isinstance(value, (int, float)):
            p = float(value)
            return f"{key}: {band(p)} (P={p:.2f})"
        return f"{key}: (no probability)"

    return f"{key}: {answer}"


def format_answers(result: Any, order: list[str] | None = None) -> str:
    """Render every answer, optionally in the order the questions were asked."""
    answers = answers_of(result)
    if not answers:
        return "laya returned no answers"
    keys = [k for k in (order or []) if k in answers]
    keys += [k for k in answers if k not in keys]
    return "\n".join(format_answer(k, answers[k]) for k in keys)


def clip(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"
