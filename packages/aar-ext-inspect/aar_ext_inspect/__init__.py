"""
Aar extension: /inspect

Provides a ``/inspect`` slash-command that analyses the current session
and prints a human-readable report via ``ctx.logger``.

Usage:
    /inspect          — concise session summary
    /inspect verbose  — include detail of the last 20 events
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from agent.extensions.api import ExtensionAPI, ExtensionContext

# ---------------------------------------------------------------------------
# Event-type constants (match agent.core.events.EventType string values)
# ---------------------------------------------------------------------------

_T_USER = "user_message"
_T_ASSISTANT = "assistant_message"
_T_TOOL_CALL = "tool_call"
_T_TOOL_RES = "tool_result"
_T_ERROR = "error"
_T_META = "provider_meta"
_T_STREAM = "stream_chunk"
_T_REASONING = "reasoning"
_T_SESSION = "session"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _etype(ev: Any) -> str:
    """Return the event's type string, lower-cased and stripped.

    ``EventType`` is a ``str`` subclass (StrEnum-like) whose ``str()`` renders
    as ``"EventType.USER_MESSAGE"``.  We need the plain value string
    (``"user_message"``), so we prefer ``.value`` when available.
    """
    raw = getattr(ev, "type", None)
    if raw is None:
        return ev.__class__.__name__.lower()
    # Prefer .value (works for both Enum and plain-str fields)
    if hasattr(raw, "value"):
        return str(raw.value).lower().strip()
    return str(raw).lower().strip()


def _short(text: str | None, limit: int = 400) -> str:
    if not text:
        return ""
    s = str(text)
    return s if len(s) <= limit else s[:limit] + "… (truncated)"


# ---------------------------------------------------------------------------
# Extension entry-point
# ---------------------------------------------------------------------------


def register(api: ExtensionAPI) -> None:
    """Register the /inspect slash-command."""

    @api.command("inspect", description="Analyse the current session and print a summary report")
    def inspect_command(args: str, ctx: ExtensionContext) -> str:
        session = ctx.session
        cfg = ctx.config

        # Guard: session object is None or has no session_id (can happen when
        # the extension context still holds the internal bootstrap placeholder
        # and update_session() has not been called yet).
        if session is None or not getattr(session, "session_id", None):
            report = "No active session — send a message or resume a session first."
            ctx.logger.info(report)
            return report

        lines: list[str] = []

        lines.append("=== Session Inspect Report ===")

        # ------------------------------------------------------------------
        # Session metadata  (fields that live directly on Session)
        # ------------------------------------------------------------------
        sid = getattr(session, "session_id", None) or "<no-session-id>"
        trace = getattr(session, "trace_id", None)
        step_count = getattr(session, "step_count", 0)
        state = getattr(session, "state", None)

        lines.append(f"Session ID : {sid}")
        if trace:
            lines.append(f"Trace ID   : {trace}")
        lines.append(f"State      : {state}")
        lines.append(f"Step count : {step_count}")

        # Token / cost fields stored directly on Session
        total_in = int(getattr(session, "total_input_tokens", 0) or 0)
        total_out = int(getattr(session, "total_output_tokens", 0) or 0)
        total_cost = float(getattr(session, "total_cost", 0.0) or 0.0)

        # ------------------------------------------------------------------
        # Events  (session.events is the canonical list)
        # ------------------------------------------------------------------
        events: list[Any] = list(getattr(session, "events", None) or [])
        lines.append(f"Total events: {len(events)}")

        # Per-type counters
        type_counts: Counter[str] = Counter()
        tool_call_counter: Counter[str] = Counter()
        tool_result_counter: Counter[str] = Counter()
        last_assistant: str | None = None
        meta_in = 0
        meta_out = 0

        for ev in events:
            t = _etype(ev)
            type_counts[t] += 1

            if t == _T_TOOL_CALL:
                tool_call_counter[str(getattr(ev, "tool_name", "<unknown>"))] += 1

            elif t == _T_TOOL_RES:
                tool_result_counter[str(getattr(ev, "tool_name", "<unknown>"))] += 1

            elif t == _T_ASSISTANT:
                content = getattr(ev, "content", None)
                if content:
                    last_assistant = str(content)

            elif t == _T_META:
                usage = getattr(ev, "usage", None) or {}
                meta_in += int(usage.get("input_tokens", 0) or 0)
                meta_out += int(usage.get("output_tokens", 0) or 0)

        user_msgs = type_counts.get(_T_USER, 0)
        assistant_msgs = type_counts.get(_T_ASSISTANT, 0)
        tool_calls = type_counts.get(_T_TOOL_CALL, 0)
        tool_results = type_counts.get(_T_TOOL_RES, 0)
        errors = type_counts.get(_T_ERROR, 0)
        stream_chunks = type_counts.get(_T_STREAM, 0)
        reasoning_ev = type_counts.get(_T_REASONING, 0)

        lines.append(f"User messages     : {user_msgs}")
        lines.append(f"Assistant messages: {assistant_msgs}")
        lines.append(f"Tool calls        : {tool_calls}")
        lines.append(f"Tool results      : {tool_results}")
        lines.append(f"Errors            : {errors}")
        lines.append(f"Stream chunks     : {stream_chunks}")
        lines.append(f"Reasoning blocks  : {reasoning_ev}")

        # ------------------------------------------------------------------
        # Tool breakdown
        # ------------------------------------------------------------------
        if tool_call_counter:
            lines.append("")
            lines.append("Tool calls by name:")
            for name, cnt in tool_call_counter.most_common():
                lines.append(f"  • {name}: {cnt}×")

        if tool_result_counter:
            lines.append("Tool results by name:")
            for name, cnt in tool_result_counter.most_common():
                lines.append(f"  • {name}: {cnt}×")

        # ------------------------------------------------------------------
        # Token usage
        # ------------------------------------------------------------------
        # Prefer session-level totals (always up to date after each turn);
        # fall back to summing ProviderMeta events if the fields are zero.
        if total_in == 0 and total_out == 0:
            total_in, total_out = meta_in, meta_out

        if total_in or total_out:
            lines.append("")
            lines.append("Token usage:")
            lines.append(f"  • Input tokens : {total_in}")
            lines.append(f"  • Output tokens: {total_out}")
            lines.append(f"  • Total tokens : {total_in + total_out}")
            if total_cost:
                lines.append(f"  • Estimated cost: ${total_cost:.4f}")

        # ------------------------------------------------------------------
        # Last assistant reply
        # ------------------------------------------------------------------
        if last_assistant:
            lines.append("")
            lines.append("Last assistant message (truncated to 400 chars):")
            lines.append(_short(last_assistant, 400))

        # ------------------------------------------------------------------
        # Provider snapshot from config
        # ------------------------------------------------------------------
        provider = getattr(cfg, "provider", None)
        if provider:
            try:
                pname = getattr(provider, "name", None) or str(provider)
                pmodel = getattr(provider, "model", None)
                lines.append("")
                lines.append("Provider:")
                lines.append(f"  • Name : {pname}")
                if pmodel:
                    lines.append(f"  • Model: {pmodel}")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Verbose: last-20 events detail
        # ------------------------------------------------------------------
        verbose = (args or "").strip().lower() == "verbose"
        if verbose:
            lines.append("")
            lines.append("Event detail (last 20):")
            for i, ev in enumerate(events[-20:], 1):
                try:
                    t = _etype(ev)
                    parts: list[str] = [f"type={t}"]
                    if hasattr(ev, "tool_name"):
                        parts.append(f"tool={getattr(ev, 'tool_name')}")
                    if t == _T_ASSISTANT:
                        parts.append(f"content={_short(getattr(ev, 'content', ''), 80)}")
                    elif t == _T_ERROR:
                        parts.append(f"msg={_short(getattr(ev, 'message', ''), 80)}")
                    elif t == _T_META:
                        u = getattr(ev, "usage", {}) or {}
                        parts.append(
                            f"in={u.get('input_tokens', 0)} out={u.get('output_tokens', 0)}"
                        )
                    elif t == _T_USER:
                        parts.append(f"content={_short(getattr(ev, 'content', ''), 80)}")
                    lines.append(f"  [{i:>2}] " + " | ".join(parts))
                except Exception:
                    lines.append(f"  [{i:>2}] <uninspectable event>")

        lines.append("=== End Report ===")
        report = "\n".join(lines)
        ctx.logger.info(report)
        return report
