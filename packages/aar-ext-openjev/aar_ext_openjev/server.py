"""openjev NLI server — runs the model in its own Python environment.

This file is deliberately standalone (no package-relative imports): the aar
extension launches it by *file path* with the interpreter of a dedicated venv
that has ``torch`` + ``transformers>=5.15`` + ``bitsandbytes``, so aar itself
never needs the heavy ML stack.

    python server.py --port 8765 --bits 4 --device cuda:0

Endpoints (JSON):

    GET  /health    {"status": "loading" | "ready" | "error", ...}
    POST /predict   {"pairs": [{"premise": ..., "hypothesis": ...}]}
                    -> {"labels": [...], "probs": [[...]], "truncated": [...]}
    POST /shutdown  stop the server (frees VRAM)

The socket is bound *before* the model loads, so a second instance on the same
port fails immediately instead of loading 9 GB of weights first.  The server
exits on its own after ``--idle-timeout`` seconds without a request.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger("openjev.server")

DEFAULT_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"
DEFAULT_LABELS = ["contradiction", "entailment", "neutral"]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """What the HTTP layer needs from a model backend (a fake one in tests)."""

    def health(self) -> dict[str, Any]: ...

    def predict(self, pairs: list[tuple[str, str]]) -> dict[str, Any]: ...


@dataclass
class ModelSettings:
    model: str = "AlexWortega/openjev"
    subfolder: str = "qwen3.5-4b-nli"
    bits: int = 4  # 4 (NF4), 8 (LLM.int8) or 16 (bf16, ~9 GB VRAM)
    device: str = "cuda:0"
    max_length: int = 2048
    batch_size: int = 4


@dataclass
class OpenJevBackend:
    """Loads openjev (``Qwen3_5ForSequenceClassification``) and scores NLI pairs.

    Mirrors ``OpenJevCrossEncoder`` from the model repo: right padding, the
    ``score`` head applied to the hidden state of the last non-pad token,
    labels ``0=contradiction, 1=entailment, 2=neutral``.
    """

    settings: ModelSettings
    status: str = "loading"
    error: str | None = None
    load_seconds: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _model: Any = None
    _tok: Any = None
    _labels: list[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    _template: str = DEFAULT_TEMPLATE

    def load(self) -> None:
        t0 = time.monotonic()
        try:
            self._load()
        except Exception as exc:  # reported via /health, the process stays up
            logger.exception("model load failed")
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            return
        self.load_seconds = round(time.monotonic() - t0, 1)
        self.status = "ready"
        logger.info("model ready in %.1fs (%s)", self.load_seconds, self._vram())

    def _load(self) -> None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        s = self.settings
        if s.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment")

        kwargs: dict[str, Any] = {"subfolder": s.subfolder, "dtype": torch.bfloat16}
        # keep the classification head in bf16 — it is tiny and quantizing it hurts most
        if s.bits == 4:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["score"],
            )
        elif s.bits == 8:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=["score"]
            )
        elif s.bits != 16:
            raise ValueError(f"bits must be 4, 8 or 16, got {s.bits}")
        kwargs["device_map"] = {"": s.device}

        logger.info("loading %s/%s (%d-bit) on %s", s.model, s.subfolder, s.bits, s.device)
        tok = AutoTokenizer.from_pretrained(s.model, subfolder=s.subfolder)
        model = AutoModelForSequenceClassification.from_pretrained(s.model, **kwargs).eval()

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"  # the head pools the last non-pad token
        text_cfg = model.config.get_text_config()
        if text_cfg.pad_token_id is None:
            text_cfg.pad_token_id = tok.pad_token_id

        id2label = getattr(model.config, "id2label", None) or {}
        if len(id2label) == 3:
            self._labels = [id2label[i] for i in sorted(id2label)]
        self._template = getattr(model.config, "nli_template", None) or DEFAULT_TEMPLATE
        self._model, self._tok = model, tok

    def _vram(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                dev = torch.device(self.settings.device)
                used = torch.cuda.memory_allocated(dev) / 2**30
                return f"{torch.cuda.get_device_name(dev)}, {used:.2f} GB allocated"
        except Exception:
            pass
        return self.settings.device

    def health(self) -> dict[str, Any]:
        s = self.settings
        return {
            "status": self.status,
            "error": self.error,
            "model": f"{s.model}/{s.subfolder}",
            "bits": s.bits,
            "device": self._vram() if self.status == "ready" else s.device,
            "labels": self._labels,
            "max_length": s.max_length,
            "load_seconds": self.load_seconds,
            "pid": os.getpid(),
        }

    def _fit(self, premise: str, hypothesis: str) -> tuple[str, bool]:
        """Format one pair, cutting the *premise* (never the hypothesis) to max_length."""
        text = self._template.format(premise=premise, hypothesis=hypothesis)
        n = len(self._tok(text, add_special_tokens=True)["input_ids"])
        overflow = n - self.settings.max_length
        if overflow <= 0:
            return text, False
        p_ids = self._tok(premise, add_special_tokens=False)["input_ids"]
        keep = max(0, len(p_ids) - overflow - 8)  # small slack for re-tokenization drift
        premise = self._tok.decode(p_ids[:keep])
        return self._template.format(premise=premise, hypothesis=hypothesis), True

    def predict(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        if self.status != "ready":
            raise RuntimeError(f"model not ready (status={self.status})")
        import torch

        fitted = [self._fit(p.strip(), h.strip()) for p, h in pairs]
        probs: list[list[float]] = []
        backbone = getattr(self._model, self._model.base_model_prefix)
        with self._lock, torch.inference_mode():
            for i in range(0, len(fitted), self.settings.batch_size):
                texts = [t for t, _ in fitted[i : i + self.settings.batch_size]]
                enc = self._tok(
                    texts,
                    truncation=True,
                    max_length=self.settings.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.settings.device)
                hidden = backbone(**enc).last_hidden_state
                last = enc["attention_mask"].sum(1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
                score = self._model.score
                logits = score(pooled.to(score.weight.dtype)).float()
                probs.extend(torch.softmax(logits, -1).cpu().tolist())
        return {
            "labels": self._labels,
            "probs": probs,
            "truncated": [t for _, t in fitted],
        }


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------


class Pair(BaseModel):
    premise: str
    hypothesis: str


class PredictRequest(BaseModel):
    pairs: list[Pair] = Field(min_length=1)


class Activity:
    """Tracks the time of the last request for the idle watchdog."""

    def __init__(self) -> None:
        self.last = time.monotonic()

    def touch(self) -> None:
        self.last = time.monotonic()

    def idle_for(self) -> float:
        return time.monotonic() - self.last


def create_app(
    backend: Backend,
    activity: Activity | None = None,
    on_shutdown: Any = None,
    max_pairs: int = 64,
) -> Any:
    """Build the FastAPI app around *backend* (kept separate for testing)."""
    from fastapi import FastAPI, HTTPException

    activity = activity or Activity()
    app = FastAPI(title="openjev", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return backend.health()

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        if len(req.pairs) > max_pairs:
            raise HTTPException(422, detail=f"at most {max_pairs} pairs per request")
        activity.touch()
        state = backend.health()
        if state["status"] != "ready":
            raise HTTPException(503, detail=state.get("error") or state["status"])
        try:
            return backend.predict([(p.premise, p.hypothesis) for p in req.pairs])
        finally:
            activity.touch()

    @app.post("/shutdown")
    def shutdown() -> dict[str, str]:
        logger.info("shutdown requested via /shutdown")
        if on_shutdown is not None:
            on_shutdown()
        return {"status": "stopping"}

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = ModelSettings()
    ap = argparse.ArgumentParser(description="openjev NLI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--subfolder", default=d.subfolder)
    ap.add_argument("--bits", type=int, choices=[4, 8, 16], default=d.bits)
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--max-length", type=int, default=d.max_length)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument(
        "--idle-timeout",
        type=float,
        default=1800.0,
        help="exit after this many seconds without a request (0 = never)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((args.host, args.port))
    except OSError as exc:
        logger.error(
            "cannot bind %s:%d (%s) — is another server running?", args.host, args.port, exc
        )
        return 2

    import uvicorn

    backend = OpenJevBackend(
        ModelSettings(
            model=args.model,
            subfolder=args.subfolder,
            bits=args.bits,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    )
    activity = Activity()
    server: uvicorn.Server | None = None

    def stop() -> None:
        if server is not None:
            server.should_exit = True

    app = create_app(backend, activity, on_shutdown=stop)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))

    def watchdog() -> None:
        while not server.should_exit:
            time.sleep(10)
            if args.idle_timeout > 0 and activity.idle_for() > args.idle_timeout:
                logger.info("idle for %.0fs — shutting down", activity.idle_for())
                stop()

    threading.Thread(target=backend.load, name="openjev-load", daemon=True).start()
    threading.Thread(target=watchdog, name="openjev-idle", daemon=True).start()
    logger.info("listening on http://%s:%d (pid %d)", args.host, args.port, os.getpid())
    server.run(sockets=[sock])
    logger.info("server stopped (pid %d)", os.getpid())
    return 0


if __name__ == "__main__":
    sys.exit(main())
