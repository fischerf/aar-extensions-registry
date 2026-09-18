"""laya decision server — runs the model in its own Python environment, CPU-only.

This file is deliberately standalone (no package-relative imports): the aar
extension launches it by *file path* with the interpreter of a dedicated venv
that has ``laya`` + CPU ``torch``, so aar itself never needs the ML stack.

    python server.py --port 8770 --threads 4

Endpoints (JSON):

    GET  /health    {"status": "loading" | "ready" | "error", ...}
    POST /predict   {"state": <str|object>, "questions": {qid: {...}}}
                    -> {"model": ..., "answers": {qid: {...}}, "usage": {...}}
    POST /shutdown  stop the server

The socket is bound *before* the model loads, so a second instance on the same
port fails immediately instead of downloading weights first.  The server exits
on its own after ``--idle-timeout`` seconds without a request.

CPU is enforced two ways: the extension clears ``CUDA_VISIBLE_DEVICES`` in the
child environment, and this module clears it again before torch is ever
imported.  Laya's own loader picks CUDA when it is visible, so both matter.
"""

from __future__ import annotations

import argparse
import logging
import os

# Must happen before torch is imported anywhere in this process.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Protocol  # noqa: E402

from pydantic import BaseModel, Field  # noqa: E402

logger = logging.getLogger("laya.server")

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """What the HTTP layer needs from a model backend (a fake one in tests)."""

    def health(self) -> dict[str, Any]: ...

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class ModelSettings:
    model: str = "convaiinnovations/laya"
    threads: int = 0  # torch intra-op threads; 0 = leave torch's default alone
    max_questions: int = 16


@dataclass
class LayaBackend:
    """Loads Laya through the ``laya`` package and answers typed questions.

    Everything that touches ``laya`` is funnelled through here, because it is a
    0.1.x package: ``load()`` may or may not accept ``device``, and the agent
    object exposes ``predict()`` (pip wrapper) or ``system_one()`` (the raw
    ``RLAgent`` from the model repo).  Both are probed at runtime.
    """

    settings: ModelSettings
    status: str = "loading"
    error: str | None = None
    load_seconds: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _agent: Any = None
    _predict: Any = None

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
        logger.info("model ready in %.1fs on %s", self.load_seconds, DEVICE)

    def _load(self) -> None:
        import torch

        s = self.settings
        if s.threads > 0:
            torch.set_num_threads(s.threads)

        import laya

        logger.info("loading %s on %s (%d threads)", s.model, DEVICE, torch.get_num_threads())
        try:
            agent = laya.load(s.model, device=DEVICE)
        except TypeError:  # older/newer signature without a device kwarg
            agent = laya.load(s.model)

        predict = getattr(agent, "predict", None) or getattr(agent, "system_one", None)
        if predict is None:
            raise RuntimeError(
                "the loaded laya agent exposes neither predict() nor system_one() — "
                "pin a compatible laya release (>=0.1.5,<0.2)"
            )
        self._agent, self._predict = agent, predict

    def health(self) -> dict[str, Any]:
        s = self.settings
        return {
            "status": self.status,
            "error": self.error,
            "model": s.model,
            "device": DEVICE,
            "threads": s.threads or _torch_threads(),
            "max_questions": s.max_questions,
            "load_seconds": self.load_seconds,
            "pid": os.getpid(),
        }

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if self.status != "ready":
            raise RuntimeError(f"model not ready (status={self.status})")
        # Laya batches internally; the lock keeps concurrent requests off one
        # another's tensors, which torch does not guarantee for a shared module.
        with self._lock:
            result = self._predict(state, questions)
        if not isinstance(result, dict) or "answers" not in result:
            raise RuntimeError(f"unexpected laya result shape: {type(result).__name__}")
        return result


def _torch_threads() -> int:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    state: str | dict[str, Any]
    questions: dict[str, Any] = Field(min_length=1)


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
    max_questions: int = 16,
) -> Any:
    """Build the FastAPI app around *backend* (kept separate for testing)."""
    from fastapi import FastAPI, HTTPException

    activity = activity or Activity()
    app = FastAPI(title="laya", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return backend.health()

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        if len(req.questions) > max_questions:
            raise HTTPException(422, detail=f"at most {max_questions} questions per request")
        activity.touch()
        state = backend.health()
        if state["status"] != "ready":
            raise HTTPException(503, detail=state.get("error") or state["status"])
        try:
            return backend.predict(req.state, req.questions)
        except (KeyError, TypeError, ValueError) as exc:
            # Malformed question spec — the caller can fix this, so 422 not 500.
            raise HTTPException(422, detail=f"{type(exc).__name__}: {exc}") from exc
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
    ap = argparse.ArgumentParser(description="laya decision server (CPU-only)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--model", default=d.model)
    ap.add_argument(
        "--threads",
        type=int,
        default=d.threads,
        help="torch intra-op threads (0 = leave torch's default alone)",
    )
    ap.add_argument("--max-questions", type=int, default=d.max_questions)
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

    backend = LayaBackend(
        ModelSettings(
            model=args.model,
            threads=args.threads,
            max_questions=args.max_questions,
        )
    )
    activity = Activity()
    server: uvicorn.Server | None = None

    def stop() -> None:
        if server is not None:
            server.should_exit = True

    app = create_app(backend, activity, on_shutdown=stop, max_questions=args.max_questions)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))

    def watchdog() -> None:
        while not server.should_exit:
            time.sleep(10)
            if args.idle_timeout > 0 and activity.idle_for() > args.idle_timeout:
                logger.info("idle for %.0fs — shutting down", activity.idle_for())
                stop()

    threading.Thread(target=backend.load, name="laya-load", daemon=True).start()
    threading.Thread(target=watchdog, name="laya-idle", daemon=True).start()
    logger.info("listening on http://%s:%d (pid %d)", args.host, args.port, os.getpid())
    server.run(sockets=[sock])
    logger.info("server stopped (pid %d)", os.getpid())
    return 0


if __name__ == "__main__":
    sys.exit(main())
