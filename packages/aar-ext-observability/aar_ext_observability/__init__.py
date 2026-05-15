"""Aar extension: observability — structured metrics after each turn.

Emits per-turn metrics (tokens, cost, duration, tool calls, errors) as
structured log records. Designed as a base for Prometheus, OpenTelemetry,
or custom metrics pipelines.
"""

from __future__ import annotations

import json
import time
from typing import Any


def register(api: Any) -> None:
    """Register the observability extension."""

    turn_count = 0
    session_start_time: float = 0.0

    @api.on("session_start")
    def on_start(event: Any, ctx: Any) -> None:
        nonlocal session_start_time
        session_start_time = time.monotonic()
        ctx.logger.info(
            "observability: session started",
            extra={"session_id": ctx.session.session_id},
        )

    @api.on("after_turn")
    def on_after_turn(event: Any, ctx: Any) -> None:
        nonlocal turn_count
        turn_count += 1

        metrics = {
            "event": "turn_complete",
            "session_id": ctx.session.session_id,
            "turn": turn_count,
            "step_count": ctx.session.step_count,
            "total_input_tokens": ctx.session.total_input_tokens,
            "total_output_tokens": ctx.session.total_output_tokens,
            "total_tokens": ctx.session.total_tokens,
            "total_cost_usd": round(ctx.session.total_cost, 6),
            "elapsed_s": round(time.monotonic() - session_start_time, 2),
        }

        ctx.logger.info("observability: %s", json.dumps(metrics))

        # Emit on the extension event bus for other extensions to consume
        api.events.emit("metrics:turn", metrics)

    @api.on("session_end")
    def on_end(event: Any, ctx: Any) -> None:
        elapsed = time.monotonic() - session_start_time if session_start_time else 0
        summary = {
            "event": "session_complete",
            "session_id": ctx.session.session_id,
            "total_turns": turn_count,
            "total_steps": ctx.session.step_count,
            "total_input_tokens": ctx.session.total_input_tokens,
            "total_output_tokens": ctx.session.total_output_tokens,
            "total_tokens": ctx.session.total_tokens,
            "total_cost_usd": round(ctx.session.total_cost, 6),
            "elapsed_s": round(elapsed, 2),
        }

        ctx.logger.info("observability: session summary: %s", json.dumps(summary))
        api.events.emit("metrics:session", summary)

    @api.on("error")
    def on_error(event: Any, ctx: Any) -> None:
        error_data = {
            "event": "error",
            "session_id": ctx.session.session_id,
            "message": getattr(event, "message", str(event)),
            "recoverable": getattr(event, "recoverable", True),
        }
        ctx.logger.warning("observability: %s", json.dumps(error_data))
        api.events.emit("metrics:error", error_data)

    # Slash command — for the user, not the LLM.
    @api.command("stats", description="Show current session metrics (turns, elapsed time)")
    def stats_command(args: str, ctx: Any) -> str:
        return json.dumps(
            {
                "turns": turn_count,
                "elapsed_s": round(time.monotonic() - session_start_time, 2)
                if session_start_time
                else 0,
            },
            indent=2,
        )
