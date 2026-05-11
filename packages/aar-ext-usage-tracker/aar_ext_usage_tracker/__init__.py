"""Aar extension: usage-tracker — API quota tracking via ~/.aar/usage.json.

Tracks per-provider monthly request counts and token usage.  Warns when
approaching quota limits and provides a ``/usage`` slash command and a
``usage_status`` tool the LLM can invoke.

Quota limits are read from the active provider's ``extra.quota`` config::

    "extra": {
        "quota": {
            "monthly_requests": 1000,
            "warn_at_percent": 80.0
        }
    }
"""

from __future__ import annotations

from typing import Any


def register(api: Any) -> None:
    """Register the usage-tracker extension."""

    from aar_ext_usage_tracker.tracker import UsageTracker

    tracker = UsageTracker()
    _prev_input: int = 0
    _prev_output: int = 0
    _warned: bool = False

    # -- lifecycle ----------------------------------------------------------

    @api.on("session_start")
    def on_start(event: Any, ctx: Any) -> None:
        nonlocal _prev_input, _prev_output, _warned
        tracker.load()
        _prev_input = 0
        _prev_output = 0
        _warned = False

        # Seed quota from provider config when present
        provider_key = ctx.config.provider
        provider_cfg = ctx.config.resolve_provider()
        quota_cfg = provider_cfg.extra.get("quota", {})
        if quota_cfg:
            tracker.set_quota(
                provider_key,
                monthly_requests=quota_cfg.get("monthly_requests", 0),
                warn_at_percent=quota_cfg.get("warn_at_percent", 80.0),
            )
            tracker.save()

        # Early warning if already at or near the limit
        exceeded, msg = tracker.check_quota(provider_key)
        if exceeded:
            ctx.logger.warning("usage-tracker: %s", msg)
        else:
            warning, msg = tracker.check_warning(provider_key)
            if warning:
                ctx.logger.warning("usage-tracker: %s", msg)

    @api.on("after_turn")
    def on_after_turn(event: Any, ctx: Any) -> None:
        nonlocal _prev_input, _prev_output, _warned
        provider_key = ctx.config.provider

        # Calculate delta tokens for this turn
        delta_input = max(0, ctx.session.total_input_tokens - _prev_input)
        delta_output = max(0, ctx.session.total_output_tokens - _prev_output)
        _prev_input = ctx.session.total_input_tokens
        _prev_output = ctx.session.total_output_tokens

        tracker.record_request(
            provider_key,
            input_tokens=delta_input,
            output_tokens=delta_output,
        )
        tracker.save()

        # Quota checks
        if not _warned:
            warning, msg = tracker.check_warning(provider_key)
            if warning:
                ctx.logger.warning("usage-tracker: %s", msg)
                _warned = True

        exceeded, msg = tracker.check_quota(provider_key)
        if exceeded:
            ctx.logger.error("usage-tracker: %s", msg)

    @api.on("session_end")
    def on_end(event: Any, ctx: Any) -> None:
        tracker.save()

    # -- slash command ------------------------------------------------------

    @api.command("usage", description="Show API usage and quota status")
    def usage_command(args: str, ctx: Any) -> str:
        tracker.load()  # reload latest from disk
        provider = args.strip() if args and args.strip() else None
        return tracker.format_status(provider_key=provider)

    # -- tool ---------------------------------------------------------------

    @api.tool(
        name="usage_status",
        description=(
            "Show current API usage statistics and quota status. "
            "Returns monthly request counts, token usage, and remaining quota."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Optional provider key to filter (e.g. 'claude', 'gp')",
                },
            },
            "additionalProperties": False,
        },
    )
    def usage_status_tool(provider: str = "", **kwargs: Any) -> str:
        tracker.load()
        return tracker.format_status(provider_key=provider or None)

    # -- system prompt hint -------------------------------------------------

    api.append_system_prompt(
        "The /usage command shows API quota and usage statistics. "
        "Use the usage_status tool to check remaining monthly requests."
    )
