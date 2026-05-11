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

Standalone usage — drop this file into ``~/.aar/extensions/`` as
``aar_ext_usage_tracker.py`` and it will be loaded automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_PATH: Path = Path.home() / ".aar" / "usage.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Quota:
    """Per-provider quota configuration."""

    monthly_requests: int = 0  # 0 = unlimited
    warn_at_percent: float = 80.0  # warn when usage reaches this %


@dataclass
class MonthlyUsage:
    """Usage counters for a single calendar month."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    first_request_at: str = ""
    last_request_at: str = ""


@dataclass
class ProviderUsage:
    """Per-provider usage data and quota config."""

    quota: Quota = field(default_factory=Quota)
    months: dict[str, MonthlyUsage] = field(default_factory=dict)

    def current_month(self) -> MonthlyUsage:
        """Return (or create) the usage record for the current UTC month."""
        key = _month_key()
        if key not in self.months:
            self.months[key] = MonthlyUsage()
        return self.months[key]


@dataclass
class UsageData:
    """Root usage data structure persisted to disk."""

    version: int = 1
    providers: dict[str, ProviderUsage] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _month_key() -> str:
    """Return the current UTC month as ``YYYY-MM``."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize(data: UsageData) -> dict[str, Any]:
    """Convert ``UsageData`` to a JSON-compatible dict."""
    result: dict[str, Any] = {"version": data.version, "providers": {}}
    for key, pu in data.providers.items():
        result["providers"][key] = {
            "quota": {
                "monthly_requests": pu.quota.monthly_requests,
                "warn_at_percent": pu.quota.warn_at_percent,
            },
            "months": {
                mk: {
                    "requests": mu.requests,
                    "input_tokens": mu.input_tokens,
                    "output_tokens": mu.output_tokens,
                    "first_request_at": mu.first_request_at,
                    "last_request_at": mu.last_request_at,
                }
                for mk, mu in pu.months.items()
            },
        }
    return result


def _deserialize(raw: dict[str, Any]) -> UsageData:
    """Reconstruct ``UsageData`` from a parsed JSON dict."""
    data = UsageData(version=raw.get("version", 1))
    for key, pu_raw in raw.get("providers", {}).items():
        q_raw = pu_raw.get("quota", {})
        quota = Quota(
            monthly_requests=q_raw.get("monthly_requests", 0),
            warn_at_percent=q_raw.get("warn_at_percent", 80.0),
        )
        months: dict[str, MonthlyUsage] = {}
        for mk, mu_raw in pu_raw.get("months", {}).items():
            months[mk] = MonthlyUsage(
                requests=mu_raw.get("requests", 0),
                input_tokens=mu_raw.get("input_tokens", 0),
                output_tokens=mu_raw.get("output_tokens", 0),
                first_request_at=mu_raw.get("first_request_at", ""),
                last_request_at=mu_raw.get("last_request_at", ""),
            )
        data.providers[key] = ProviderUsage(quota=quota, months=months)
    return data


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class UsageTracker:
    """Tracks API usage per provider per month, persisted to ``~/.aar/usage.json``."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_PATH
        self._data = UsageData()

    # -- persistence --------------------------------------------------------

    def load(self) -> None:
        """Load usage data from disk (no-op if the file doesn't exist)."""
        if not self._path.is_file():
            self._data = UsageData()
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = _deserialize(raw)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            self._data = UsageData()

    def save(self) -> None:
        """Persist usage data to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(_serialize(self._data), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # -- recording ----------------------------------------------------------

    def record_request(
        self,
        provider_key: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a single API request for *provider_key*."""
        pu = self._ensure_provider(provider_key)
        month = pu.current_month()
        now = datetime.now(timezone.utc).isoformat()

        month.requests += 1
        month.input_tokens += input_tokens
        month.output_tokens += output_tokens

        if not month.first_request_at:
            month.first_request_at = now
        month.last_request_at = now

    # -- quota management ---------------------------------------------------

    def set_quota(
        self,
        provider_key: str,
        monthly_requests: int = 0,
        warn_at_percent: float = 80.0,
    ) -> None:
        """Set (or update) the quota for *provider_key*."""
        pu = self._ensure_provider(provider_key)
        pu.quota.monthly_requests = monthly_requests
        pu.quota.warn_at_percent = warn_at_percent

    def check_quota(self, provider_key: str) -> tuple[bool, str]:
        """Check if the monthly quota is exceeded.

        Returns ``(exceeded, message)``.
        """
        pu = self._data.providers.get(provider_key)
        if not pu or pu.quota.monthly_requests <= 0:
            return False, ""

        month = pu.current_month()
        limit = pu.quota.monthly_requests

        if month.requests >= limit:
            return True, (
                f"Monthly request limit reached for '{provider_key}': {month.requests}/{limit}"
            )
        return False, ""

    def check_warning(self, provider_key: str) -> tuple[bool, str]:
        """Check if usage is approaching the quota threshold.

        Returns ``(warning, message)``.
        """
        pu = self._data.providers.get(provider_key)
        if not pu or pu.quota.monthly_requests <= 0:
            return False, ""

        month = pu.current_month()
        limit = pu.quota.monthly_requests
        threshold = int(limit * pu.quota.warn_at_percent / 100.0)

        if threshold <= month.requests < limit:
            remaining = limit - month.requests
            pct = month.requests / limit * 100
            return True, (
                f"Approaching monthly limit for '{provider_key}': "
                f"{month.requests}/{limit} ({pct:.0f}%) \u2014 {remaining} remaining"
            )
        return False, ""

    # -- read helpers -------------------------------------------------------

    def get_month_usage(self, provider_key: str) -> MonthlyUsage | None:
        """Return the current month's usage for *provider_key* (or ``None``)."""
        pu = self._data.providers.get(provider_key)
        if not pu:
            return None
        return pu.months.get(_month_key())

    def get_quota(self, provider_key: str) -> Quota | None:
        """Return the quota for *provider_key* (or ``None``)."""
        pu = self._data.providers.get(provider_key)
        return pu.quota if pu else None

    # -- formatting ---------------------------------------------------------

    def format_status(self, provider_key: str | None = None) -> str:
        """Format usage status as human-readable text."""
        lines: list[str] = []
        keys = [provider_key] if provider_key else sorted(self._data.providers.keys())
        month_key = _month_key()

        for key in keys:
            pu = self._data.providers.get(key)
            if not pu:
                lines.append(f"{key}: no usage data")
                continue

            month = pu.months.get(month_key, MonthlyUsage())
            lines.append(f"\u2500\u2500 {key} ({month_key}) \u2500\u2500")

            if pu.quota.monthly_requests > 0:
                pct = month.requests / pu.quota.monthly_requests * 100
                remaining = pu.quota.monthly_requests - month.requests
                lines.append(
                    f"  requests:  {month.requests} / {pu.quota.monthly_requests} "
                    f"({pct:.1f}%) \u2014 {remaining} remaining"
                )
            else:
                lines.append(f"  requests:  {month.requests} (no limit)")

            lines.append(f"  input:     {month.input_tokens:,} tokens")
            lines.append(f"  output:    {month.output_tokens:,} tokens")
            total = month.input_tokens + month.output_tokens
            lines.append(f"  total:     {total:,} tokens")

            if month.first_request_at:
                lines.append(f"  first:     {month.first_request_at}")
            if month.last_request_at:
                lines.append(f"  last:      {month.last_request_at}")

        return "\n".join(lines) if lines else "No usage data recorded."

    # -- internals ----------------------------------------------------------

    def _ensure_provider(self, key: str) -> ProviderUsage:
        if key not in self._data.providers:
            self._data.providers[key] = ProviderUsage()
        return self._data.providers[key]


# ---------------------------------------------------------------------------
# Extension entry-point
# ---------------------------------------------------------------------------


def _resolve_provider_key(config: Any) -> str:
    """Extract a stable string key for the active provider.

    The CLI's ``_build_config`` replaces ``config.provider`` (a string key) with
    the resolved ``ProviderConfig`` object so flags like ``--model`` can mutate it.
    This helper recovers the original key in both cases.
    """
    prov = config.provider
    if isinstance(prov, str):
        return prov
    # ProviderConfig object — find the matching key in the providers dict
    for key, pcfg in getattr(config, "providers", {}).items():
        if pcfg is prov:
            return key
    # Fallback: use the model name as identifier
    return getattr(prov, "model", None) or getattr(prov, "name", "unknown")


def register(api: Any) -> None:
    """Register the usage-tracker extension."""

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
        provider_key = _resolve_provider_key(ctx.config)
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
        provider_key = _resolve_provider_key(ctx.config)

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
