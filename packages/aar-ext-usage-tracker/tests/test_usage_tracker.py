"""Tests for the usage-tracker extension."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aar_ext_usage_tracker import (
    MonthlyUsage,
    ProviderUsage,
    Quota,
    UsageData,
    UsageTracker,
    _deserialize,
    _month_key,
    _serialize,
)

# ---------------------------------------------------------------------------
# Tracker core
# ---------------------------------------------------------------------------


@pytest.fixture()
def tracker(tmp_path: Path) -> UsageTracker:
    return UsageTracker(path=tmp_path / "usage.json")


def test_fresh_tracker_has_no_data(tracker: UsageTracker) -> None:
    assert tracker.format_status() == "No usage data recorded."


def test_record_request_increments_counters(tracker: UsageTracker) -> None:
    tracker.record_request("claude", input_tokens=100, output_tokens=50)
    tracker.record_request("claude", input_tokens=200, output_tokens=100)

    usage = tracker.get_month_usage("claude")
    assert usage is not None
    assert usage.requests == 2
    assert usage.input_tokens == 300
    assert usage.output_tokens == 150


def test_record_request_sets_timestamps(tracker: UsageTracker) -> None:
    tracker.record_request("claude")
    usage = tracker.get_month_usage("claude")
    assert usage is not None
    assert usage.first_request_at != ""
    assert usage.last_request_at != ""


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    t1 = UsageTracker(path=path)
    t1.set_quota("claude", monthly_requests=1000, warn_at_percent=80.0)
    t1.record_request("claude", input_tokens=500, output_tokens=200)
    t1.save()

    t2 = UsageTracker(path=path)
    t2.load()

    quota = t2.get_quota("claude")
    assert quota is not None
    assert quota.monthly_requests == 1000
    assert quota.warn_at_percent == 80.0

    usage = t2.get_month_usage("claude")
    assert usage is not None
    assert usage.requests == 1
    assert usage.input_tokens == 500
    assert usage.output_tokens == 200


def test_load_missing_file(tracker: UsageTracker) -> None:
    tracker.load()
    assert tracker.format_status() == "No usage data recorded."


def test_load_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    path.write_text("not valid json {{{", encoding="utf-8")
    t = UsageTracker(path=path)
    t.load()
    assert t.format_status() == "No usage data recorded."


# ---------------------------------------------------------------------------
# Quota checks
# ---------------------------------------------------------------------------


def test_check_quota_no_limit(tracker: UsageTracker) -> None:
    tracker.record_request("claude")
    exceeded, msg = tracker.check_quota("claude")
    assert exceeded is False
    assert msg == ""


def test_check_quota_under_limit(tracker: UsageTracker) -> None:
    tracker.set_quota("claude", monthly_requests=10)
    tracker.record_request("claude")
    exceeded, msg = tracker.check_quota("claude")
    assert exceeded is False


def test_check_quota_at_limit(tracker: UsageTracker) -> None:
    tracker.set_quota("claude", monthly_requests=3)
    for _ in range(3):
        tracker.record_request("claude")
    exceeded, msg = tracker.check_quota("claude")
    assert exceeded is True
    assert "3/3" in msg


def test_check_quota_over_limit(tracker: UsageTracker) -> None:
    tracker.set_quota("claude", monthly_requests=2)
    for _ in range(5):
        tracker.record_request("claude")
    exceeded, msg = tracker.check_quota("claude")
    assert exceeded is True


def test_check_quota_unknown_provider(tracker: UsageTracker) -> None:
    exceeded, msg = tracker.check_quota("nonexistent")
    assert exceeded is False
    assert msg == ""


# ---------------------------------------------------------------------------
# Warning checks
# ---------------------------------------------------------------------------


def test_check_warning_below_threshold(tracker: UsageTracker) -> None:
    tracker.set_quota("claude", monthly_requests=100, warn_at_percent=80.0)
    for _ in range(50):
        tracker.record_request("claude")
    warning, msg = tracker.check_warning("claude")
    assert warning is False


def test_check_warning_at_threshold(tracker: UsageTracker) -> None:
    tracker.set_quota("claude", monthly_requests=100, warn_at_percent=80.0)
    for _ in range(80):
        tracker.record_request("claude")
    warning, msg = tracker.check_warning("claude")
    assert warning is True
    assert "80/100" in msg
    assert "20 remaining" in msg


def test_check_warning_at_limit_no_warning(tracker: UsageTracker) -> None:
    """At the limit itself, check_quota catches it — check_warning returns False."""
    tracker.set_quota("claude", monthly_requests=100)
    for _ in range(100):
        tracker.record_request("claude")
    warning, _ = tracker.check_warning("claude")
    assert warning is False  # at limit → exceeded, not warning


def test_check_warning_no_limit(tracker: UsageTracker) -> None:
    tracker.record_request("claude")
    warning, _ = tracker.check_warning("claude")
    assert warning is False


# ---------------------------------------------------------------------------
# Multiple providers
# ---------------------------------------------------------------------------


def test_multiple_providers(tracker: UsageTracker) -> None:
    tracker.set_quota("claude", monthly_requests=100)
    tracker.set_quota("gp", monthly_requests=200)
    tracker.record_request("claude", input_tokens=10)
    tracker.record_request("gp", input_tokens=20)
    tracker.record_request("gp", input_tokens=30)

    c = tracker.get_month_usage("claude")
    g = tracker.get_month_usage("gp")
    assert c is not None and c.requests == 1
    assert g is not None and g.requests == 2
    assert g.input_tokens == 50


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_format_status_with_quota(tracker: UsageTracker) -> None:
    tracker.set_quota("claude", monthly_requests=1000)
    tracker.record_request("claude", input_tokens=5000, output_tokens=2000)
    status = tracker.format_status("claude")
    assert "1 / 1000" in status
    assert "5,000 tokens" in status
    assert "2,000 tokens" in status
    assert "999 remaining" in status


def test_format_status_no_quota(tracker: UsageTracker) -> None:
    tracker.record_request("claude")
    status = tracker.format_status("claude")
    assert "no limit" in status


def test_format_status_unknown_provider(tracker: UsageTracker) -> None:
    status = tracker.format_status("nonexistent")
    assert "no usage data" in status


def test_format_status_all_providers(tracker: UsageTracker) -> None:
    tracker.record_request("alpha")
    tracker.record_request("beta")
    status = tracker.format_status()
    assert "alpha" in status
    assert "beta" in status


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_serialize_deserialize_roundtrip() -> None:
    data = UsageData(
        version=1,
        providers={
            "claude": ProviderUsage(
                quota=Quota(monthly_requests=500, warn_at_percent=90.0),
                months={
                    "2025-07": MonthlyUsage(
                        requests=42,
                        input_tokens=100000,
                        output_tokens=50000,
                        first_request_at="2025-07-01T00:00:00+00:00",
                        last_request_at="2025-07-17T12:00:00+00:00",
                    )
                },
            )
        },
    )
    raw = _serialize(data)
    restored = _deserialize(raw)

    assert restored.version == 1
    assert "claude" in restored.providers
    pu = restored.providers["claude"]
    assert pu.quota.monthly_requests == 500
    assert pu.quota.warn_at_percent == 90.0
    mu = pu.months["2025-07"]
    assert mu.requests == 42
    assert mu.input_tokens == 100000
    assert mu.output_tokens == 50000
    assert mu.first_request_at == "2025-07-01T00:00:00+00:00"


def test_serialize_produces_valid_json() -> None:
    data = UsageData()
    raw = _serialize(data)
    text = json.dumps(raw)
    assert json.loads(text) == raw


# ---------------------------------------------------------------------------
# Extension registration
# ---------------------------------------------------------------------------


class FakeEventBus:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, object]] = []

    def emit(self, event: str, payload: object = None) -> None:
        self.emitted.append((event, payload))

    def on(self, event: str):  # noqa: ANN201
        def decorator(fn):  # noqa: ANN001, ANN202
            return fn

        return decorator


class FakeAPI:
    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self._tools: list[tuple[str, object]] = []
        self._commands: dict[str, tuple[str, object]] = {}
        self._prompts: list[str] = []
        self.events = FakeEventBus()

    def on(self, event: str):  # noqa: ANN201
        def decorator(fn):  # noqa: ANN001, ANN202
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return decorator

    def tool(self, name: str, description: str, input_schema: dict):  # noqa: ANN201
        def decorator(fn):  # noqa: ANN001, ANN202
            self._tools.append((name, fn))
            return fn

        return decorator

    def command(self, name: str, *, description: str = ""):  # noqa: ANN201
        def decorator(fn):  # noqa: ANN001, ANN202
            self._commands[name] = (description, fn)
            return fn

        return decorator

    def append_system_prompt(self, text: str) -> None:
        self._prompts.append(text)


class FakeProviderConfig:
    def __init__(self) -> None:
        self.extra: dict = {"quota": {"monthly_requests": 100, "warn_at_percent": 80.0}}


class FakeConfig:
    def __init__(self) -> None:
        self.provider = "test"
        self._provider_cfg = FakeProviderConfig()

    def resolve_provider(self) -> FakeProviderConfig:
        return self._provider_cfg


class FakeSession:
    session_id = "test-session"
    total_input_tokens = 0
    total_output_tokens = 0


class FakeCtx:
    def __init__(self) -> None:
        self.logger = MagicMock()
        self.session = FakeSession()
        self.config = FakeConfig()


def test_register_creates_handlers_tool_and_command() -> None:
    from aar_ext_usage_tracker import register

    api = FakeAPI()
    register(api)

    assert "session_start" in api._handlers
    assert "after_turn" in api._handlers
    assert "session_end" in api._handlers
    assert len(api._tools) == 1
    assert api._tools[0][0] == "usage_status"
    assert "usage" in api._commands
    assert len(api._prompts) == 1


def test_session_start_seeds_quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aar_ext_usage_tracker import register

    # Patch the tracker's default path
    monkeypatch.setattr(
        "aar_ext_usage_tracker._DEFAULT_PATH",
        tmp_path / "usage.json",
    )

    api = FakeAPI()
    register(api)
    ctx = FakeCtx()

    api._handlers["session_start"][0](None, ctx)

    # Verify quota was saved
    raw = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert raw["providers"]["test"]["quota"]["monthly_requests"] == 100


def test_after_turn_records_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aar_ext_usage_tracker import register

    monkeypatch.setattr(
        "aar_ext_usage_tracker._DEFAULT_PATH",
        tmp_path / "usage.json",
    )

    api = FakeAPI()
    register(api)
    ctx = FakeCtx()

    api._handlers["session_start"][0](None, ctx)

    # Simulate a turn with tokens
    ctx.session.total_input_tokens = 500
    ctx.session.total_output_tokens = 200
    api._handlers["after_turn"][0](None, ctx)

    raw = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    month_key = _month_key()
    month = raw["providers"]["test"]["months"][month_key]
    assert month["requests"] == 1
    assert month["input_tokens"] == 500
    assert month["output_tokens"] == 200


def test_after_turn_computes_deltas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aar_ext_usage_tracker import register

    monkeypatch.setattr(
        "aar_ext_usage_tracker._DEFAULT_PATH",
        tmp_path / "usage.json",
    )

    api = FakeAPI()
    register(api)
    ctx = FakeCtx()

    api._handlers["session_start"][0](None, ctx)

    # Turn 1
    ctx.session.total_input_tokens = 500
    ctx.session.total_output_tokens = 200
    api._handlers["after_turn"][0](None, ctx)

    # Turn 2
    ctx.session.total_input_tokens = 1200
    ctx.session.total_output_tokens = 400
    api._handlers["after_turn"][0](None, ctx)

    raw = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    month_key = _month_key()
    month = raw["providers"]["test"]["months"][month_key]
    assert month["requests"] == 2
    assert month["input_tokens"] == 1200  # 500 + 700
    assert month["output_tokens"] == 400  # 200 + 200


def test_usage_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aar_ext_usage_tracker import register

    monkeypatch.setattr(
        "aar_ext_usage_tracker._DEFAULT_PATH",
        tmp_path / "usage.json",
    )

    api = FakeAPI()
    register(api)
    ctx = FakeCtx()

    api._handlers["session_start"][0](None, ctx)
    ctx.session.total_input_tokens = 100
    ctx.session.total_output_tokens = 50
    api._handlers["after_turn"][0](None, ctx)

    _, handler = api._commands["usage"]
    result = handler("", ctx)
    assert "test" in result
    assert "1 / 100" in result
