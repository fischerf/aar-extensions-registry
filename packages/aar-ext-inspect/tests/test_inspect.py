from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from agent.extensions.api import ExtensionAPI, ExtensionContext

# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


def _make_ctx(session: Any) -> ExtensionContext:
    return ExtensionContext(
        session=session,
        config=SimpleNamespace(provider=SimpleNamespace(name="ollama", model="llama3")),
        signal=__import__("asyncio").Event(),
        logger=logging.getLogger("test.inspect"),
    )


def _fake_session(session_id: str = "abc123", events: list | None = None) -> Any:
    return SimpleNamespace(
        session_id=session_id,
        trace_id="trace-xyz",
        step_count=3,
        state="IDLE",
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost=0.001,
        events=events or [],
    )


def _register() -> tuple[Any, Any]:
    """Return (api, inspect_fn)."""
    api = ExtensionAPI(name="inspect_test")
    mod = importlib.import_module("aar_ext_inspect")
    mod.register(api)
    _, fn = api._commands["inspect"]
    return api, fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_and_command_present() -> None:
    """register() must populate the API with an 'inspect' slash-command."""
    api, fn = _register()
    assert "inspect" in api._commands
    desc, _ = api._commands["inspect"]
    assert callable(fn)
    assert isinstance(desc, str) and desc


def test_inspect_returns_string() -> None:
    """inspect_command must return a non-empty string (not None)."""
    _, fn = _register()
    ctx = _make_ctx(_fake_session())
    result = fn("", ctx)
    assert isinstance(result, str), "Expected a str return value for TUI/CLI display"
    assert result.strip(), "Return value must not be empty"


def test_inspect_contains_session_id() -> None:
    """The returned report must include the session ID from ctx.session."""
    _, fn = _register()
    session = _fake_session(session_id="deadbeef1234")
    ctx = _make_ctx(session)
    result = fn("", ctx)
    assert "deadbeef1234" in result, "Report must contain the actual session ID"


def test_inspect_session_id_matches_loaded_session() -> None:
    """When a session is loaded and the context is updated, /inspect must show
    the loaded session's ID — not a stale bootstrap ID."""
    _, fn = _register()

    bootstrap_session = _fake_session(session_id="bootstrap-000")
    loaded_session = _fake_session(session_id="loaded-session-999", events=[])

    # Simulate what the CLI does: create context with bootstrap, then update_session
    from dataclasses import replace

    ctx = _make_ctx(bootstrap_session)
    ctx = replace(ctx, session=loaded_session)

    result = fn("", ctx)
    assert "loaded-session-999" in result, "Report must reflect the updated (loaded) session"
    assert "bootstrap-000" not in result, "Report must not show the stale bootstrap session ID"


def test_inspect_report_structure() -> None:
    """Report must contain expected section headers and provider info."""
    _, fn = _register()
    ctx = _make_ctx(_fake_session())
    result = fn("", ctx)
    assert "=== Session Inspect Report ===" in result
    assert "=== End Report ===" in result
    assert "Provider:" in result
    assert "ollama" in result


def test_inspect_verbose_includes_event_detail() -> None:
    """Passing 'verbose' includes the event detail section."""
    _, fn = _register()
    ev = SimpleNamespace(type="user_message", content="hello")
    ctx = _make_ctx(_fake_session(events=[ev]))
    result = fn("verbose", ctx)
    assert "Event detail" in result


def test_inspect_counts_events_correctly() -> None:
    """Event counters in the report must match the actual events list."""
    _, fn = _register()
    events = [
        SimpleNamespace(type="user_message", content="hi"),
        SimpleNamespace(type="assistant_message", content="hello"),
        SimpleNamespace(type="tool_call", tool_name="read_file"),
        SimpleNamespace(type="tool_result", tool_name="read_file"),
    ]
    ctx = _make_ctx(_fake_session(events=events))
    result = fn("", ctx)
    assert "User messages     : 1" in result
    assert "Assistant messages: 1" in result
    assert "Tool calls        : 1" in result
    assert "Tool results      : 1" in result


def test_inspect_also_logs(caplog: pytest.LogCaptureFixture) -> None:
    """The report must still be emitted via ctx.logger (log trail preserved)."""
    _, fn = _register()
    ctx = _make_ctx(_fake_session(session_id="log-check"))
    with caplog.at_level(logging.INFO, logger="test.inspect"):
        fn("", ctx)
    assert any("log-check" in rec.getMessage() for rec in caplog.records)


def test_inspect_no_session_returns_message() -> None:
    """When session is None, a friendly 'no active session' string is returned."""
    _, fn = _register()
    ctx = _make_ctx(None)
    result = fn("", ctx)
    assert isinstance(result, str)
    assert "no active session" in result.lower()


def test_inspect_session_with_no_id_returns_message() -> None:
    """When the session object has no session_id (bootstrap placeholder), return
    a friendly message rather than an empty or stale report."""
    _, fn = _register()
    # Simulate the internal bootstrap Session that has no id yet
    bare_session = SimpleNamespace(
        session_id=None,
        trace_id=None,
        step_count=0,
        state="IDLE",
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost=0.0,
        events=[],
    )
    ctx = _make_ctx(bare_session)
    result = fn("", ctx)
    assert isinstance(result, str)
    assert "no active session" in result.lower()


def test_inspect_bootstrap_stale_id_not_shown_after_update() -> None:
    """After update_session() is called (simulated by replacing ctx.session),
    the report must show the real loaded session's ID, never the bootstrap one."""
    _, fn = _register()
    from dataclasses import replace

    bootstrap = SimpleNamespace(
        session_id="stale-bootstrap-id",
        trace_id=None,
        step_count=0,
        state="IDLE",
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost=0.0,
        events=[],
    )
    real_session = _fake_session(session_id="real-loaded-id-9999")

    # Start with stale bootstrap, then simulate the transport calling update_session
    ctx = _make_ctx(bootstrap)
    ctx = replace(ctx, session=real_session)

    result = fn("", ctx)
    assert "real-loaded-id-9999" in result
    assert "stale-bootstrap-id" not in result
