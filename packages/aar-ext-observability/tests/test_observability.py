"""Tests for the observability extension."""
from __future__ import annotations

from unittest.mock import MagicMock
from collections import defaultdict
from aar_ext_observability import register


class FakeEventBus:
    def __init__(self):
        self.emitted = []

    def emit(self, event, payload=None):
        self.emitted.append((event, payload))

    def on(self, event):
        def decorator(fn):
            return fn
        return decorator


class FakeAPI:
    def __init__(self):
        self._handlers = {}
        self._tools = []
        self._prompts = []
        self.events = FakeEventBus()

    def on(self, event):
        def decorator(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn
        return decorator

    def tool(self, name, description, input_schema):
        def decorator(fn):
            self._tools.append((name, fn))
            return fn
        return decorator

    def append_system_prompt(self, text):
        self._prompts.append(text)


class FakeSession:
    session_id = "test-123"
    step_count = 3
    total_input_tokens = 1000
    total_output_tokens = 500
    total_tokens = 1500
    total_cost = 0.05


class FakeCtx:
    def __init__(self):
        self.logger = MagicMock()
        self.session = FakeSession()


def test_register_sets_up_handlers():
    api = FakeAPI()
    register(api)
    assert "session_start" in api._handlers
    assert "after_turn" in api._handlers
    assert "session_end" in api._handlers
    assert "error" in api._handlers
    assert len(api._tools) == 1
    assert api._tools[0][0] == "session_stats"


def test_after_turn_emits_metrics():
    api = FakeAPI()
    register(api)
    ctx = FakeCtx()
    # Simulate session start
    api._handlers["session_start"][0](None, ctx)
    # Simulate after_turn
    api._handlers["after_turn"][0](None, ctx)
    assert len(api.events.emitted) == 1
    assert api.events.emitted[0][0] == "metrics:turn"
    payload = api.events.emitted[0][1]
    assert payload["session_id"] == "test-123"
    assert payload["turn"] == 1


def test_session_end_emits_summary():
    api = FakeAPI()
    register(api)
    ctx = FakeCtx()
    api._handlers["session_start"][0](None, ctx)
    api._handlers["session_end"][0](None, ctx)
    assert any(e[0] == "metrics:session" for e in api.events.emitted)


def test_error_emits_event():
    api = FakeAPI()
    register(api)
    ctx = FakeCtx()
    api._handlers["session_start"][0](None, ctx)

    class FakeError:
        message = "something broke"
        recoverable = True

    api._handlers["error"][0](FakeError(), ctx)
    assert any(e[0] == "metrics:error" for e in api.events.emitted)


def test_session_stats_tool():
    api = FakeAPI()
    register(api)
    ctx = FakeCtx()
    api._handlers["session_start"][0](None, ctx)
    stats_fn = api._tools[0][1]
    result = stats_fn()
    assert "turns" in result
