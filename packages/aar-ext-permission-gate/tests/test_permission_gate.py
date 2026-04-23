"""Tests for the permission-gate extension."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from aar_ext_permission_gate import register, DANGEROUS_PATTERNS


class FakeAPI:
    def __init__(self):
        self._handlers = {}
        self._prompts = []

    def on(self, event):
        def decorator(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn
        return decorator

    def append_system_prompt(self, text):
        self._prompts.append(text)

    @staticmethod
    def block(reason):
        return ("BLOCK", reason)


class FakeEvent:
    def __init__(self, tool_name="bash", arguments=None):
        self.tool_name = tool_name
        self.arguments = arguments or {}


class FakeCtx:
    def __init__(self):
        self.logger = MagicMock()


def test_register_sets_up_handlers():
    api = FakeAPI()
    register(api)
    assert "tool_call" in api._handlers
    assert "session_start" in api._handlers
    assert len(api._prompts) == 1


def test_blocks_rm_rf():
    api = FakeAPI()
    register(api)
    handler = api._handlers["tool_call"][0]
    event = FakeEvent(arguments={"command": "rm -rf /"})
    result = handler(event, FakeCtx())
    assert result is not None
    assert result[0] == "BLOCK"


def test_allows_safe_commands():
    api = FakeAPI()
    register(api)
    handler = api._handlers["tool_call"][0]
    event = FakeEvent(arguments={"command": "ls -la"})
    result = handler(event, FakeCtx())
    assert result is None


def test_ignores_non_bash_tools():
    api = FakeAPI()
    register(api)
    handler = api._handlers["tool_call"][0]
    event = FakeEvent(tool_name="read_file", arguments={"command": "rm -rf /"})
    result = handler(event, FakeCtx())
    assert result is None


@pytest.mark.parametrize("pattern", DANGEROUS_PATTERNS)
def test_all_patterns_blocked(pattern):
    api = FakeAPI()
    register(api)
    handler = api._handlers["tool_call"][0]
    event = FakeEvent(arguments={"command": f"echo; {pattern}; echo done"})
    result = handler(event, FakeCtx())
    assert result is not None
