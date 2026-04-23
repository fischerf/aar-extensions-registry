"""Tests for the protected-paths extension."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from aar_ext_protected_paths import register, PROTECTED_PATTERNS, _is_protected


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
    def __init__(self, tool_name="write_file", arguments=None):
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


def test_blocks_env_write():
    api = FakeAPI()
    register(api)
    handler = api._handlers["tool_call"][0]
    event = FakeEvent(arguments={"path": "project/.env"})
    result = handler(event, FakeCtx())
    assert result is not None
    assert result[0] == "BLOCK"


def test_blocks_pem_write():
    api = FakeAPI()
    register(api)
    handler = api._handlers["tool_call"][0]
    event = FakeEvent(arguments={"path": "certs/server.pem"})
    result = handler(event, FakeCtx())
    assert result is not None


def test_allows_normal_file():
    api = FakeAPI()
    register(api)
    handler = api._handlers["tool_call"][0]
    event = FakeEvent(arguments={"path": "src/main.py"})
    result = handler(event, FakeCtx())
    assert result is None


def test_ignores_read_tools():
    api = FakeAPI()
    register(api)
    handler = api._handlers["tool_call"][0]
    event = FakeEvent(tool_name="read_file", arguments={"path": ".env"})
    result = handler(event, FakeCtx())
    assert result is None


def test_is_protected_returns_pattern():
    assert _is_protected("project/.env") is not None
    assert _is_protected("src/main.py") is None


@pytest.mark.parametrize("path", [
    ".env", "app/.env.production", "secrets", "creds/credentials.json",
    "keys/server.pem", "cert.key", "home/.ssh/id_rsa", ".npmrc",
])
def test_sensitive_paths_blocked(path):
    assert _is_protected(path) is not None
