"""Tests for the mcp-tools extension."""
from __future__ import annotations

from unittest.mock import MagicMock
from aar_ext_mcp_tools import register


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


def test_register_sets_up_handlers():
    api = FakeAPI()
    register(api)
    assert "session_start" in api._handlers
    assert len(api._prompts) == 1


def test_system_prompt_mentions_mcp():
    api = FakeAPI()
    register(api)
    assert "mcp" in api._prompts[0].lower()
