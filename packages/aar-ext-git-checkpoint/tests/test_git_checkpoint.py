"""Tests for the git-checkpoint extension."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from aar_ext_git_checkpoint import register, _is_git_repo, _has_changes, _run_git


class FakeAPI:
    def __init__(self):
        self._handlers = {}
        self._tools = []
        self._prompts = []

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


class FakeCtx:
    def __init__(self):
        self.logger = MagicMock()


def test_register_sets_up_handlers():
    api = FakeAPI()
    register(api)
    assert "session_start" in api._handlers
    assert "after_turn" in api._handlers
    assert len(api._tools) == 1
    assert api._tools[0][0] == "git_rollback"


@patch("aar_ext_git_checkpoint._run_git", return_value=(0, "true"))
def test_is_git_repo_true(mock_git):
    assert _is_git_repo() is True


@patch("aar_ext_git_checkpoint._run_git", return_value=(1, ""))
def test_is_git_repo_false(mock_git):
    assert _is_git_repo() is False


@patch("aar_ext_git_checkpoint._run_git", return_value=(0, " M file.py\n?? new.txt"))
def test_has_changes_true(mock_git):
    assert _has_changes() is True


@patch("aar_ext_git_checkpoint._run_git", return_value=(0, ""))
def test_has_changes_false(mock_git):
    assert _has_changes() is False


def test_rollback_tool_not_in_repo():
    api = FakeAPI()
    register(api)
    # Simulate session_start with no git repo
    with patch("aar_ext_git_checkpoint._is_git_repo", return_value=False):
        api._handlers["session_start"][0](None, FakeCtx())
    rollback_fn = api._tools[0][1]
    result = rollback_fn(steps=1)
    assert "not in a git repository" in result
