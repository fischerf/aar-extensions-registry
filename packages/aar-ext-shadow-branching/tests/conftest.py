"""Shared fixtures: an isolated temp git repo and a started session.

``test_shadow_branching.py`` defines the fakes (FakeAPI/FakeSession/FakeCtx)
and the repo bootstrap; these fixtures re-export them so other test modules
(the panel tests) can use the same started-session setup without importing
fixtures across modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from test_shadow_branching import _init_repo

    path = tmp_path / "project"
    _init_repo(path)
    monkeypatch.chdir(path)
    return path


@pytest.fixture
def session_api(repo: Path):
    """A started session (session_start handler fired) on an isolated repo."""
    from test_shadow_branching import FakeAPI, FakeCtx, FakeSession, register

    api = FakeAPI()
    register(api)
    session = FakeSession(session_id="s1")
    ctx = FakeCtx(session)
    for handler in api.handlers["session_start"]:
        handler(None, ctx)
    return api, ctx, repo
