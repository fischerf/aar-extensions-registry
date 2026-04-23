"""End-to-end tests for the shadow-branching extension.

The tests drive a real git repo in ``tmp_path`` so we exercise actual git
semantics (branch renames, squash merges, conflict detection, etc.) rather
than mocking git out. That matters because the protocol lives or dies by the
real behaviour of ``git reset --hard``, ``git branch -m`` and
``git merge --squash``.
"""
from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aar_ext_shadow_branching import (
    _list_branches,
    _short_hash,
    register,
)


# ---------------------------------------------------------------------------
# Fake ExtensionAPI / ExtensionContext — match the shape of the real API just
# well enough to let the extension register and fire handlers.
# ---------------------------------------------------------------------------


class FakeAPI:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Callable]] = {}
        self.commands: dict[str, tuple[str, Callable]] = {}
        self.system_prompt_parts: list[str] = []

    def on(self, event: str) -> Callable:
        def deco(fn: Callable) -> Callable:
            self.handlers.setdefault(event, []).append(fn)
            return fn

        return deco

    def command(self, name: str, *, description: str = "") -> Callable:
        def deco(fn: Callable) -> Callable:
            self.commands[name] = (description, fn)
            return fn

        return deco

    def tool(self, *args: Any, **kwargs: Any) -> Callable:  # unused but keeps parity
        def deco(fn: Callable) -> Callable:
            return fn

        return deco

    def append_system_prompt(self, text: str) -> None:
        self.system_prompt_parts.append(text)


class FakeSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.metadata: dict[str, Any] = {}


class FakeCtx:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.logger = logging.getLogger(f"aar.ext.shadow_branching.test.{session.session_id}")


# ---------------------------------------------------------------------------
# Helpers for driving the repo
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    )


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-b", "main", cwd=path)
    _git("config", "user.name", "Test", cwd=path)
    _git("config", "user.email", "test@example.com", cwd=path)
    _git("config", "commit.gpgsign", "false", cwd=path)
    (path / "README.md").write_text("initial\n", encoding="utf-8")
    _git("add", "README.md", cwd=path)
    _git("commit", "-m", "initial", cwd=path)


def _fire_tool_result(api: FakeAPI, ctx: FakeCtx, tool_name: str = "write_file") -> None:
    event = SimpleNamespace(tool_name=tool_name)
    for handler in api.handlers.get("tool_result", []):
        handler(event, ctx)


def _run_cmd(api: FakeAPI, name: str, args: str, ctx: FakeCtx) -> None:
    _, handler = api.commands[name]
    handler(args, ctx)


def _session_state(ctx: FakeCtx) -> dict[str, Any]:
    return ctx.session.metadata["shadow_branching"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "project"
    _init_repo(path)
    monkeypatch.chdir(path)
    return path


@pytest.fixture
def session_api(repo: Path) -> tuple[FakeAPI, FakeCtx, Path]:
    """A started session (session_start handler fired) on an isolated repo."""
    api = FakeAPI()
    register(api)
    session = FakeSession(session_id="s1")
    ctx = FakeCtx(session)
    for handler in api.handlers["session_start"]:
        handler(None, ctx)
    return api, ctx, repo


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_register_sets_up_handlers_and_commands() -> None:
    api = FakeAPI()
    register(api)

    assert "session_start" in api.handlers
    assert "tool_result" in api.handlers
    for cmd in ("undo", "revert", "fork", "switch", "forks", "done"):
        assert cmd in api.commands, f"expected /{cmd} to be registered"
    assert any("shadow-branching" in part for part in api.system_prompt_parts)


def test_session_start_creates_shadow_and_anchor(session_api: tuple[FakeAPI, FakeCtx, Path]) -> None:
    _api, ctx, repo = session_api

    assert _short_hash("HEAD", cwd=repo)
    branches = _list_branches("aar/session-*", cwd=repo)
    assert branches == ["aar/session-s1"]

    # aar-init anchor present
    log = _git("log", "--oneline", cwd=repo).stdout
    assert "aar-init: base=main" in log

    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1"
    assert st["original_branch"] == "main"
    assert st["enabled"] is True
    assert st["fork_counter"] == 0
    assert st["turn_counter"] == 0


def test_session_start_without_git_uses_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "no_git"
    plain.mkdir()
    monkeypatch.chdir(plain)

    api = FakeAPI()
    register(api)
    ctx = FakeCtx(FakeSession("s0"))
    api.handlers["session_start"][0](None, ctx)

    assert (plain / ".aar_backups").is_dir()
    st = _session_state(ctx)
    assert st["enabled"] is False
    assert st["mode"] == "fallback"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _write_and_checkpoint(
    api: FakeAPI, ctx: FakeCtx, repo: Path, filename: str, content: str
) -> None:
    (repo / filename).write_text(content, encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")


def test_tool_result_creates_checkpoint(session_api) -> None:
    api, ctx, repo = session_api

    _write_and_checkpoint(api, ctx, repo, "a.txt", "hello")

    st = _session_state(ctx)
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1
    assert st["checkpoints"][0]["tool"] == "write_file"

    log = _git("log", "--oneline", cwd=repo).stdout
    assert "aar-auto: write_file turn-1" in log


def test_tool_result_no_changes_no_checkpoint(session_api) -> None:
    api, ctx, _repo = session_api
    _fire_tool_result(api, ctx, tool_name="read_file")

    st = _session_state(ctx)
    assert st["turn_counter"] == 0
    assert st["checkpoints"] == []


def test_sensitive_file_warning(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api
    (repo / ".env").write_text("SECRET=1\n")

    caplog.set_level(logging.WARNING)
    _fire_tool_result(api, ctx, tool_name="write_file")

    assert any("sensitive" in rec.getMessage() for rec in caplog.records)
    # Still committed, but with a loud warning.
    st = _session_state(ctx)
    assert st["turn_counter"] == 1


# ---------------------------------------------------------------------------
# /undo
# ---------------------------------------------------------------------------


def test_undo_reverts_one_checkpoint(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    _run_cmd(api, "undo", "", ctx)

    # b.txt is gone after reset
    assert not (repo / "b.txt").exists()
    assert (repo / "a.txt").exists()

    st = _session_state(ctx)
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1


def test_undo_n_reverts_multiple(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")

    _run_cmd(api, "undo", "2", ctx)

    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["turn_counter"] == 1


def test_undo_refuses_with_dirty_tree(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    # dirty the tree without committing
    (repo / "dirty.txt").write_text("WIP", encoding="utf-8")

    caplog.set_level(logging.WARNING)
    _run_cmd(api, "undo", "", ctx)

    assert (repo / "dirty.txt").exists()  # still there — not reset
    assert any("uncommitted changes" in rec.getMessage() for rec in caplog.records)


def test_undo_force_discards_dirty_tree(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    (repo / "dirty.txt").write_text("WIP", encoding="utf-8")

    _run_cmd(api, "undo", "1 --force", ctx)

    assert not (repo / "dirty.txt").exists()
    assert not (repo / "b.txt").exists()


def test_undo_beyond_available_checkpoints(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    caplog.set_level(logging.WARNING)
    _run_cmd(api, "undo", "5", ctx)

    assert any("cannot undo" in rec.getMessage() for rec in caplog.records)
    st = _session_state(ctx)
    assert st["turn_counter"] == 1


def test_revert_is_alias_of_undo(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    _run_cmd(api, "revert", "1", ctx)

    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()


# ---------------------------------------------------------------------------
# /fork
# ---------------------------------------------------------------------------


def test_fork_preserves_current_and_starts_fresh(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    _run_cmd(api, "fork", "", ctx)

    branches = set(_list_branches("aar/session-*", cwd=repo))
    assert "aar/session-s1" in branches  # new active
    assert "aar/session-s1-fork-1" in branches  # preserved

    # Still on the aar/session-s1 branch
    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "aar/session-s1"

    # a.txt is still present (forked FROM HEAD keeps history identical)
    assert (repo / "a.txt").exists()

    st = _session_state(ctx)
    assert st["fork_counter"] == 1


def test_fork_back_n_rewinds(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")

    _run_cmd(api, "fork", "2", ctx)

    # Only a.txt should remain on the active branch
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()

    # Preserved branch still has all three
    _git("checkout", "aar/session-s1-fork-1", cwd=repo)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert (repo / "c.txt").exists()


def test_fork_of_fork_of_fork_numbers_monotonically(session_api) -> None:
    """The central scenario: fork -> work -> fork -> work -> fork. Fork numbering
    is derived from the branches already on disk, so we must end with four
    distinct aar/session-s1 branches (one active + three preserved)."""
    api, ctx, repo = session_api

    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")  # turn 1
    _run_cmd(api, "fork", "", ctx)  # preserve -> fork-1, active stays at turn 1

    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")  # turn 2 (on fresh branch)
    _run_cmd(api, "fork", "", ctx)  # preserve -> fork-2

    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")  # turn 3
    _run_cmd(api, "fork", "", ctx)  # preserve -> fork-3

    branches = set(_list_branches("aar/session-*", cwd=repo))
    assert branches == {
        "aar/session-s1",
        "aar/session-s1-fork-1",
        "aar/session-s1-fork-2",
        "aar/session-s1-fork-3",
    }

    st = _session_state(ctx)
    assert st["fork_counter"] == 3

    # Each preserved fork should hold the cumulative files at the time it was
    # forked — fork-1 has only a.txt, fork-2 has a+b, fork-3 has a+b+c.
    _git("checkout", "aar/session-s1-fork-1", cwd=repo)
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()

    _git("checkout", "aar/session-s1-fork-2", cwd=repo)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()

    _git("checkout", "aar/session-s1-fork-3", cwd=repo)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert (repo / "c.txt").exists()


def test_fork_beyond_checkpoints_is_refused(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    caplog.set_level(logging.WARNING)
    _run_cmd(api, "fork", "9", ctx)

    assert any("cannot fork" in rec.getMessage() for rec in caplog.records)
    branches = _list_branches("aar/session-s1*", cwd=repo)
    assert branches == ["aar/session-s1"]  # unchanged


# ---------------------------------------------------------------------------
# /switch
# ---------------------------------------------------------------------------


def test_switch_between_forks(session_api) -> None:
    """After fork-of-fork-of-fork, /switch can hop to any preserved branch and
    the state (turn counter, checkpoints) is reconstructed from git history."""
    api, ctx, repo = session_api

    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")

    # Jump to the first fork using a bare numeric shorthand
    _run_cmd(api, "switch", "1", ctx)
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1-fork-1"
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1

    # Jump to fork-2 using the fork-K shorthand
    _run_cmd(api, "switch", "fork-2", ctx)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1-fork-2"
    assert st["turn_counter"] == 2

    # Jump back to the live branch using the full name
    _run_cmd(api, "switch", "aar/session-s1", ctx)
    assert (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1"
    assert st["turn_counter"] == 3


def test_switch_refuses_dirty_tree(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    (repo / "dirty.txt").write_text("WIP")

    caplog.set_level(logging.WARNING)
    _run_cmd(api, "switch", "fork-1", ctx)

    assert any("uncommitted changes" in rec.getMessage() for rec in caplog.records)
    # We did not switch
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1"


def test_switch_unknown_branch(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, _repo = session_api
    caplog.set_level(logging.WARNING)
    _run_cmd(api, "switch", "nonexistent", ctx)
    assert any("does not exist" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# /forks listing
# ---------------------------------------------------------------------------


def test_forks_lists_all_branches_and_marks_active(
    session_api, caplog: pytest.LogCaptureFixture
) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _run_cmd(api, "fork", "", ctx)

    caplog.set_level(logging.INFO)
    _run_cmd(api, "forks", "", ctx)

    msgs = [rec.getMessage() for rec in caplog.records]
    joined = "\n".join(msgs)
    assert "aar/session-s1" in joined
    assert "aar/session-s1-fork-1" in joined
    assert "aar/session-s1-fork-2" in joined
    # active marker present on exactly one line
    active_lines = [m for m in msgs if "(active)" in m]
    assert len(active_lines) == 1
    assert "aar/session-s1" in active_lines[0] and "fork-" not in active_lines[0]


# ---------------------------------------------------------------------------
# /done — squash merge, including conflict handling
# ---------------------------------------------------------------------------


def test_done_squashes_shadow_into_base(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    _run_cmd(api, "done", "finishing up", ctx)

    # Now on main with a single new commit
    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "main"
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()

    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "finishing up" in log
    # No aar-auto commits on main (they were squashed)
    assert "aar-auto:" not in log


def test_done_refuses_with_remaining_forks(
    session_api, caplog: pytest.LogCaptureFixture
) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    caplog.set_level(logging.INFO)
    _run_cmd(api, "done", "", ctx)
    # still on the shadow branch because /done aborted
    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "aar/session-s1"
    assert any("fork branches still exist" in rec.getMessage() for rec in caplog.records)


def test_done_with_yes_proceeds_despite_forks(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    _run_cmd(api, "done", "--yes merged", ctx)

    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "main"
    # b.txt from the active shadow is on main, but fork branches still exist
    assert (repo / "b.txt").exists()
    branches = _list_branches("aar/session-*", cwd=repo)
    assert "aar/session-s1-fork-1" in branches


def test_done_aborts_on_merge_conflict(
    session_api, caplog: pytest.LogCaptureFixture
) -> None:
    api, ctx, repo = session_api

    # Diverge: edit shared file on shadow
    _write_and_checkpoint(api, ctx, repo, "README.md", "shadow version\n")

    # Meanwhile, simulate a concurrent change on main
    _git("checkout", "main", cwd=repo)
    (repo / "README.md").write_text("main version\n", encoding="utf-8")
    _git("commit", "-am", "conflicting change on main", cwd=repo)

    # Go back to shadow to trigger /done
    _git("checkout", "aar/session-s1", cwd=repo)

    caplog.set_level(logging.WARNING)
    _run_cmd(api, "done", "--yes", ctx)

    # No new merge commit — we are on main with a conflict marker present
    assert any(
        "merge conflicts" in rec.getMessage() for rec in caplog.records
    ), "expected conflict warning"
    # Check that the working tree actually shows conflicts
    status = _git("status", "--porcelain", cwd=repo).stdout
    assert "UU" in status or "AA" in status or "DD" in status or "README.md" in status


def test_done_no_message_uses_default(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    _run_cmd(api, "done", "", ctx)

    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "aar: squashed session s1" in log


# ---------------------------------------------------------------------------
# Session resume — metadata + branch reconstruction
# ---------------------------------------------------------------------------


def test_resume_existing_session_reconstructs_state(repo: Path) -> None:
    # First pass: start a session, make work, don't /done
    api1 = FakeAPI()
    register(api1)
    ctx1 = FakeCtx(FakeSession("resume1"))
    api1.handlers["session_start"][0](None, ctx1)

    (repo / "a.txt").write_text("one", encoding="utf-8")
    _fire_tool_result(api1, ctx1, tool_name="write_file")
    (repo / "b.txt").write_text("two", encoding="utf-8")
    _fire_tool_result(api1, ctx1, tool_name="edit_file")

    # Second pass: new API instance, same session id — should resume branch.
    api2 = FakeAPI()
    register(api2)
    ctx2 = FakeCtx(FakeSession("resume1"))
    api2.handlers["session_start"][0](None, ctx2)

    st = _session_state(ctx2)
    assert st["shadow_branch"] == "aar/session-resume1"
    assert st["original_branch"] == "main"
    assert st["turn_counter"] == 2
    assert len(st["checkpoints"]) == 2

    # Further work should continue numbering from 3.
    (repo / "c.txt").write_text("three", encoding="utf-8")
    _fire_tool_result(api2, ctx2, tool_name="write_file")
    st2 = _session_state(ctx2)
    assert st2["turn_counter"] == 3

    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "aar-auto: write_file turn-3" in log


def test_session_start_leaves_other_sessions_untouched(repo: Path) -> None:
    # Simulate a stale shadow branch from a prior session
    _git("checkout", "-b", "aar/session-zzzold", cwd=repo)
    _git("commit", "--allow-empty", "-m", "aar-init: base=main", cwd=repo)
    _git("checkout", "main", cwd=repo)

    api = FakeAPI()
    register(api)
    ctx = FakeCtx(FakeSession("fresh1"))
    api.handlers["session_start"][0](None, ctx)

    branches = set(_list_branches("aar/session-*", cwd=repo))
    assert "aar/session-zzzold" in branches
    assert "aar/session-fresh1" in branches


# ---------------------------------------------------------------------------
# Sensitive-file detection — direct helper check
# ---------------------------------------------------------------------------


def test_sensitive_detector_flags_expected_paths() -> None:
    from aar_ext_shadow_branching import _flag_sensitive

    # "?? path" lines from git status --porcelain
    output = (
        "?? .env\n"
        "?? src/main.py\n"
        " M secrets/id_rsa\n"
        "A  config/credentials.json\n"
        "?? docs/README.md\n"
    )
    flagged = _flag_sensitive(output)
    assert ".env" in flagged
    assert "secrets/id_rsa" in flagged
    assert "config/credentials.json" in flagged
    assert "src/main.py" not in flagged
    assert "docs/README.md" not in flagged
