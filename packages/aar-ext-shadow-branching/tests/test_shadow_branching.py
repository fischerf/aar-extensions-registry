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
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


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


def _fire_session_end(api: FakeAPI, ctx: FakeCtx) -> None:
    from types import SimpleNamespace

    event = SimpleNamespace(action="ended")
    for handler in api.handlers.get("session_end", []):
        handler(event, ctx)


def _run_cmd(api: FakeAPI, name: str, args: str, ctx: FakeCtx) -> str | None:
    _, handler = api.commands[name]
    return handler(args, ctx)


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


def test_session_start_creates_shadow_and_anchor(
    session_api: tuple[FakeAPI, FakeCtx, Path],
) -> None:
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


def test_session_end_commits_dirty_jsonl(session_api) -> None:
    """session_end must auto-commit files left dirty after store.save() (e.g. JSONL).

    The transport writes the session JSONL *after* agent.run() returns, which is
    after every extension event including tool_result.  Without the session_end
    hook the working tree stays dirty between turns and /fork or /switch blocks."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "content")

    # Simulate the transport writing the session JSONL after agent.run()
    (repo / "session.jsonl").write_text('{"session_id": "s1"}\n')

    # Verify it is actually dirty before session_end fires
    rc, status, _ = (
        subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True),
        None,
        None,
    )
    assert rc.stdout.strip(), "Expected dirty tree before session_end"

    _fire_session_end(api, ctx)

    # After session_end the tree must be clean
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert not result.stdout.strip(), "Working tree must be clean after session_end commit"

    # The commit must use the aar-meta: prefix (not aar-auto:, not a checkpoint)
    log = _git("log", "--oneline", cwd=repo).stdout
    assert "aar-meta: session sync" in log

    # turn_counter and checkpoints must be unchanged (it's not a checkpoint)
    st = _session_state(ctx)
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1


def test_session_end_no_op_when_clean(session_api) -> None:
    """session_end must not create an empty commit when the tree is already clean."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "content")

    log_before = _git("log", "--oneline", cwd=repo).stdout
    _fire_session_end(api, ctx)
    log_after = _git("log", "--oneline", cwd=repo).stdout

    assert log_before == log_after, "No new commit should be created on a clean tree"


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

    result = _run_cmd(api, "undo", "", ctx)

    assert not (repo / "b.txt").exists()
    assert (repo / "a.txt").exists()
    st = _session_state(ctx)
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1
    assert result is not None
    assert result.startswith("↩")
    assert "1 checkpoint" in result


def test_undo_n_reverts_multiple(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")

    result = _run_cmd(api, "undo", "2", ctx)

    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1
    assert result is not None
    assert result.startswith("↩")
    assert "2 checkpoint" in result


def test_fork_auto_commits_pending(session_api, caplog: pytest.LogCaptureFixture) -> None:
    """Pending changes are auto-committed before /fork so git branch rename works."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    # Drop a pending file (session store write) before forking
    (repo / "session.jsonl").write_text('{"event": "tool_result"}\n')

    caplog.set_level(logging.DEBUG)
    _run_cmd(api, "fork", "", ctx)

    branches = set(_list_branches("aar/session-*", cwd=repo))
    assert "aar/session-s1" in branches
    assert "aar/session-s1-fork-1" in branches

    # session.jsonl must be committed on the preserved fork branch
    rc = subprocess.run(
        ["git", "show", "aar/session-s1-fork-1:session.jsonl"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, (
        "session.jsonl should have been auto-committed onto the preserved fork"
    )


def test_done_auto_commits_pending(session_api, caplog: pytest.LogCaptureFixture) -> None:
    """Pending changes are auto-committed before /done so the dirty-tree guard is not hit."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    # Drop a pending file before /done
    (repo / "session.jsonl").write_text('{"event": "done"}\n')

    caplog.set_level(logging.INFO)
    _run_cmd(api, "done", "", ctx)

    # /done squashes and checks out the base branch — if we landed here cleanly
    # there should be no "uncommitted changes present" warning.
    assert not any("uncommitted changes present" in rec.getMessage() for rec in caplog.records)
    # And the squash commit should be on main
    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "main"


def test_undo_refuses_with_dirty_tree(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    # dirty the tree without committing
    (repo / "dirty.txt").write_text("WIP", encoding="utf-8")

    caplog.set_level(logging.WARNING)
    result = _run_cmd(api, "undo", "", ctx)

    assert (repo / "dirty.txt").exists()  # still there — not reset
    assert any("uncommitted changes" in rec.getMessage() for rec in caplog.records)
    assert result is not None
    assert result.startswith("✗")
    assert "uncommitted" in result


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
    result = _run_cmd(api, "undo", "9", ctx)

    assert any("cannot undo" in rec.getMessage() for rec in caplog.records)
    assert result is not None
    assert result.startswith("✗")
    assert "1 checkpoint" in result
    st = _session_state(ctx)
    assert st["turn_counter"] == 1


def test_revert_is_alias_of_undo(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    result = _run_cmd(api, "revert", "", ctx)

    assert not (repo / "b.txt").exists()
    st = _session_state(ctx)
    assert st["turn_counter"] == 1
    assert result is not None
    assert result.startswith("↩")
    assert not (repo / "b.txt").exists()


# ---------------------------------------------------------------------------
# /fork
# ---------------------------------------------------------------------------


def test_fork_preserves_current_and_starts_fresh(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    result = _run_cmd(api, "fork", "", ctx)

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
    assert result is not None
    assert result.startswith("⑂")
    assert "fork-1" in result
    assert "aar/session-s1" in result


def test_fork_back_n_rewinds(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")

    result = _run_cmd(api, "fork", "2", ctx)
    assert result is not None
    assert "rewound 2 checkpoint" in result

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
    result = _run_cmd(api, "fork", "9", ctx)

    assert any("cannot fork" in rec.getMessage() for rec in caplog.records)
    branches = _list_branches("aar/session-s1*", cwd=repo)
    assert branches == ["aar/session-s1"]  # unchanged
    assert result is not None
    assert result.startswith("✗")
    assert "1 checkpoint" in result


# ---------------------------------------------------------------------------
# /switch
# ---------------------------------------------------------------------------


def test_switch_no_args_shows_hint(session_api) -> None:
    """Calling /switch with no args must return a hint with current branch and targets."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    result = _run_cmd(api, "switch", "", ctx)

    assert result is not None
    assert "Current branch:" in result
    assert "aar/session-s1" in result
    assert "aar/session-s1-fork-1" in result
    assert "Usage:" in result
    # Must NOT have switched — still on the active shadow
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1"


def test_switch_main_shorthand_returns_to_canonical_shadow(session_api) -> None:
    """'main', 'active', and 'shadow' keywords all switch back to the canonical
    aar/session-<id> branch from a preserved fork."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    # Switch away to the preserved fork first
    _run_cmd(api, "switch", "fork-1", ctx)
    assert _session_state(ctx)["shadow_branch"] == "aar/session-s1-fork-1"

    # Switch back using 'main'
    result = _run_cmd(api, "switch", "main", ctx)
    assert result is not None
    assert result.startswith("⇄")
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1"
    assert (repo / "b.txt").exists()


def test_switch_active_shorthand(session_api) -> None:
    """'active' keyword resolves to the canonical shadow branch."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)

    _run_cmd(api, "switch", "fork-1", ctx)
    result = _run_cmd(api, "switch", "active", ctx)

    assert result is not None
    assert result.startswith("⇄")
    assert _session_state(ctx)["shadow_branch"] == "aar/session-s1"


def test_switch_shadow_shorthand(session_api) -> None:
    """'shadow' keyword resolves to the canonical shadow branch."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)

    _run_cmd(api, "switch", "fork-1", ctx)
    result = _run_cmd(api, "switch", "shadow", ctx)

    assert result is not None
    assert result.startswith("⇄")
    assert _session_state(ctx)["shadow_branch"] == "aar/session-s1"


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
    result = _run_cmd(api, "switch", "1", ctx)
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1-fork-1"
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1
    assert result is not None
    assert result.startswith("⇄")
    assert "fork-1" in result

    # Jump to fork-2 using the fork-K shorthand
    result = _run_cmd(api, "switch", "fork-2", ctx)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1-fork-2"
    assert st["turn_counter"] == 2
    assert result is not None
    assert result.startswith("⇄")
    assert "fork-2" in result

    # Jump back to the live branch using the full name
    result = _run_cmd(api, "switch", "aar/session-s1", ctx)
    assert (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1"
    assert st["turn_counter"] == 3
    assert result is not None
    assert result.startswith("⇄")


def test_switch_auto_commits_pending_and_succeeds(
    session_api, caplog: pytest.LogCaptureFixture
) -> None:
    """Pending (uncommitted) changes — e.g. session JSONL writes — are auto-committed
    before /switch so the command is not blocked by a dirty working tree."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    # Simulate a session-store write that lands outside of tool_result
    (repo / "session.jsonl").write_text('{"event": "message"}\n')

    caplog.set_level(logging.DEBUG)
    _run_cmd(api, "switch", "fork-1", ctx)

    # Switch must have succeeded
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1-fork-1"
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()

    # The pending file must have been committed on the shadow branch, not left dirty
    _git("checkout", "aar/session-s1", cwd=repo)
    rc = subprocess.run(
        ["git", "show", "HEAD:session.jsonl"], cwd=repo, capture_output=True, text=True
    )
    assert rc.returncode == 0, "session.jsonl should have been auto-committed"


def test_switch_still_warns_on_truly_dirty_after_auto_commit_failure(
    session_api, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the auto-commit itself fails (e.g. git commit errors out), the dirty-tree
    warning must still be emitted and the switch must not proceed."""
    import aar_ext_shadow_branching as ext

    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    (repo / "session.jsonl").write_text('{"event": "message"}\n')

    original_run_git = ext._run_git

    def _failing_commit(*args, **kwargs):
        if args and args[0] == "commit":
            return 1, "", "simulated commit failure"
        return original_run_git(*args, **kwargs)

    monkeypatch.setattr(ext, "_run_git", _failing_commit)

    caplog.set_level(logging.WARNING)
    _run_cmd(api, "switch", "fork-1", ctx)

    assert any("uncommitted changes" in rec.getMessage() for rec in caplog.records)
    st = _session_state(ctx)
    assert st["shadow_branch"] == "aar/session-s1"


def test_switch_unknown_branch(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, _repo = session_api
    caplog.set_level(logging.WARNING)
    result = _run_cmd(api, "switch", "nonexistent", ctx)
    assert any("does not exist" in rec.getMessage() for rec in caplog.records)
    assert result is not None
    assert result.startswith("✗")
    assert "does not exist" in result


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
    result = _run_cmd(api, "forks", "", ctx)

    # The return value is the canonical output shown in the TUI/CLI
    assert result is not None
    assert "aar/session-s1" in result
    assert "aar/session-s1-fork-1" in result
    assert "aar/session-s1-fork-2" in result
    # active marker present on exactly one line
    active_lines = [line for line in result.splitlines() if "◀ active" in line]
    assert len(active_lines) == 1
    assert "aar/session-s1" in active_lines[0] and "fork-" not in active_lines[0]
    # log still emitted for the record
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("aar/session-s1" in m for m in msgs)


# ---------------------------------------------------------------------------
# /done — squash merge, including conflict handling
# ---------------------------------------------------------------------------


def test_done_squashes_shadow_into_base(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    result = _run_cmd(api, "done", "finishing up", ctx)

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
    assert result is not None
    assert result.startswith("✓")
    assert "main" in result


def test_done_refuses_with_remaining_forks(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    caplog.set_level(logging.INFO)
    result = _run_cmd(api, "done", "", ctx)
    # still on the shadow branch because /done aborted
    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "aar/session-s1"
    assert result is not None
    assert result.startswith("⚠")
    assert "fork" in result
    assert "--yes" in result
    assert any("fork branches still exist" in rec.getMessage() for rec in caplog.records)


def test_done_with_yes_proceeds_despite_forks(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "fork", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    result = _run_cmd(api, "done", "--yes", ctx)

    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "main"
    assert "aar/session-s1-fork-1" in _list_branches("aar/session-s1*", cwd=repo)
    assert result is not None
    assert result.startswith("✓")
    assert "main" in result
    # b.txt from the active shadow is on main, but fork branches still exist
    assert (repo / "b.txt").exists()
    branches = _list_branches("aar/session-*", cwd=repo)
    assert "aar/session-s1-fork-1" in branches


def test_done_aborts_on_merge_conflict(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api

    # Diverge: edit shared file on shadow
    _write_and_checkpoint(api, ctx, repo, "README.md", "shadow version\n")

    # Now edit the same file on main so merge will conflict
    _git("checkout", "main", cwd=repo)
    (repo / "README.md").write_text("main version\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo)
    _git("commit", "-m", "main edit", cwd=repo)
    _git("checkout", "aar/session-s1", cwd=repo)

    caplog.set_level(logging.WARNING)
    result = _run_cmd(api, "done", "--yes", ctx)

    # No new merge commit — we are on main with a conflict marker present
    assert any("merge conflicts" in rec.getMessage() for rec in caplog.records), (
        "expected conflict warning"
    )
    # Check that the working tree actually shows conflicts
    status = _git("status", "--porcelain", cwd=repo).stdout
    assert "UU" in status or "AA" in status or "DD" in status or "README.md" in status
    assert result is not None
    assert result.startswith("✗")
    assert "conflict" in result


def test_done_no_message_uses_default(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "content")

    result = _run_cmd(api, "done", "", ctx)

    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "aar: squashed session s1" in log
    assert result is not None
    assert result.startswith("✓")


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
