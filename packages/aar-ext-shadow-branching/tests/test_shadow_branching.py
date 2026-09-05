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
        self.events: list[Any] = []
        self.step_count: int = 0


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


def _fire_before_turn(api: FakeAPI, ctx: FakeCtx) -> None:
    for handler in api.handlers.get("before_turn", []):
        handler(None, ctx)


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
    assert "before_turn" in api.handlers
    for cmd in ("undo", "revert", "branch", "switch", "branches", "done"):
        assert cmd in api.commands, f"expected /{cmd} to be registered"
    assert any("shadow-branching" in part for part in api.system_prompt_parts)


def test_session_start_creates_shadow_and_anchor(
    session_api: tuple[FakeAPI, FakeCtx, Path],
) -> None:
    _api, ctx, repo = session_api

    assert _short_hash("HEAD", cwd=repo)
    branches = _list_branches("shadow/session-*", cwd=repo)
    assert branches == ["shadow/session-s1"]

    # shadow-init anchor present
    log = _git("log", "--oneline", cwd=repo).stdout
    assert "shadow-init: base=main" in log

    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-s1"
    assert st["original_branch"] == "main"
    assert st["enabled"] is True
    assert st["branch_counter"] == 0
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

    assert (plain / ".shadow_backups").is_dir()
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


def _save_real_session(repo: Path, ctx: FakeCtx, events: list[Any], step_count: int) -> None:
    from agent.core.session import Session as RealSession
    from agent.memory.session_store import SessionStore

    ctx.session.events = list(events)
    ctx.session.step_count = step_count
    store = SessionStore(base_dir=repo / ".agent" / "sessions")
    store.save(
        RealSession(
            session_id=ctx.session.session_id,
            events=ctx.session.events,
            step_count=step_count,
            metadata=ctx.session.metadata,
        )
    )


def test_tool_result_creates_checkpoint(session_api) -> None:
    api, ctx, repo = session_api

    _write_and_checkpoint(api, ctx, repo, "a.txt", "hello")

    st = _session_state(ctx)
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1
    assert st["checkpoints"][0]["tool"] == "write_file"

    log = _git("log", "--oneline", cwd=repo).stdout
    assert "shadow-auto: write_file turn-1" in log


def test_session_end_commits_dirty_jsonl(session_api) -> None:
    """session_end must auto-commit files left dirty after store.save() (e.g. JSONL).

    The transport writes the session JSONL *after* agent.run() returns, which is
    after every extension event including tool_result.  Without the session_end
    hook the working tree stays dirty between turns and /branch or /switch blocks."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "content")

    # Simulate the transport writing the session JSONL after agent.run()
    (repo / "session.jsonl").write_text('{"session_id": "s1"}\n')

    # Verify it is actually dirty before session_end fires
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert status.stdout.strip(), "Expected dirty tree before session_end"

    _fire_session_end(api, ctx)

    # After session_end the tree must be clean
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert not result.stdout.strip(), "Working tree must be clean after session_end commit"

    # The commit must use the shadow-meta: prefix (not shadow-auto:, not a checkpoint)
    log = _git("log", "--oneline", cwd=repo).stdout
    assert "shadow-meta: session sync" in log

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


def test_before_turn_commits_dirty_jsonl_between_turns(session_api) -> None:
    """before_turn must sweep the session JSONL written after the previous turn.

    The transport writes the JSONL outside the agent loop, so between two
    tool-free turns the working tree is dirty with nothing but the session file.
    Without the before_turn hook /branch and /switch would be blocked."""
    api, ctx, repo = session_api

    # Simulate the transport writing the session JSONL after a tool-free turn
    # (no tool_result fired — just an LLM answer).
    (repo / "session.jsonl").write_text('{"session_id": "s1", "turn": 1}\n')

    # Verify it is actually dirty
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert result.stdout.strip(), "Expected dirty tree before before_turn"

    # Fire before_turn (the next user message arriving)
    _fire_before_turn(api, ctx)

    # Tree must now be clean
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert not result.stdout.strip(), "Working tree must be clean after before_turn sweep"

    # Committed with the shadow-meta: prefix, not as a checkpoint
    log = _git("log", "--oneline", cwd=repo).stdout
    assert "shadow-meta: turn sync" in log

    # turn_counter and checkpoints untouched
    st = _session_state(ctx)
    assert st["turn_counter"] == 0
    assert st["checkpoints"] == []


def test_before_turn_no_op_when_clean(session_api) -> None:
    """before_turn must not create a commit when the tree is already clean."""
    api, ctx, repo = session_api

    log_before = _git("log", "--oneline", cwd=repo).stdout
    _fire_before_turn(api, ctx)
    log_after = _git("log", "--oneline", cwd=repo).stdout

    assert log_before == log_after, "No new commit should be created on a clean tree"


def test_branch_after_tool_free_turns_succeeds(session_api) -> None:
    """/branch must succeed even if the only pending change is the session JSONL
    written after a tool-free turn (no checkpoint exists yet)."""
    api, ctx, repo = session_api

    # Tool-free turn: transport writes the JSONL, then the next turn starts
    (repo / "session.jsonl").write_text('{"session_id": "s1", "turn": 1}\n')
    _fire_before_turn(api, ctx)

    # Now the user issues /branch — must not fail with "dirty tree" or similar
    result = _run_cmd(api, "branch", "", ctx)

    assert result is not None
    assert not result.startswith("✗"), f"Expected success but got: {result}"
    branches = set(_list_branches("shadow/session-*", cwd=repo))
    assert "shadow/session-s1" in branches
    assert "shadow/session-s1-branch-1" in branches


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


def test_undo_skips_shadow_meta_commits(session_api) -> None:
    """/undo counts logical checkpoints, not raw commits at HEAD."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    (repo / "session.jsonl").write_text('{"after": "a"}\n', encoding="utf-8")
    _fire_session_end(api, ctx)

    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    (repo / "session.jsonl").write_text('{"after": "b"}\n', encoding="utf-8")
    _fire_session_end(api, ctx)

    result = _run_cmd(api, "undo", "1", ctx)

    assert result is not None and result.startswith("↩")
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    st = _session_state(ctx)
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1
    log = _git("log", "--oneline", "-1", cwd=repo).stdout
    assert "shadow-meta: session sync" in log


def test_undo_reloads_session_events_to_reverted_checkpoint(repo: Path) -> None:
    from agent.core.events import AssistantMessage, UserMessage

    api = FakeAPI()
    register(api)
    session = FakeSession("undo_reload")
    ctx = FakeCtx(session)
    api.handlers["session_start"][0](None, ctx)

    _write_and_checkpoint(api, ctx, repo, "first.txt", "one")
    _save_real_session(
        repo,
        ctx,
        [UserMessage(content="create first.txt"), AssistantMessage(content="created first.txt")],
        1,
    )

    _write_and_checkpoint(api, ctx, repo, "second.txt", "two")
    _save_real_session(
        repo,
        ctx,
        [
            UserMessage(content="create first.txt"),
            AssistantMessage(content="created first.txt"),
            UserMessage(content="create second.txt"),
            AssistantMessage(content="created second.txt"),
        ],
        2,
    )

    result = _run_cmd(api, "undo", "1", ctx)

    assert result is not None and result.startswith("↩")
    assert (repo / "first.txt").exists()
    assert not (repo / "second.txt").exists()
    contents = " ".join(getattr(e, "content", "") or "" for e in ctx.session.events)
    assert "first.txt" in contents
    assert "second.txt" not in contents
    assert ctx.session.step_count == 1


def test_branch_auto_commits_pending(session_api, caplog: pytest.LogCaptureFixture) -> None:
    """Pending changes are auto-committed before /branch so git branch rename works."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    # Drop a pending file (session store write) before branching
    (repo / "session.jsonl").write_text('{"event": "tool_result"}\n')

    caplog.set_level(logging.DEBUG)
    _run_cmd(api, "branch", "", ctx)

    branches = set(_list_branches("shadow/session-*", cwd=repo))
    assert "shadow/session-s1" in branches
    assert "shadow/session-s1-branch-1" in branches

    # session.jsonl must be committed on the preserved branch
    rc = subprocess.run(
        ["git", "show", "shadow/session-s1-branch-1:session.jsonl"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, (
        "session.jsonl should have been auto-committed onto the preserved branch"
    )


def test_session_store_save_commits_pending_jsonl(session_api, tmp_path: Path) -> None:
    """After ``SessionStore.save()`` writes the session JSONL, the extension
    should immediately commit the file so the tree is never left dirty between
    turns.

    Regression: ``session_end`` fires BEFORE the transport calls
    ``store.save()``, so the extension's ``session_end`` sweep runs when the
    JSONL has not yet been written. Without a save-hook, the tree stays dirty
    until the next ``before_turn`` or slash-command.
    """
    from agent.core.session import Session
    from agent.core.state import AgentState
    from agent.memory.session_store import SessionStore

    api, ctx, repo = session_api

    # Produce at least one real checkpoint so the shadow branch exists.
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    # Now mimic the transport: real SessionStore.save() into the project repo
    # with the SAME session_id the extension was started with.
    store = SessionStore(base_dir=repo / ".agent" / "sessions")
    session = Session(
        session_id="s1",
        run_id="r1",
        trace_id="t1",
        state=AgentState.COMPLETED,
        step_count=1,
        metadata={},
        events=[],
    )
    store.save(session)

    # After save the tree must be clean — the save-hook committed the JSONL.
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert not result.stdout.strip(), (
        f"tree must be clean after store.save(), got: {result.stdout!r}"
    )

    # And the commit itself must carry the shadow-meta: session-saved label.
    log = _git("log", "--oneline", cwd=repo).stdout
    assert "shadow-meta: session-saved" in log, (
        f"expected 'shadow-meta: session-saved' commit, log:\n{log}"
    )


def test_session_store_save_noop_when_session_unknown(session_api, tmp_path: Path) -> None:
    """The save hook must not commit on behalf of sessions that aren't ours.

    A SessionStore save for a session_id that shadow-branching doesn't know
    about should leave the tree in whatever state it was — we don't own it.
    """
    from agent.core.session import Session
    from agent.core.state import AgentState
    from agent.memory.session_store import SessionStore

    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    # Dirty the tree with something the extension has no reason to touch.
    (repo / "stray.txt").write_text("not ours\n", encoding="utf-8")

    store = SessionStore(base_dir=repo / ".agent" / "sessions")
    foreign = Session(
        session_id="NOT_OUR_SESSION",
        run_id="r0",
        trace_id="t0",
        state=AgentState.COMPLETED,
        step_count=0,
        metadata={},
        events=[],
    )
    store.save(foreign)

    # stray.txt must still be untracked — the save hook should not have
    # committed it on behalf of a session it doesn't own.
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    assert "stray.txt" in result.stdout, (
        f"save hook should not have committed on behalf of an unknown session; "
        f"status:\n{result.stdout}"
    )


def test_branch_refuses_when_pre_sync_fails(
    session_api, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the pre-branch sweep fails to stage pending work (e.g. ``git add -A``
    times out on a huge venv/), /branch must refuse to proceed rather than
    silently preserve a branch that does not contain the user's actual work.

    Regression: before the fix, ``_run_git`` swallowed TimeoutExpired, the
    subsequent ``git diff --cached --quiet`` saw an empty index and
    ``_auto_commit_pending`` returned False silently. /branch then branched
    off a tip that was missing the session's real changes.
    """
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    # Pending work the user expects to be preserved by /branch.
    (repo / "file.md").write_text("real work\n", encoding="utf-8")
    (repo / "rss_downloader.py").write_text("# tool output\n", encoding="utf-8")

    # Simulate `git add -A` failing (as if it timed out on a big tree).
    import aar_ext_shadow_branching as ext

    real_run_git = ext._run_git

    def flaky_run_git(*args: str, **kwargs: Any) -> tuple[int, str, str]:
        if len(args) >= 2 and args[0] == "add" and args[1] == "-A":
            return 1, "", "git command timed out"
        return real_run_git(*args, **kwargs)

    monkeypatch.setattr(ext, "_run_git", flaky_run_git)

    caplog.set_level(logging.WARNING)
    result = _run_cmd(api, "branch", "", ctx)

    # /branch must refuse + surface the underlying failure, not preserve a
    # fake-empty branch.
    assert result is not None
    assert result.startswith("✗"), f"expected refusal, got: {result!r}"
    assert "working tree" in result.lower()
    # The `git add -A` failure itself must be logged so the user can act on it.
    assert any("git add -A failed" in rec.getMessage() for rec in caplog.records), (
        "expected explicit warning about git add -A failure"
    )
    # And no preserved branch must have been created with the wrong content.
    branches = set(_list_branches("shadow/session-*", cwd=repo))
    assert "shadow/session-s1-branch-1" not in branches, (
        "branch must not be preserved when pre-sync could not commit pending work"
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
# /branch
# ---------------------------------------------------------------------------


def test_branch_preserves_current_and_starts_fresh(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    result = _run_cmd(api, "branch", "", ctx)

    branches = set(_list_branches("shadow/session-*", cwd=repo))
    assert "shadow/session-s1" in branches  # new active
    assert "shadow/session-s1-branch-1" in branches  # preserved

    # Still on the shadow/session-s1 branch
    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "shadow/session-s1"

    # a.txt is still present (branched FROM HEAD keeps history identical)
    assert (repo / "a.txt").exists()

    st = _session_state(ctx)
    assert st["branch_counter"] == 1
    assert result is not None
    assert result.startswith("⑂")
    assert "branch-1" in result
    assert "shadow/session-s1" in result


def test_branch_back_n_rewinds(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")

    result = _run_cmd(api, "branch", "2", ctx)
    assert result is not None
    assert "rewound 2 checkpoint" in result

    # Only a.txt should remain on the active branch
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()

    # Preserved branch still has all three
    _git("checkout", "shadow/session-s1-branch-1", cwd=repo)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert (repo / "c.txt").exists()


def test_branch_back_n_skips_shadow_meta_commits(session_api) -> None:
    """/branch N counts logical checkpoints even when meta commits are interleaved."""
    api, ctx, repo = session_api
    for filename, marker in (("a.txt", "a"), ("b.txt", "b"), ("c.txt", "c")):
        _write_and_checkpoint(api, ctx, repo, filename, marker)
        (repo / "session.jsonl").write_text(f'{{"after": "{marker}"}}\n', encoding="utf-8")
        _fire_session_end(api, ctx)

    result = _run_cmd(api, "branch", "1", ctx)

    assert result is not None
    assert "rewound 1 checkpoint" in result
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    log = _git("log", "--oneline", "-1", cwd=repo).stdout
    assert "shadow-meta: session sync" in log


def test_branch_of_branch_of_branch_numbers_monotonically(session_api) -> None:
    """The central scenario: branch -> work -> branch -> work -> branch. Branch numbering
    is derived from the branches already on disk, so we must end with four
    distinct shadow/session-s1 branches (one active + three preserved)."""
    api, ctx, repo = session_api

    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")  # turn 1
    _run_cmd(api, "branch", "", ctx)  # preserve -> branch-1, active stays at turn 1

    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")  # turn 2 (on fresh branch)
    _run_cmd(api, "branch", "", ctx)  # preserve -> branch-2

    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")  # turn 3
    _run_cmd(api, "branch", "", ctx)  # preserve -> branch-3

    branches = set(_list_branches("shadow/session-*", cwd=repo))
    assert branches == {
        "shadow/session-s1",
        "shadow/session-s1-branch-1",
        "shadow/session-s1-branch-2",
        "shadow/session-s1-branch-3",
    }

    st = _session_state(ctx)
    assert st["branch_counter"] == 3

    # Each preserved branch should hold the cumulative files at the time it was
    # branched — branch-1 has only a.txt, branch-2 has a+b, branch-3 has a+b+c.
    _git("checkout", "shadow/session-s1-branch-1", cwd=repo)
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()

    _git("checkout", "shadow/session-s1-branch-2", cwd=repo)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()

    _git("checkout", "shadow/session-s1-branch-3", cwd=repo)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert (repo / "c.txt").exists()


def test_branch_beyond_checkpoints_is_refused(
    session_api, caplog: pytest.LogCaptureFixture
) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    caplog.set_level(logging.WARNING)
    result = _run_cmd(api, "branch", "9", ctx)

    assert any("cannot branch" in rec.getMessage() for rec in caplog.records)
    branches = _list_branches("shadow/session-s1*", cwd=repo)
    assert branches == ["shadow/session-s1"]  # unchanged
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
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    result = _run_cmd(api, "switch", "", ctx)

    assert result is not None
    assert "Current branch:" in result
    assert "shadow/session-s1" in result
    assert "shadow/session-s1-branch-1" in result
    assert "Usage:" in result
    # Must NOT have switched — still on the active shadow
    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-s1"


def test_switch_main_shorthand_returns_to_canonical_shadow(session_api) -> None:
    """'main', 'active', and 'shadow' keywords all switch back to the canonical
    shadow/session-<id> branch from a preserved branch copy."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    # Switch away to the preserved branch first
    _run_cmd(api, "switch", "branch-1", ctx)
    assert _session_state(ctx)["shadow_branch"] == "shadow/session-s1-branch-1"

    # Switch back using 'main'
    result = _run_cmd(api, "switch", "main", ctx)
    assert result is not None
    assert result.startswith("⇄")
    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-s1"
    assert (repo / "b.txt").exists()


def test_switch_active_shorthand(session_api) -> None:
    """'active' keyword resolves to the canonical shadow branch."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)

    _run_cmd(api, "switch", "branch-1", ctx)
    result = _run_cmd(api, "switch", "active", ctx)

    assert result is not None
    assert result.startswith("⇄")
    assert _session_state(ctx)["shadow_branch"] == "shadow/session-s1"


def test_switch_shadow_shorthand(session_api) -> None:
    """'shadow' keyword resolves to the canonical shadow branch."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)

    _run_cmd(api, "switch", "branch-1", ctx)
    result = _run_cmd(api, "switch", "shadow", ctx)

    assert result is not None
    assert result.startswith("⇄")
    assert _session_state(ctx)["shadow_branch"] == "shadow/session-s1"


def test_switch_between_forks(session_api) -> None:
    """After branch-of-branch-of-branch, /switch can hop to any preserved branch and
    the state (turn counter, checkpoints) is reconstructed from git history."""
    api, ctx, repo = session_api

    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")

    # Jump to the first fork using a bare numeric shorthand
    result = _run_cmd(api, "switch", "1", ctx)
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-s1-branch-1"
    assert st["turn_counter"] == 1
    assert len(st["checkpoints"]) == 1
    assert result is not None
    assert result.startswith("⇄")
    assert "branch-1" in result

    # Jump to branch-2 using the branch-K shorthand
    result = _run_cmd(api, "switch", "branch-2", ctx)
    assert (repo / "a.txt").exists()
    assert (repo / "b.txt").exists()
    assert not (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-s1-branch-2"
    assert st["turn_counter"] == 2
    assert result is not None
    assert result.startswith("⇄")
    assert "branch-2" in result

    # Jump back to the live branch using the full name
    result = _run_cmd(api, "switch", "shadow/session-s1", ctx)
    assert (repo / "c.txt").exists()
    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-s1"
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
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    # Simulate a session-store write that lands outside of tool_result
    (repo / "session.jsonl").write_text('{"event": "message"}\n')

    caplog.set_level(logging.DEBUG)
    _run_cmd(api, "switch", "branch-1", ctx)

    # Switch must have succeeded
    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-s1-branch-1"
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()

    # The pending file must have been committed on the shadow branch, not left dirty
    _git("checkout", "shadow/session-s1", cwd=repo)
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
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    (repo / "session.jsonl").write_text('{"event": "message"}\n')

    original_run_git = ext._run_git

    def _failing_commit(*args, **kwargs):
        if args and args[0] == "commit":
            return 1, "", "simulated commit failure"
        return original_run_git(*args, **kwargs)

    monkeypatch.setattr(ext, "_run_git", _failing_commit)

    caplog.set_level(logging.WARNING)
    _run_cmd(api, "switch", "branch-1", ctx)

    assert any("uncommitted changes" in rec.getMessage() for rec in caplog.records)
    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-s1"


def test_switch_unknown_branch(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, _repo = session_api
    caplog.set_level(logging.WARNING)
    result = _run_cmd(api, "switch", "nonexistent", ctx)
    assert any("does not exist" in rec.getMessage() for rec in caplog.records)
    assert result is not None
    assert result.startswith("✗")
    assert "does not exist" in result


# ---------------------------------------------------------------------------
# /branches listing
# ---------------------------------------------------------------------------


def test_branches_lists_all_branches_and_marks_active(
    session_api, caplog: pytest.LogCaptureFixture
) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _run_cmd(api, "branch", "", ctx)

    caplog.set_level(logging.INFO)
    result = _run_cmd(api, "branches", "", ctx)

    # The return value is the canonical output shown in the TUI/CLI
    assert result is not None
    assert "shadow/session-s1" in result
    assert "shadow/session-s1-branch-1" in result
    assert "shadow/session-s1-branch-2" in result
    # active marker present on exactly one line
    active_lines = [line for line in result.splitlines() if "◀ active" in line]
    assert len(active_lines) == 1
    # active is on the canonical root (no branch suffix) since we did two /branch calls
    # leaving the active shadow on shadow/session-s1
    assert "shadow/session-s1" in active_lines[0] and "branch-" not in active_lines[0]
    # log still emitted for the record
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("shadow/session-s1" in m for m in msgs)


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
    # No shadow-auto commits on main (they were squashed)
    assert "shadow-auto:" not in log
    assert result is not None
    assert result.startswith("✓")
    assert "main" in result


def test_done_refuses_with_remaining_forks(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    caplog.set_level(logging.INFO)
    result = _run_cmd(api, "done", "", ctx)
    # still on the shadow branch because /done aborted
    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "shadow/session-s1"
    assert result is not None
    assert result.startswith("⚠")
    assert "branch" in result  # branch names contain "branch" (shadow/session-s1-branch-1)
    assert "--yes" in result
    assert any("preserved branches still exist" in rec.getMessage() for rec in caplog.records)


def test_done_with_yes_proceeds_despite_forks(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    result = _run_cmd(api, "done", "--yes", ctx)

    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "main"
    assert "shadow/session-s1-branch-1" in _list_branches("shadow/session-s1*", cwd=repo)
    assert result is not None
    assert result.startswith("✓")
    assert "main" in result
    # b.txt from the active shadow is on main, but preserved branches still exist
    assert (repo / "b.txt").exists()
    branches = _list_branches("shadow/session-*", cwd=repo)
    assert "shadow/session-s1-branch-1" in branches


def test_done_aborts_on_merge_conflict(session_api, caplog: pytest.LogCaptureFixture) -> None:
    api, ctx, repo = session_api

    # Diverge: edit shared file on shadow
    _write_and_checkpoint(api, ctx, repo, "README.md", "shadow version\n")

    # Now edit the same file on main so merge will conflict
    _git("checkout", "main", cwd=repo)
    (repo / "README.md").write_text("main version\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo)
    _git("commit", "-m", "main edit", cwd=repo)
    _git("checkout", "shadow/session-s1", cwd=repo)

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
    assert "Aar session s1:" in log
    assert "checkpoint(s)" in log
    assert result is not None
    assert result.startswith("✓")


def test_done_disarms_hooks_on_base_branch(session_api) -> None:
    """After /done, HEAD is on the user's base branch — nothing may commit there.

    Regression: /done merged correctly but left ``state.enabled`` True, so the
    SessionStore.save hook, the before_turn / session_end sweeps and the next
    tool_result all kept firing while HEAD sat on ``main``, dropping
    ``shadow-meta:`` and ``shadow-auto:`` commits straight onto the user's
    branch — the exact thing the extension promises never to do.
    """
    from agent.core.session import Session
    from agent.core.state import AgentState
    from agent.memory.session_store import SessionStore

    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")

    result = _run_cmd(api, "done", "finishing up", ctx)
    assert result is not None and result.startswith("✓")

    # Now simulate everything that normally happens after the command returns:
    # the transport saving the JSONL, and the agent doing more work.
    store = SessionStore(base_dir=repo / ".agent" / "sessions")
    store.save(
        Session(
            session_id="s1",
            run_id="r1",
            trace_id="t1",
            state=AgentState.COMPLETED,
            step_count=1,
            metadata={},
            events=[],
        )
    )
    _fire_before_turn(api, ctx)
    (repo / "b.txt").write_text("two", encoding="utf-8")
    _fire_tool_result(api, ctx)
    _fire_session_end(api, ctx)

    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True
    ).stdout

    assert current == "main"
    assert "shadow-auto:" not in log
    assert "shadow-meta:" not in log
    assert len(log.strip().splitlines()) == 2, (
        f"only 'initial' + the squash commit expected on main, got:\n{log}"
    )

    # The post-/done work is still there in the tree, just uncommitted — the
    # extension stepped aside rather than swallowing it.
    status = _git("status", "--porcelain", cwd=repo).stdout
    assert "b.txt" in status

    # Subsequent commands explain the state instead of failing generically.
    assert "already merged" in (_run_cmd(api, "undo", "", ctx) or "")
    assert "already merged" in (_run_cmd(api, "branch", "", ctx) or "")
    assert "already merged" in (_run_cmd(api, "switch", "main", ctx) or "")
    assert "already merged" in (_run_cmd(api, "done", "", ctx) or "")

    assert _session_state(ctx)["mode"] == "done"
    assert _session_state(ctx)["enabled"] is False


def test_session_start_after_done_stays_disabled(repo: Path) -> None:
    """Resuming a session that was merged via /done must not re-arm it.

    /done deliberately keeps ``shadow/session-<id>`` around, so the resume path
    would otherwise check that branch back out and continue committing on top
    of work that is already merged.
    """
    api1 = FakeAPI()
    register(api1)
    ctx1 = FakeCtx(FakeSession("done1"))
    api1.handlers["session_start"][0](None, ctx1)

    (repo / "a.txt").write_text("one", encoding="utf-8")
    _fire_tool_result(api1, ctx1, tool_name="write_file")
    assert (_run_cmd(api1, "done", "merge it", ctx1) or "").startswith("✓")

    persisted = dict(_session_state(ctx1))
    assert persisted["mode"] == "done"
    # The shadow branch is intentionally left behind — that is the hazard.
    assert "shadow/session-done1" in _list_branches("shadow/session-done1*", cwd=repo)

    # Cold resume: fresh register(), same session id, metadata loaded from disk.
    api2 = FakeAPI()
    register(api2)
    session2 = FakeSession("done1")
    session2.metadata["shadow_branching"] = persisted
    ctx2 = FakeCtx(session2)
    api2.handlers["session_start"][0](None, ctx2)

    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert current == "main", "resume must not check the merged shadow branch back out"
    assert _session_state(ctx2)["enabled"] is False
    assert _session_state(ctx2)["mode"] == "done"

    # And the hooks stay quiet on the base branch.
    (repo / "b.txt").write_text("two", encoding="utf-8")
    _fire_tool_result(api2, ctx2)
    _fire_session_end(api2, ctx2)
    log = _git("log", "--oneline", cwd=repo).stdout
    assert "shadow-auto:" not in log
    assert "shadow-meta:" not in log


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
    assert st["shadow_branch"] == "shadow/session-resume1"
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
    assert "shadow-auto: write_file turn-3" in log


def test_session_start_leaves_other_sessions_untouched(repo: Path) -> None:
    # Simulate a stale shadow branch from a prior session
    _git("checkout", "-b", "shadow/session-zzzold", cwd=repo)
    _git("commit", "--allow-empty", "-m", "shadow-init: base=main", cwd=repo)
    _git("checkout", "main", cwd=repo)

    api = FakeAPI()
    register(api)
    ctx = FakeCtx(FakeSession("fresh1"))
    api.handlers["session_start"][0](None, ctx)

    branches = set(_list_branches("shadow/session-*", cwd=repo))
    assert "shadow/session-zzzold" in branches
    assert "shadow/session-fresh1" in branches


# ---------------------------------------------------------------------------
# Cross-session safety guards
# ---------------------------------------------------------------------------


def test_session_start_rebases_to_base_when_on_stale_shadow_branch(repo: Path) -> None:
    """Starting a fresh session while HEAD is on another session's shadow branch
    should auto-checkout the old branch's base (main) before creating the new
    shadow branch — preventing cross-session file/conversation inconsistency."""
    # Session A: create a shadow branch with some work
    api_a = FakeAPI()
    register(api_a)
    ctx_a = FakeCtx(FakeSession("old_session"))
    api_a.handlers["session_start"][0](None, ctx_a)

    (repo / "from_old.txt").write_text("old session work", encoding="utf-8")
    _fire_tool_result(api_a, ctx_a, tool_name="write_file")

    # Confirm HEAD is on the old shadow branch
    branch_out = _git("branch", "--show-current", cwd=repo).stdout.strip()
    assert branch_out == "shadow/session-old_session"

    # Session B: start a NEW session — repo is still on old_session's branch
    api_b = FakeAPI()
    register(api_b)
    ctx_b = FakeCtx(FakeSession("new_session"))
    api_b.handlers["session_start"][0](None, ctx_b)

    st = _session_state(ctx_b)
    assert st["enabled"] is True
    assert st["shadow_branch"] == "shadow/session-new_session"
    # The new shadow must be rooted on main, NOT on the old shadow branch
    assert st["original_branch"] == "main"

    # The shadow-init anchor must record base=main
    log = _git("log", "--oneline", "--grep=shadow-init:", "shadow/session-new_session", cwd=repo)
    assert "base=main" in log.stdout


def test_session_start_stale_shadow_warning_logged(
    repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When rebasing away from a stale shadow branch a warning must be logged."""
    # Create old session's shadow branch
    _git("checkout", "-b", "shadow/session-stale1", cwd=repo)
    _git("commit", "--allow-empty", "-m", "shadow-init: base=main", "--no-verify", cwd=repo)

    # Start new session while on stale branch
    api = FakeAPI()
    register(api)
    ctx = FakeCtx(FakeSession("fresh2"))
    with caplog.at_level(logging.WARNING):
        api.handlers["session_start"][0](None, ctx)

    assert any("shadow branch of another session" in r.message for r in caplog.records)
    assert any("--session" in r.message for r in caplog.records)


def test_session_start_uses_exact_session_branch_namespace(repo: Path) -> None:
    """Session IDs with shared prefixes (s1 vs s10) must not be confused."""
    api_old = FakeAPI()
    register(api_old)
    ctx_old = FakeCtx(FakeSession("s10"))
    api_old.handlers["session_start"][0](None, ctx_old)

    branch_out = _git("branch", "--show-current", cwd=repo).stdout.strip()
    assert branch_out == "shadow/session-s10"

    api_new = FakeAPI()
    register(api_new)
    ctx_new = FakeCtx(FakeSession("s1"))
    api_new.handlers["session_start"][0](None, ctx_new)

    st = _session_state(ctx_new)
    assert st["shadow_branch"] == "shadow/session-s1"
    assert st["original_branch"] == "main"
    log = _git("log", "--oneline", "--grep=shadow-init:", "shadow/session-s1", cwd=repo)
    assert "base=main" in log.stdout


def test_switch_rejects_cross_session_branch(repo: Path) -> None:
    """/switch must refuse to check out a branch belonging to a different session."""
    # Session A
    api_a = FakeAPI()
    register(api_a)
    ctx_a = FakeCtx(FakeSession("sess_a"))
    api_a.handlers["session_start"][0](None, ctx_a)

    (repo / "a.txt").write_text("a", encoding="utf-8")
    _fire_tool_result(api_a, ctx_a, tool_name="write_file")

    # Go back to main so session B doesn't trigger the stale-branch guard
    _git("checkout", "main", cwd=repo)

    # Session B
    api_b = FakeAPI()
    register(api_b)
    ctx_b = FakeCtx(FakeSession("sess_b"))
    api_b.handlers["session_start"][0](None, ctx_b)

    # Try to /switch to session A's branch — must be rejected
    result = _run_cmd(api_b, "switch", "shadow/session-sess_a", ctx_b)
    assert result is not None
    assert "different session" in result
    assert "✗" in result

    # Confirm we're still on session B's branch
    branch_out = _git("branch", "--show-current", cwd=repo).stdout.strip()
    assert branch_out == "shadow/session-sess_b"


def test_switch_uses_exact_session_branch_namespace(repo: Path) -> None:
    api = FakeAPI()
    register(api)
    ctx = FakeCtx(FakeSession("s1"))
    api.handlers["session_start"][0](None, ctx)

    _git("branch", "shadow/session-s10", cwd=repo)

    result = _run_cmd(api, "switch", "shadow/session-s10", ctx)

    assert result is not None
    assert "different session" in result
    assert _git("branch", "--show-current", cwd=repo).stdout.strip() == "shadow/session-s1"


def test_switch_allows_own_session_branches(repo: Path) -> None:
    """/switch must still allow switching between branches of the same session."""
    api, ctx, _ = FakeAPI(), None, repo

    api = FakeAPI()
    register(api)
    ctx = FakeCtx(FakeSession("mine"))
    api.handlers["session_start"][0](None, ctx)

    # Create some work and branch
    (repo / "x.txt").write_text("x", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")
    result = _run_cmd(api, "branch", "", ctx)
    assert result is not None and "branch-1" in result

    # /switch back to branch-1
    result = _run_cmd(api, "switch", "1", ctx)
    assert result is not None
    assert "switched to" in result.lower() or "⇄" in result

    # /switch back to main shadow
    result = _run_cmd(api, "switch", "main", ctx)
    assert result is not None
    assert "⇄" in result


def test_switch_survives_subsequent_session_start(repo: Path) -> None:
    """/switch must not be undone by the next run_loop's session_start.

    In chat mode, ``run_loop`` fires ``session_start`` on every user message.
    Before the fix, ``on_start`` would unconditionally ``git checkout`` the
    canonical shadow branch, silently reverting any ``/switch`` the user did
    between turns.  This test reproduces that exact scenario."""
    api = FakeAPI()
    register(api)
    session = FakeSession("surv")
    ctx = FakeCtx(session)
    api.handlers["session_start"][0](None, ctx)

    # Create work + branch
    (repo / "a.txt").write_text("a", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")
    result = _run_cmd(api, "branch", "", ctx)
    assert result is not None and "branch-1" in result

    # More work on fresh shadow
    (repo / "b.txt").write_text("b", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")

    # /switch to branch-1
    result = _run_cmd(api, "switch", "1", ctx)
    assert result is not None and "⇄" in result
    st = _session_state(ctx)
    assert st["shadow_branch"] == "shadow/session-surv-branch-1"

    # Confirm git HEAD is on branch-1
    branch_out = _git("branch", "--show-current", cwd=repo).stdout.strip()
    assert branch_out == "shadow/session-surv-branch-1"

    # Simulate the next run_loop call — fires session_start again
    api.handlers["session_start"][0](None, ctx)

    # State must still reflect branch-1, NOT the canonical shadow
    st2 = _session_state(ctx)
    assert st2["shadow_branch"] == "shadow/session-surv-branch-1", (
        f"session_start re-initialised and switched back to {st2['shadow_branch']}"
    )

    # git HEAD must still be on branch-1
    branch_out2 = _git("branch", "--show-current", cwd=repo).stdout.strip()
    assert branch_out2 == "shadow/session-surv-branch-1", (
        f"session_start checked out {branch_out2} instead of staying on branch-1"
    )

    # File state must still match branch-1 (a.txt yes, b.txt no)
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists()


def test_branch_reloads_session_events_to_fork_point(repo: Path) -> None:
    """/branch must reload session events from the fork point's JSONL so the
    in-memory session no longer contains conversation about work that now lives
    only on the preserved branch."""
    from agent.core.events import AssistantMessage, UserMessage
    from agent.core.session import Session as RealSession
    from agent.memory.session_store import SessionStore

    store = SessionStore(base_dir=repo / ".agent" / "sessions")

    api = FakeAPI()
    register(api)
    session = FakeSession("br_reload")
    ctx = FakeCtx(session)
    api.handlers["session_start"][0](None, ctx)

    # Turn 1: create a file + checkpoint
    (repo / "first.txt").write_text("one", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")

    # Simulate conversation events and save the session JSONL
    ctx.session.events = [
        UserMessage(content="create first.txt"),
        AssistantMessage(content="Done, created first.txt"),
    ]
    real_session = RealSession(
        session_id="br_reload",
        events=ctx.session.events,
        step_count=1,
        metadata=ctx.session.metadata,
    )
    store.save(real_session)
    _fire_before_turn(api, ctx)  # commit the JSONL

    # Turn 2: more work + more conversation
    (repo / "second.txt").write_text("two", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")

    ctx.session.events = [
        UserMessage(content="create first.txt"),
        AssistantMessage(content="Done, created first.txt"),
        UserMessage(content="now create second.txt"),
        AssistantMessage(content="Done, created second.txt"),
    ]
    real_session2 = RealSession(
        session_id="br_reload",
        events=ctx.session.events,
        step_count=2,
        metadata=ctx.session.metadata,
    )
    store.save(real_session2)
    _fire_before_turn(api, ctx)  # commit the JSONL

    # Sanity: 4 events in memory, both files on disk
    assert len(ctx.session.events) == 4
    assert (repo / "second.txt").exists()

    # /branch — preserves current state as branch-1, starts fresh from HEAD
    result = _run_cmd(api, "branch", "", ctx)
    assert result is not None and "branch-1" in result

    # After /branch, the session events must reflect the fork point's JSONL
    # (4 events — same as HEAD since we branched from HEAD with no rewind).
    # The key property: the events match what's on disk for the new branch.
    assert len(ctx.session.events) == 4

    # Now test /branch N (rewind): go back to turn-1
    # First do another piece of work so we have something to rewind
    (repo / "third.txt").write_text("three", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")
    ctx.session.events.append(UserMessage(content="create third.txt"))
    ctx.session.events.append(AssistantMessage(content="Done, created third.txt"))
    real_session3 = RealSession(
        session_id="br_reload",
        events=list(ctx.session.events),
        step_count=3,
        metadata=ctx.session.metadata,
    )
    store.save(real_session3)
    _fire_before_turn(api, ctx)

    assert len(ctx.session.events) == 6

    # /branch 1 — rewind 1 checkpoint, preserving current as branch-2
    result = _run_cmd(api, "branch", "1", ctx)
    assert result is not None and "branch-2" in result

    # After rewinding 1 checkpoint the JSONL on disk corresponds to the
    # commit before the third.txt checkpoint.  The session events must be
    # reloaded from that JSONL — so we should have fewer events than the 6
    # we had before branching.  The exact count depends on which commit the
    # branch-point JSONL was saved at; the important invariant is that the
    # events about "third.txt" are gone.
    assert len(ctx.session.events) < 6, (
        f"expected events to shrink after /branch 1, got {len(ctx.session.events)}"
    )
    contents = " ".join(getattr(e, "content", "") or "" for e in ctx.session.events)
    assert "third" not in contents.lower(), (
        "session events should not mention third.txt after rewinding past it"
    )


def test_switch_reloads_session_events_from_target_branch(repo: Path) -> None:
    """/switch must reload the session's conversation history from the target
    branch's JSONL so the LLM sees events matching the files on disk."""
    from agent.memory.session_store import SessionStore

    store = SessionStore(base_dir=repo / ".agent" / "sessions")

    api = FakeAPI()
    register(api)
    session = FakeSession("sw_reload")
    ctx = FakeCtx(session)
    api.handlers["session_start"][0](None, ctx)

    # Turn 1: write a file + simulate a session save (creates JSONL on branch)
    (repo / "first.txt").write_text("hello", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")

    # Fake some conversation events on the session object so the JSONL has content
    from agent.core.events import AssistantMessage, UserMessage

    ctx.session.events = [
        UserMessage(content="create first.txt"),
        AssistantMessage(content="Done, created first.txt"),
    ]
    # Manually save the session so the JSONL is written and committed
    from agent.core.session import Session as RealSession

    real_session = RealSession(
        session_id="sw_reload",
        events=ctx.session.events,
        step_count=1,
        metadata=ctx.session.metadata,
    )
    store.save(real_session)
    # Let the save hook commit it
    _fire_before_turn(api, ctx)

    # /branch to preserve this state
    result = _run_cmd(api, "branch", "", ctx)
    assert result is not None and "branch-1" in result

    # Turn 2 on the new branch: different work + different conversation
    (repo / "second.txt").write_text("world", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")

    ctx.session.events = [
        UserMessage(content="create first.txt"),
        AssistantMessage(content="Done, created first.txt"),
        UserMessage(content="now create second.txt"),
        AssistantMessage(content="Done, created second.txt"),
    ]
    real_session2 = RealSession(
        session_id="sw_reload",
        events=ctx.session.events,
        step_count=2,
        metadata=ctx.session.metadata,
    )
    store.save(real_session2)
    _fire_before_turn(api, ctx)

    # Sanity: we have 4 events and second.txt exists
    assert len(ctx.session.events) == 4
    assert (repo / "second.txt").exists()

    # /switch to branch-1 — should reload events from that branch's JSONL
    result = _run_cmd(api, "switch", "1", ctx)
    assert result is not None and "⇄" in result

    # Files must match branch-1 (first.txt yes, second.txt no)
    assert (repo / "first.txt").exists()
    assert not (repo / "second.txt").exists()

    # Session events must have been reloaded — should be the 2 events from branch-1
    assert len(ctx.session.events) == 2
    assert ctx.session.events[0].content == "create first.txt"
    assert ctx.session.step_count == 1


def test_switch_reload_missing_jsonl_clears_events(repo: Path) -> None:
    """/switch to a branch that has no JSONL should clear session events
    rather than leaving stale history from the previous branch."""
    api = FakeAPI()
    register(api)
    session = FakeSession("sw_nojsonl")
    ctx = FakeCtx(session)
    api.handlers["session_start"][0](None, ctx)

    # Create some work + checkpoint
    (repo / "a.txt").write_text("a", encoding="utf-8")
    _fire_tool_result(api, ctx, tool_name="write_file")

    # Put some events on the live session
    from agent.core.events import UserMessage

    ctx.session.events = [UserMessage(content="hello")]

    # /branch preserves current state
    result = _run_cmd(api, "branch", "", ctx)
    assert result is not None and "branch-1" in result

    # On the new (fresh) branch there's no JSONL yet
    # /switch back to branch-1 which also has no JSONL saved by SessionStore
    result = _run_cmd(api, "switch", "1", ctx)
    assert result is not None and "⇄" in result

    # Events should be cleared since there's no JSONL to load
    assert len(ctx.session.events) == 0


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
