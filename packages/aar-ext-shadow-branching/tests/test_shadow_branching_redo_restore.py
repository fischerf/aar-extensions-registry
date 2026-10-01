"""/redo, /restore, --dry-run previews and panel action previews."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from test_shadow_branching import FakeAPI, _fire_tool_result, _run_cmd, _session_state

# ``repo`` / ``session_api`` fixtures come from conftest.py.

_ids = iter(range(10_000))


def _prompt(ctx: Any, text: str) -> None:
    ctx.session.events.append(
        SimpleNamespace(type="user_message", id=f"r{next(_ids)}", content=text, data={})
    )


def _edit(api: FakeAPI, ctx: Any, repo, name: str, body: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    _fire_tool_result(api, ctx, "write_file")


def _turns(ctx: Any) -> list[int]:
    return [c["turn"] for c in _session_state(ctx)["checkpoints"]]


def _setup(api: FakeAPI, ctx: Any, repo) -> None:
    """p1: t1 (a.txt) t2 (b.txt) · p2: t3 (a.txt again)."""
    _prompt(ctx, "first")
    _edit(api, ctx, repo, "a.txt", "one\n")
    _edit(api, ctx, repo, "b.txt", "two\n")
    _prompt(ctx, "second")
    _edit(api, ctx, repo, "a.txt", "one\nmore\n")


# ---------------------------------------------------------------------------
# /undo previews
# ---------------------------------------------------------------------------


def test_undo_reports_what_was_dropped(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    out = _run_cmd(api, "undo", "p2", ctx)
    assert out.startswith("↩ reverted 1 checkpoint(s) (p2 and later) → ")
    assert out.endswith("· dropped 1 file +1 −0 · /redo restores")


def test_undo_dry_run_changes_nothing(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    out = _run_cmd(api, "undo", "p1 --dry-run", ctx)
    assert (
        out == "• /undo would drop 3 checkpoint(s) (p1 and later) · 2 files +3 −0 — nothing changed"
    )
    assert _turns(ctx) == [1, 2, 3]
    assert (repo / "a.txt").read_text() == "one\nmore\n"
    assert _session_state(ctx)["redo_stack"] == []


# ---------------------------------------------------------------------------
# /redo
# ---------------------------------------------------------------------------


def test_redo_reapplies_the_last_undo(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    head_before = _session_state(ctx)["checkpoints"][-1]["hash"]
    _run_cmd(api, "undo", "2", ctx)
    assert _turns(ctx) == [1] and not (repo / "b.txt").exists()

    out = _run_cmd(api, "redo", "", ctx)
    assert out.startswith("↪ restored 2 checkpoint(s) → ")
    assert _turns(ctx) == [1, 2, 3]
    assert _session_state(ctx)["checkpoints"][-1]["hash"] == head_before
    assert (repo / "b.txt").read_text() == "two\n"
    assert (repo / "a.txt").read_text() == "one\nmore\n"
    assert _run_cmd(api, "redo", "", ctx) == "• nothing to redo"


def test_redo_stacks_multiple_undos(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    _run_cmd(api, "undo", "", ctx)
    _run_cmd(api, "undo", "", ctx)
    assert _turns(ctx) == [1]
    assert _run_cmd(api, "redo", "", ctx).endswith("· 1 more /redo")
    assert _turns(ctx) == [1, 2]
    _run_cmd(api, "redo", "", ctx)
    assert _turns(ctx) == [1, 2, 3]


def test_new_checkpoint_clears_redo(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    _run_cmd(api, "undo", "", ctx)
    _edit(api, ctx, repo, "c.txt", "new\n")
    assert _run_cmd(api, "redo", "", ctx) == "• nothing to redo"


def test_new_prompt_blocks_redo(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    _run_cmd(api, "undo", "", ctx)
    _prompt(ctx, "just a question, no edits")
    assert (
        _run_cmd(api, "redo", "", ctx)
        == "✗ cannot redo — the conversation moved on since the /undo"
    )
    assert _run_cmd(api, "redo", "", ctx) == "• nothing to redo"


def test_branch_clears_redo(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    _run_cmd(api, "undo", "", ctx)
    _run_cmd(api, "branch", "", ctx)
    assert _run_cmd(api, "redo", "", ctx) == "• nothing to redo"


# ---------------------------------------------------------------------------
# /restore
# ---------------------------------------------------------------------------


def test_restore_defaults_to_before_last_change_of_the_file(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    out = _run_cmd(api, "restore", "a.txt", ctx)
    assert out.startswith("⟲ restored a.txt to its state before t3 · checkpoint t4 ")
    assert out.endswith("· 1 file +0 −1 · /undo t4 takes it back")
    assert (repo / "a.txt").read_text() == "one\n"
    assert (repo / "b.txt").read_text() == "two\n", "other files untouched"
    cp = _session_state(ctx)["checkpoints"][-1]
    assert cp["tool"] == "restore" and cp["prompt_text"] == "/restore a.txt"
    assert "prompt" not in cp

    # It is an ordinary checkpoint: /undo takes it back.
    _run_cmd(api, "undo", "t4", ctx)
    assert (repo / "a.txt").read_text() == "one\nmore\n"


def test_restore_before_a_prompt_removes_a_created_file(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    out = _run_cmd(api, "restore", "b.txt p1", ctx)
    assert "restored b.txt (removed — it did not exist) before p1" in out
    assert not (repo / "b.txt").exists()


def test_restore_refusals(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    assert _run_cmd(api, "restore", "", ctx).startswith("✗ usage: /restore")
    assert (
        _run_cmd(api, "restore", "../outside.txt", ctx)
        == "✗ ../outside.txt is outside the repository"
    )
    assert _run_cmd(api, "restore", "nope.txt", ctx).startswith("✗ no checkpoint changed nope.txt")
    assert _run_cmd(api, "restore", "a.txt t9", ctx) == "✗ no checkpoint t9 on shadow"
    (repo / "a.txt").write_text("dirty\n", encoding="utf-8")
    assert _run_cmd(api, "restore", "a.txt", ctx) == (
        "✗ a.txt has uncommitted changes — commit them or use --force"
    )
    assert _run_cmd(api, "restore", "a.txt --force", ctx).startswith("⟲ restored a.txt")
    assert (repo / "a.txt").read_text() == "one\n"
    assert _run_cmd(api, "restore", "a.txt t1", ctx).startswith("⟲ restored a.txt")
    assert not (repo / "a.txt").exists()


def test_restore_groups_under_its_own_label(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    _run_cmd(api, "restore", "a.txt", ctx)
    lines = _run_cmd(api, "branches", "", ctx).splitlines()
    assert any(ln.lstrip("│ ├└─").startswith("/restore a.txt  1 cp") for ln in lines)


# ---------------------------------------------------------------------------
# /done --dry-run and panel previews
# ---------------------------------------------------------------------------


def test_done_dry_run_has_no_side_effects(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    (repo / "pending.txt").write_text("x", encoding="utf-8")
    out = _run_cmd(api, "done", "--dry-run", ctx)
    assert out == (
        "• /done would squash 3 checkpoint(s) from shadow/session-s1 into main"
        " · 2 files +3 −0 — nothing changed"
    )
    assert (repo / "pending.txt").exists()
    assert _session_state(ctx)["enabled"] is True
    assert len(_session_state(ctx)["checkpoints"]) == 3


def test_panel_actions_preview(session_api) -> None:
    api, ctx, repo = session_api
    _setup(api, ctx, repo)
    _run_cmd(api, "branch", "1", ctx)  # preserve, fresh line without p2
    panel = api.panels[0]
    if not hasattr(panel.actions[0], "preview"):  # pragma: no cover — older core
        return
    root = panel.snapshot(ctx)

    def walk(node):
        yield node
        for c in node.children:
            yield from walk(c)

    nodes = list(walk(root))
    p1 = next(n for n in nodes if n.kind == "prompt" and n.data["active_branch"])
    t1 = next(
        n
        for n in nodes
        if n.kind == "checkpoint" and n.data["active_branch"] and n.data["turn"] == 1
    )
    sibling = next(n for n in nodes if n.kind == "branch" and not n.data["active"])

    assert panel.action("undo_prompt").preview(p1, ctx) == (
        "drops 2 checkpoint(s) · 2 files +2 −0 · /redo restores"
    )
    assert panel.action("undo").preview(t1, ctx) == (
        "drops 1 checkpoint(s) · 1 file +1 −0 · /redo restores"
    )
    assert panel.action("delete").preview(sibling, ctx) == "3 checkpoint(s) · 2 files +3 −0 vs main"
    assert (
        panel.action("done").preview(root, ctx).startswith("• /done would squash 2 checkpoint(s)")
    )
    assert panel.action("undo").to_dict()["preview"] is True
