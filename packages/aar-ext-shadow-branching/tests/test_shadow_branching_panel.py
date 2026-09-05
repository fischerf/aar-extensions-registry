"""UI-panel tests for the shadow-branching extension.

Drives the ``UIPanel`` the extension registers (snapshot + actions) against a
real temp git repo, exactly like the slash-command tests — no TUI involved.
The panel is a thin layer over the commands, so these tests focus on the
mapping: node ``data`` → command arguments, refusals, and change signalling.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

import pytest
from agent.extensions.api import UIInvocation, UINode, UIPanel, run_ui_action, run_ui_snapshot
from test_shadow_branching import (
    FakeAPI,
    _fire_tool_result,
    _git,
    _run_cmd,
    _session_state,
    _write_and_checkpoint,
)

# ``repo`` / ``session_api`` fixtures come from conftest.py.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panel(api: FakeAPI) -> UIPanel:
    assert len(api.panels) == 1, "extension should register exactly one panel"
    return api.panels[0]


def _snap(api: FakeAPI, ctx: Any) -> UINode:
    root = _panel(api).snapshot(ctx)
    assert isinstance(root, UINode)
    return root


def _act(api: FakeAPI, ctx: Any, action_id: str, node: UINode, **args: Any) -> str | None:
    action = _panel(api).action(action_id)
    assert action is not None, f"no action {action_id!r}"
    assert action.applies_to(node), f"{action_id} does not apply to kind {node.kind!r}"
    return action.handler(UIInvocation(node=node, ctx=ctx, args=args))


def _first(root: UINode, kind: str) -> UINode:
    hit = next((n for n in _walk(root) if n.kind == kind), None)
    assert hit is not None, f"no node of kind {kind!r}"
    return hit


def _walk(node: UINode):
    yield node
    for child in node.children:
        yield from _walk(child)


def _active_branch(root: UINode) -> UINode:
    return next(n for n in _walk(root) if n.kind == "branch" and n.data.get("active"))


def _current_branch(repo) -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Registration + snapshot shape
# ---------------------------------------------------------------------------


def test_panel_registered_with_expected_actions(session_api) -> None:
    api, _ctx, _repo = session_api
    panel = _panel(api)
    assert panel.name == "shadow_branching"
    assert panel.title.endswith("Shadow")
    assert {a.id for a in panel.actions} == {
        "undo",
        "branch",
        "switch",
        "diff",
        "delete",
        "done",
        "refresh",
    }
    undo = panel.action("undo")
    assert undo is not None and undo.destructive and "force" in undo.inputs
    done = panel.action("done")
    assert done is not None and done.destructive and "message" in done.inputs
    assert panel.action("diff").mutates is False  # type: ignore[union-attr]
    assert panel.action("refresh").mutates is False  # type: ignore[union-attr]


def test_snapshot_shape_newest_checkpoint_first(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    root = _snap(api, ctx)
    assert root.kind == "root"
    assert root.label == "session s1"
    assert [c.kind for c in root.children] == ["base", "branch", "info"]

    base = root.children[0]
    assert base.label.startswith("main @ ")
    assert base.data["name"] == "main"

    active = root.children[1]
    assert "● active" in active.label
    assert active.data == {"name": "shadow/session-s1", "active": True, "checkpoints": 2}
    turns = [c.data["turn"] for c in active.children]
    assert turns == [2, 1], "newest checkpoint must be on top"

    tip, older = active.children
    assert tip.kind == "checkpoint"
    assert tip.data["n_back"] == 0 and tip.style == "active" and tip.label.endswith("●")
    assert older.data["n_back"] == 1 and older.style == ""
    assert tip.data["files"] == 1 and tip.data["flagged"] is False
    assert "write_file" in tip.label

    pending = root.children[2]
    assert pending.style == "dim" and "clean" in pending.label


def test_snapshot_flags_sensitive_checkpoint(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, ".env", "SECRET=1")

    tip = _active_branch(_snap(api, ctx)).children[0]
    assert tip.data["flagged"] is True
    assert "⚠" in tip.label
    assert tip.style == "warn"


def test_snapshot_reports_pending_changes(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    (repo / "new.txt").write_text("new", encoding="utf-8")

    pending = _first(_snap(api, ctx), "info")
    assert pending.style == "warn"
    assert "1 modified" in pending.label and "1 untracked" in pending.label


def test_snapshot_backfills_details_for_reconstructed_checkpoints(repo) -> None:
    """Checkpoints rebuilt from ``git log`` on resume carry no files/flagged —
    the snapshot must compute them from the commit instead of crashing."""
    from test_shadow_branching import FakeCtx, FakeSession, register

    api1 = FakeAPI()
    register(api1)
    ctx1 = FakeCtx(FakeSession("re1"))
    api1.handlers["session_start"][0](None, ctx1)
    _write_and_checkpoint(api1, ctx1, repo, "credentials.json", "{}")

    api2 = FakeAPI()
    register(api2)
    ctx2 = FakeCtx(FakeSession("re1"))
    api2.handlers["session_start"][0](None, ctx2)
    assert "files" not in _session_state(ctx2)["checkpoints"][0]

    tip = _active_branch(_snap(api2, ctx2)).children[0]
    assert tip.data["files"] == 1
    assert tip.data["flagged"] is True


def test_status_chip(session_api) -> None:
    api, ctx, repo = session_api
    assert _panel(api).status_text(ctx) == "⎇ session-s1 · 0 cp"
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    assert _panel(api).status_text(ctx) == "⎇ session-s1 · 1 cp"


def test_changed_is_set_by_checkpoint_and_cleared_by_transport(session_api) -> None:
    api, ctx, repo = session_api
    panel = _panel(api)
    panel.changed.clear()
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    assert panel.changed.is_set()
    panel.changed.clear()
    _fire_tool_result(api, ctx, tool_name="read_file")  # nothing to commit
    assert not panel.changed.is_set()


def test_run_ui_snapshot_runs_sync_snapshot_off_loop(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    root = asyncio.run(run_ui_snapshot(_panel(api), ctx))
    assert root.kind == "root"
    assert root.to_dict()["children"][1]["data"]["checkpoints"] == 1


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def test_undo_to_here_drops_later_checkpoints_only(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    _write_and_checkpoint(api, ctx, repo, "c.txt", "three")

    active = _active_branch(_snap(api, ctx))
    turn1 = next(n for n in active.children if n.data["turn"] == 1)
    assert turn1.data["n_back"] == 2

    result = _act(api, ctx, "undo", turn1)
    assert result is not None and result.startswith("↩")
    assert len(_session_state(ctx)["checkpoints"]) == 1
    assert (repo / "a.txt").exists()
    assert not (repo / "b.txt").exists() and not (repo / "c.txt").exists()

    # The tree now shows turn 1 as the tip.
    tip = _active_branch(_snap(api, ctx)).children[0]
    assert tip.data["turn"] == 1 and tip.data["n_back"] == 0


def test_undo_on_tip_is_refused(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    tip = _active_branch(_snap(api, ctx)).children[0]
    before = _git("rev-parse", "HEAD", cwd=repo).stdout

    result = _act(api, ctx, "undo", tip)
    assert result is not None and "nothing to undo" in result
    assert _git("rev-parse", "HEAD", cwd=repo).stdout == before
    assert len(_session_state(ctx)["checkpoints"]) == 1


def test_undo_force_discards_dirty_tree(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")
    (repo / "a.txt").write_text("dirty", encoding="utf-8")
    turn1 = next(n for n in _active_branch(_snap(api, ctx)).children if n.data["turn"] == 1)

    refused = _act(api, ctx, "undo", turn1)
    assert refused is not None and refused.startswith("✗")

    forced = _act(api, ctx, "undo", turn1, force=True)
    assert forced is not None and forced.startswith("↩")
    assert (repo / "a.txt").read_text(encoding="utf-8") == "one"


def test_branch_from_checkpoint_and_from_active_branch(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    turn1 = next(n for n in _active_branch(_snap(api, ctx)).children if n.data["turn"] == 1)
    result = _act(api, ctx, "branch", turn1)
    assert result is not None and result.startswith("⑂")

    root = _snap(api, ctx)
    branches = [n for n in _walk(root) if n.kind == "branch"]
    assert [b.data["active"] for b in branches] == [True, False]
    assert branches[0].data["checkpoints"] == 1  # fresh shadow from turn 1
    sibling = branches[1]
    assert sibling.data["name"] == "shadow/session-s1-branch-1"
    assert sibling.expanded is False
    assert "(2 cp)" in sibling.label
    assert [c.data["turn"] for c in sibling.children] == [2, 1]
    assert all(c.data["active_branch"] is False for c in sibling.children)

    # Fork from the active branch node == /branch with no N.
    result2 = _act(api, ctx, "branch", branches[0])
    assert result2 is not None and result2.startswith("⑂")
    assert "shadow/session-s1-branch-2" in {
        n.data["name"] for n in _walk(_snap(api, ctx)) if n.kind == "branch"
    }


def test_branch_and_undo_refused_on_sibling_checkpoints(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)

    sibling = next(n for n in _walk(_snap(api, ctx)) if n.kind == "branch" and not n.data["active"])
    cp = sibling.children[0]
    assert (_act(api, ctx, "undo", cp) or "").startswith("✗")
    assert (_act(api, ctx, "branch", cp) or "").startswith("✗")
    assert (_act(api, ctx, "branch", sibling) or "").startswith("✗")


def test_switch_then_delete_rules(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)

    root = _snap(api, ctx)
    active = _active_branch(root)
    sibling = next(n for n in _walk(root) if n.kind == "branch" and not n.data["active"])
    base = _first(root, "base")

    assert (_act(api, ctx, "switch", active) or "").startswith("•")
    assert (_act(api, ctx, "delete", active) or "").startswith("✗ refusing")
    # ``delete`` is not offered on the base node at all; the handler refuses
    # anyway in case a client (ACP) sends it.
    delete = _panel(api).action("delete")
    assert delete is not None and not delete.applies_to(base)
    refused = delete.handler(UIInvocation(node=base, ctx=ctx, args={}))
    assert (refused or "").startswith("✗ refusing")

    switched = _act(api, ctx, "switch", sibling)
    assert switched is not None and switched.startswith("⇄")
    assert _current_branch(repo) == "shadow/session-s1-branch-1"
    assert _active_branch(_snap(api, ctx)).data["name"] == "shadow/session-s1-branch-1"

    # The canonical branch is now the non-active sibling — deletable.
    canonical = next(
        n for n in _walk(_snap(api, ctx)) if n.kind == "branch" and not n.data["active"]
    )
    assert canonical.data["name"] == "shadow/session-s1"
    result = _act(api, ctx, "delete", canonical)
    assert result == "✓ deleted shadow/session-s1"
    remaining = {
        ln.strip().lstrip("* ").strip()
        for ln in _git("branch", "--list", cwd=repo).stdout.splitlines()
    }
    assert "shadow/session-s1" not in remaining
    assert [n.data["name"] for n in _walk(_snap(api, ctx)) if n.kind == "branch"] == [
        "shadow/session-s1-branch-1"
    ]


def test_diff_shows_checkpoint_stat(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    tip = _active_branch(_snap(api, ctx)).children[0]
    out = _act(api, ctx, "diff", tip)
    assert out is not None
    assert "a.txt" in out and "shadow-auto: write_file turn-1" in out


def test_done_via_panel_uses_message_and_implies_yes(session_api) -> None:
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    _run_cmd(api, "branch", "", ctx)  # preserved branch → /done would need --yes
    _write_and_checkpoint(api, ctx, repo, "b.txt", "two")

    root = _snap(api, ctx)
    result = _act(api, ctx, "done", root, message="merged from panel")
    assert result is not None and result.startswith("✓")
    assert _current_branch(repo) == "main"
    assert "merged from panel" in _git("log", "--oneline", cwd=repo).stdout

    after = _snap(api, ctx)
    assert after.data["mode"] == "done"
    labels = [n.label for n in _walk(after)]
    assert any("inactive" in lbl for lbl in labels)
    assert any("merged via /done" in lbl for lbl in labels)
    assert _panel(api).status_text(ctx) == ""
    # No actions apply to the inactive info nodes except refresh.
    info = _first(after, "info")
    assert [a.id for a in _panel(api).actions_for(info)] == ["refresh"]


def test_refresh_action_is_noop_message(session_api) -> None:
    api, ctx, _repo = session_api
    assert _act(api, ctx, "refresh", _snap(api, ctx)) is None


@pytest.mark.parametrize("action_id", ["undo", "branch", "switch", "diff", "delete", "done"])
def test_actions_run_through_core_helper(session_api, action_id: str) -> None:
    """``run_ui_action`` must accept the plugin's sync handlers (threaded)."""
    api, ctx, repo = session_api
    _write_and_checkpoint(api, ctx, repo, "a.txt", "one")
    root = _snap(api, ctx)
    action = _panel(api).action(action_id)
    assert action is not None
    node = next(n for n in _walk(root) if action.applies_to(n))
    if action_id == "done":
        node = root
    result = asyncio.run(run_ui_action(action, UIInvocation(node=node, ctx=ctx, args={})))
    assert result is None or isinstance(result, str)
