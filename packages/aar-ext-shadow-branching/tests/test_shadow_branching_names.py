"""Branch names (/branch … <name>, /label), /compare, /done --cleanup, and the
patch in a checkpoint's panel description."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from test_shadow_branching import (
    FakeAPI,
    _fire_tool_result,
    _list_branches,
    _run_cmd,
    _session_state,
)

# ``repo`` / ``session_api`` fixtures come from conftest.py.

_ids = iter(range(10_000))


def _prompt(ctx: Any, text: str) -> None:
    ctx.session.events.append(
        SimpleNamespace(type="user_message", id=f"n{next(_ids)}", content=text, data={})
    )


def _edit(api: FakeAPI, ctx: Any, repo, name: str, body: str) -> None:
    (repo / name).write_text(body, encoding="utf-8")
    _fire_tool_result(api, ctx, "write_file")


def _two_attempts(api: FakeAPI, ctx: Any, repo) -> None:
    """p1 shared; p2 tried with sqlite (preserved as "sqlite-attempt"), then
    retried as p3 with json on the active line."""
    _prompt(ctx, "set up the app")
    _edit(api, ctx, repo, "app.py", "app\n")
    _prompt(ctx, "add storage with sqlite")
    _edit(api, ctx, repo, "store.py", "sqlite\n")
    _edit(api, ctx, repo, "schema.sql", "create table t;\n")
    assert _run_cmd(api, "branch", "p2 sqlite-attempt", ctx).startswith(
        '⑂ branch-1 "sqlite-attempt" preserved as shadow/session-s1-branch-1'
    )
    _prompt(ctx, "add storage with json files")
    _edit(api, ctx, repo, "store.py", "json\n")


def test_named_preserved_branch_everywhere(session_api) -> None:
    api, ctx, repo = session_api
    _two_attempts(api, ctx, repo)

    tree = _run_cmd(api, "branches", "", ctx)
    assert '○ shadow/session-s1-branch-1 "sqlite-attempt"  (3 cp)' in tree

    panel = api.panels[0]
    root = panel.snapshot(ctx)
    sibling = next(n for n in root.children if n.kind == "branch" and not n.data["active"])
    assert sibling.label == "sqlite-attempt (3 cp)"

    out = _run_cmd(api, "switch", "sqlite-attempt", ctx)
    assert out.startswith("⇄ switched to shadow/session-s1-branch-1")
    assert (repo / "store.py").read_text() == "sqlite\n"


def test_label_names_the_active_line(session_api) -> None:
    api, ctx, repo = session_api
    _two_attempts(api, ctx, repo)
    assert _run_cmd(api, "label", '"json attempt"', ctx) == (
        '✎ shadow/session-s1 is now "json attempt"'
    )
    assert api.panels[0].status_text(ctx) == "⎇ json attempt · 2 cp"
    assert '● shadow/session-s1 "json attempt"' in _run_cmd(api, "branches", "", ctx)

    # The name travels with the work when that line is preserved by /branch.
    _run_cmd(api, "branch", "", ctx)
    assert '"json attempt"' in _run_cmd(api, "branches", "", ctx)
    assert api.panels[0].status_text(ctx) == "⎇ shadow · 2 cp"

    assert _run_cmd(api, "label", "", ctx) == "✎ cleared the name of shadow/session-s1"


def test_compare_two_attempts(session_api) -> None:
    api, ctx, repo = session_api
    _two_attempts(api, ctx, repo)
    out = _run_cmd(api, "compare", "sqlite-attempt", ctx)
    lines = out.splitlines()
    assert lines[0] == "⇄ shadow (2 cp, active) vs sqlite-attempt (3 cp)"
    assert lines[1].startswith("  forked at ")
    mine = lines[lines.index("only on shadow:") + 1]
    theirs_at = lines.index("only on sqlite-attempt:")
    assert mine.startswith('  p3 "add storage with json files"  1 cp')
    assert lines[theirs_at + 1].startswith('  p2 "add storage with sqlite"  2 cp')
    assert "shadow → sqlite-attempt:" in lines
    assert "schema.sql" in out and "store.py" in out
    assert lines[-1].startswith("keep sqlite-attempt: /switch sqlite-attempt")

    patch = _run_cmd(api, "compare", "1 --patch", ctx)
    assert "-json" in patch and "+sqlite" in patch

    assert _run_cmd(api, "compare", "", ctx).startswith("✗ usage: /compare")
    assert _run_cmd(api, "compare", "nope", ctx) == (
        "✗ no branch 'nope' in this session — see /branches"
    )
    assert _run_cmd(api, "compare", "main", ctx) == "• that is the active line — nothing to compare"


def test_done_cleanup_deletes_shadow_branches(session_api) -> None:
    api, ctx, repo = session_api
    _two_attempts(api, ctx, repo)
    out = _run_cmd(api, "done", "--yes --cleanup ship json storage", ctx)
    assert out.startswith("✓ squashed shadow/session-s1 → main as ")
    assert out.endswith("· deleted 2 shadow branch(es)")
    assert _list_branches("shadow/session-s1*") == []
    assert _session_state(ctx)["mode"] == "done"


def test_done_without_cleanup_keeps_branches(session_api) -> None:
    api, ctx, repo = session_api
    _two_attempts(api, ctx, repo)
    out = _run_cmd(api, "done", "--yes", ctx)
    assert "deleted" not in out
    assert len(_list_branches("shadow/session-s1*")) == 2


def test_checkpoint_description_includes_the_patch(session_api) -> None:
    api, ctx, repo = session_api
    _two_attempts(api, ctx, repo)
    panel = api.panels[0]
    root = panel.snapshot(ctx)
    active = next(n for n in root.children if n.kind == "branch" and n.data["active"])
    tip = active.children[0].children[0]
    text = panel.describe(tip, ctx)
    assert "tip of the active shadow branch" in text
    assert "diff --git a/store.py b/store.py" in text
    assert "-sqlite" not in text and "+json" in text
