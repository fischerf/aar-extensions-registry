"""Prompt grouping and checkpoint / prompt references (tN, pN, <sha>).

Prompts come from the session's real (non-internal) user messages; each
checkpoint records its prompt number and an excerpt in the session state and
as ``Shadow-Prompt`` / ``Shadow-Prompt-Text`` trailers on the commit, so
reconstructed branches keep the grouping.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

from test_shadow_branching import (
    FakeAPI,
    _fire_tool_result,
    _run_cmd,
    _session_state,
)

# ``repo`` / ``session_api`` fixtures come from conftest.py.


_ids = iter(range(10_000))


def _prompt(ctx: Any, text: str, *, internal: bool = False) -> None:
    """Append a user message the way Aar's session does (duck-typed)."""
    ctx.session.events.append(
        SimpleNamespace(
            type="user_message",
            id=f"u{next(_ids)}",
            content=text,
            data={"internal": True} if internal else {},
        )
    )


def _edit(api: FakeAPI, ctx: Any, repo, name: str, body: str, tool: str = "write_file") -> None:
    (repo / name).write_text(body, encoding="utf-8")
    _fire_tool_result(api, ctx, tool)


def _three_prompts(api: FakeAPI, ctx: Any, repo) -> None:
    """p1: t1 t2 · p2: t3 · p3: t4 (+ an internal nudge inside p3)."""
    _prompt(ctx, "add a login form\nwith two fields")
    _edit(api, ctx, repo, "login.py", "a\n")
    _edit(api, ctx, repo, "app.py", "b\n", "edit_file")
    _prompt(ctx, "write tests")
    _edit(api, ctx, repo, "test_login.py", "c\nd\n")
    _prompt(ctx, "rename things")
    _prompt(ctx, "loop nudge", internal=True)
    _edit(api, ctx, repo, "login.py", "a2\n", "edit_file")


def _ckpts(ctx: Any) -> list[dict[str, Any]]:
    return _session_state(ctx)["checkpoints"]


def _panel(api: FakeAPI):
    return api.panels[0]


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_checkpoints_record_prompt_number_and_excerpt(session_api) -> None:
    api, ctx, repo = session_api
    _three_prompts(api, ctx, repo)
    cps = _ckpts(ctx)
    assert [c["prompt"] for c in cps] == [1, 1, 2, 3]  # the internal nudge is not a prompt
    assert cps[0]["prompt_text"] == "add a login form"  # first line only
    assert _session_state(ctx)["prompt_counter"] == 3

    body = subprocess.run(
        ["git", "log", "-1", "--format=%s%n%b"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert body.startswith("shadow-auto: edit_file turn-4")  # subject unchanged
    assert "Shadow-Prompt: 3" in body and "Shadow-Prompt-Text: rename things" in body


def test_without_prompts_layout_stays_flat(session_api) -> None:
    api, ctx, repo = session_api
    _edit(api, ctx, repo, "a.txt", "one")
    assert "prompt" not in _ckpts(ctx)[0]
    root = _panel(api).snapshot(ctx)
    active = next(n for n in _walk(root) if n.kind == "branch")
    assert [c.kind for c in active.children] == ["checkpoint"]


def test_prompt_numbers_are_never_reused(session_api) -> None:
    api, ctx, repo = session_api
    _three_prompts(api, ctx, repo)
    assert _run_cmd(api, "undo", "p3", ctx).startswith("↩ reverted 1 checkpoint(s) (p3 and later)")
    _prompt(ctx, "try again")
    _edit(api, ctx, repo, "other.py", "x\n")
    assert _ckpts(ctx)[-1]["prompt"] == 4


# ---------------------------------------------------------------------------
# References in commands
# ---------------------------------------------------------------------------


def test_undo_by_prompt_checkpoint_and_sha(session_api) -> None:
    api, ctx, repo = session_api
    _three_prompts(api, ctx, repo)

    out = _run_cmd(api, "undo", "p2", ctx)
    assert out.startswith("↩ reverted 2 checkpoint(s) (p2 and later)")
    assert [c["turn"] for c in _ckpts(ctx)] == [1, 2]

    sha = _ckpts(ctx)[1]["hash"]
    out = _run_cmd(api, "undo", sha, ctx)
    assert out.startswith(f"↩ reverted 1 checkpoint(s) ({sha} and later)")
    assert [c["turn"] for c in _ckpts(ctx)] == [1]
    assert not (repo / "app.py").exists()

    assert _run_cmd(api, "undo", "t9", ctx) == "✗ no checkpoint t9 on shadow — see /branches"
    assert _run_cmd(api, "undo", "p7", ctx) == "✗ no prompt p7 on shadow — see /branches"
    assert _run_cmd(api, "undo", "deadbeef", ctx).startswith("✗ no checkpoint deadbee")
    assert [c["turn"] for c in _ckpts(ctx)] == [1], "failed refs must not touch the branch"


def test_plain_numbers_still_count_back(session_api) -> None:
    api, ctx, repo = session_api
    _three_prompts(api, ctx, repo)
    assert _run_cmd(api, "undo", "2", ctx).startswith("↩ reverted 2 checkpoint(s) →")
    assert [c["turn"] for c in _ckpts(ctx)] == [1, 2]


def test_branch_before_prompt_keeps_grouping_on_both_lines(session_api) -> None:
    api, ctx, repo = session_api
    _three_prompts(api, ctx, repo)
    out = _run_cmd(api, "branch", "p2", ctx)
    assert out.startswith("⑂ branch-1 preserved")
    assert [c["prompt"] for c in _ckpts(ctx)] == [1, 1]

    # The preserved line is reconstructed from git — trailers carry the prompts.
    root = _panel(api).snapshot(ctx)
    sibling = next(n for n in _walk(root) if n.kind == "branch" and not n.data["active"])
    prompts = [c for c in sibling.children if c.kind == "prompt"]
    assert [p.data["prompt"] for p in prompts] == [3, 2, 1]
    assert prompts[1].label == 'p2 "write tests"'


# ---------------------------------------------------------------------------
# /diff
# ---------------------------------------------------------------------------


def test_diff_checkpoint_prompt_and_patch(session_api) -> None:
    api, ctx, repo = session_api
    _three_prompts(api, ctx, repo)

    latest = _run_cmd(api, "diff", "", ctx)
    assert latest.startswith('t4 edit_file · p3 "rename things"')
    assert "login.py" in latest

    p1 = _run_cmd(api, "diff", "p1", ctx)
    lines = p1.splitlines()
    assert lines[0] == 'p1 "add a login form" · 2 checkpoint(s)'
    assert lines[1].startswith(" ")  # --stat columns keep their indentation
    assert "app.py" in p1 and "login.py" in p1 and "test_login.py" not in p1

    patch = _run_cmd(api, "diff", "t3 --patch", ctx)
    assert "diff --git a/test_login.py b/test_login.py" in patch
    assert "+d" in patch

    assert _run_cmd(api, "diff", "p9", ctx) == "✗ no prompt p9 on shadow"


def test_diff_without_checkpoints(session_api) -> None:
    api, ctx, _repo = session_api
    assert _run_cmd(api, "diff", "", ctx) == "• no checkpoints yet"


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def test_branches_groups_by_prompt(session_api) -> None:
    api, ctx, repo = session_api
    _three_prompts(api, ctx, repo)
    lines = _run_cmd(api, "branches", "", ctx).splitlines()
    assert lines[1].startswith("└─ ● shadow/session-s1  (4 cp)  ◀ active")
    assert lines[2].startswith('   ├─ p3 "rename things"  1 cp · +1 −1')
    assert lines[3].startswith("   │  └─ t4") and lines[3].endswith("◀ tip")
    assert lines[4].startswith('   ├─ p2 "write tests"  1 cp · +2 −0')
    assert lines[6].startswith('   └─ p1 "add a login form"  2 cp')
    assert lines[7].startswith("      ├─ t2") and lines[8].startswith("      └─ t1")


def test_checkpoint_note_names_prompt_and_ref(session_api) -> None:
    from agent.core.events import ToolResult
    from agent.extensions.api import tool_result_notes

    api, ctx, repo = session_api
    _prompt(ctx, "do it")
    (repo / "a.txt").write_text("1\n", encoding="utf-8")
    tr = ToolResult(tool_name="write_file", output="ok")
    for handler in api.handlers["tool_result"]:
        handler(tr, ctx)
    sha = _ckpts(ctx)[0]["hash"]
    assert tool_result_notes(tr) == [f"⎇ checkpoint t1 (p1) · {sha} · 1 file +1 −0 · /undo t1"]


def test_panel_prompt_nodes_and_actions(session_api) -> None:
    from agent.extensions.api import UIInvocation

    api, ctx, repo = session_api
    _three_prompts(api, ctx, repo)
    panel = _panel(api)
    root = panel.snapshot(ctx)
    active = next(n for n in _walk(root) if n.kind == "branch" and n.data["active"])
    p3, p2, p1 = active.children
    assert [p.kind for p in (p3, p2, p1)] == ["prompt"] * 3
    assert p3.label == 'p3 "rename things" ●' and p3.style == "active"
    assert p1.data["n_back"] == 4 and p2.data["n_back"] == 2 and p3.data["n_back"] == 1
    assert [c.data["turn"] for c in p1.children] == [2, 1]
    assert {a.id for a in panel.actions_for(p2)} == {
        "undo_prompt",
        "retry_prompt",
        "diff",
        "refresh",
    }

    text = panel.describe(p1, ctx)
    assert text.splitlines()[0] == 'p1 "add a login form"'
    assert "app.py" in text and "drops 4 checkpoint(s)" in text

    diff = panel.action("diff").handler(UIInvocation(node=p2, ctx=ctx))
    assert "test_login.py" in diff

    out = panel.action("undo_prompt").handler(UIInvocation(node=p2, ctx=ctx, args={}))
    assert out.startswith("↩ reverted 2 checkpoint(s)")
    assert [c["prompt"] for c in _ckpts(ctx)] == [1, 1]

    root = panel.snapshot(ctx)
    p1 = next(n for n in _walk(root) if n.kind == "prompt")
    out = panel.action("retry_prompt").handler(UIInvocation(node=p1, ctx=ctx, args={}))
    assert out.startswith("⑂ branch-1 preserved")
    assert _ckpts(ctx) == []
