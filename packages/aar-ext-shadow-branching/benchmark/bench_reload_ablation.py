"""Reload-mechanism ablation harness.

Compares the protocol with and without the ``ReloadSession`` step on a
synthetic switch-and-fork scenario.  Reports the divergence between the
in-memory event stream and the files actually present on the
currently-checked-out branch.

Hypothesis (paper §4):
    * with reload=on  -> divergence is always 0
    * with reload=off -> divergence grows with each branch operation

Usage:
    python bench_reload_ablation.py --turns 16 --branches 3 --output results/

This is a self-contained harness.  It does **not** import the real Aar
``SessionStore``; instead it ships a tiny per-branch JSONL persistence
emulating the shape of one, plus a switchable reload hook.  This isolates
the reload mechanism while the extension's pytest suite covers the real
``SessionStore`` integration.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Allow running from the source tree without installation.
HERE = Path(__file__).resolve().parent
PKG_ROOT = (
    HERE / ".." / ".." / ".." / "aar-extensions-registry" / "packages" / "aar-ext-shadow-branching"
).resolve()
if PKG_ROOT.exists():
    sys.path.insert(0, str(PKG_ROOT))

import aar_ext_shadow_branching as ext  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
LOG = logging.getLogger("ablation")

EVENTS_FILE = "session.jsonl"


# ---------------------------------------------------------------------------
# Tiny FakeAPI / Session mirroring tests/test_shadow_branching.py
# ---------------------------------------------------------------------------


class FakeAPI:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.commands: dict[str, tuple[str, Any]] = {}

    def on(self, event):
        def deco(fn):
            self.handlers.setdefault(event, []).append(fn)
            return fn

        return deco

    def command(self, name, *, description=""):
        def deco(fn):
            self.commands[name] = (description, fn)
            return fn

        return deco

    def tool(self, *a, **kw):
        return lambda fn: fn

    def append_system_prompt(self, _text):
        pass


@dataclass
class FakeSession:
    session_id: str
    events: list[dict] = field(default_factory=list)
    step_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class FakeCtx:
    session: FakeSession
    logger: logging.Logger = field(default_factory=lambda: LOG)


# ---------------------------------------------------------------------------
# Branch-local event persistence (stand-in for SessionStore)
# ---------------------------------------------------------------------------


def persist_events(repo: Path, session: FakeSession) -> None:
    """Write the session's events to ``session.jsonl`` in the working tree."""
    payload = "\n".join(json.dumps(e) for e in session.events) + "\n"
    (repo / EVENTS_FILE).write_text(payload, encoding="utf-8")


def reload_events_from_disk(session: FakeSession, repo: Path) -> bool:
    """Replace ``session.events`` with whatever is in ``session.jsonl`` on the
    currently checked-out branch.  This is the harness-local stand-in for the
    extension's ``reload_session_from_disk``."""
    p = repo / EVENTS_FILE
    if not p.exists():
        session.events.clear()
        session.step_count = 0
        return True
    session.events = [
        json.loads(line) for line in p.read_text("utf-8").splitlines() if line.strip()
    ]
    session.step_count = len(session.events)
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-b", "main", cwd=path)
    _git("config", "user.name", "Bench", cwd=path)
    _git("config", "user.email", "bench@example.com", cwd=path)
    _git("config", "commit.gpgsign", "false", cwd=path)
    (path / "README.md").write_text("initial\n", encoding="utf-8")
    _git("add", "README.md", cwd=path)
    _git("commit", "-m", "initial", cwd=path)


def fire(api: FakeAPI, event: str, ctx: FakeCtx, evt: Any = None) -> None:
    for h in api.handlers.get(event, []):
        h(evt, ctx)


def do_tool(api: FakeAPI, ctx: FakeCtx, repo: Path, path: str, content: str) -> None:
    """Simulate a tool call: write a file, append an event, fire the
    tool_result hook (which produces a ``shadow-auto`` checkpoint), then
    persist events to disk."""
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    ctx.session.events.append({"type": "tool_result", "tool": "write_file", "path": path})
    fire(api, "tool_result", ctx, SimpleNamespace(tool_name="write_file"))
    # Emulate the post-turn save by writing the events JSONL and letting the
    # extension's session_end sweep capture it as a shadow-meta commit.
    persist_events(repo, ctx.session)
    fire(api, "session_end", ctx, SimpleNamespace(action="ended"))


def run_cmd(api: FakeAPI, name: str, args: str, ctx: FakeCtx) -> str | None:
    return api.commands[name][1](args, ctx)


def divergence(session: FakeSession, repo: Path) -> int:
    """How many file paths the in-memory events reference that no longer
    exist on disk on the currently-checked-out branch?"""
    out = _git("ls-files", cwd=repo).stdout
    on_disk = {line.strip() for line in out.splitlines() if line.strip()}
    referenced = {e.get("path") for e in session.events if e.get("path")}
    return len(referenced - on_disk - {EVENTS_FILE})


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def run_scenario(reload_enabled: bool, turns: int, branches: int, work_dir: Path) -> list[dict]:
    """Drive a switch-and-fork scenario and return per-step measurements."""
    repo = work_dir / ("on" if reload_enabled else "off")
    _init_repo(repo)

    # Patch the extension's reload to either do its job (using our local
    # JSONL) or be a no-op.  The extension's own implementation requires the
    # real Aar SessionStore; we override it here so the harness is
    # self-contained.
    def _reload_real(session, logger=None):
        return reload_events_from_disk(session, repo)

    def _reload_noop(session, logger=None):
        return True

    ext.reload_session_from_disk = _reload_real if reload_enabled else _reload_noop

    api = FakeAPI()
    ext.register(api)
    session = FakeSession(session_id="bench")
    ctx = FakeCtx(session)

    # Run from the repo dir so the extension uses it as cwd.
    import os

    os.chdir(repo)
    fire(api, "session_start", ctx)

    measurements: list[dict] = []
    per_phase = max(1, turns // (branches + 1))
    branch_op_index = 0

    def measure(step: str) -> None:
        measurements.append(
            {
                "condition": "on" if reload_enabled else "off",
                "step": step,
                "branch_op_index": branch_op_index,
                "divergence": divergence(session, repo),
                "events_in_memory": len(session.events),
                "files_on_disk": len(
                    [f for f in (_git("ls-files", cwd=repo).stdout.splitlines()) if f.strip()]
                ),
            }
        )

    # Phase 0 - work on the canonical shadow branch
    for i in range(per_phase):
        do_tool(api, ctx, repo, f"phase0/file_{i:03}.txt", f"v0-{i}")
    measure("after_phase_0")

    # Branch operations
    for b in range(branches):
        rewind = max(1, per_phase // 2)
        run_cmd(api, "branch", str(rewind), ctx)
        branch_op_index += 1
        measure(f"after_branch_{b}")
        for i in range(per_phase):
            do_tool(api, ctx, repo, f"phase{b + 1}/file_{i:03}.txt", f"v{b + 1}-{i}")
        measure(f"after_phase_{b + 1}")
        # Switch back to a preserved fork (branch-1 always exists after first /branch)
        if b >= 1:
            run_cmd(api, "switch", "1", ctx)
            branch_op_index += 1
            measure(f"after_switch_{b}")

    return measurements


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--turns", type=int, default=16, help="Total number of tool calls per condition"
    )
    ap.add_argument(
        "--branches", type=int, default=3, help="Number of /branch operations to perform"
    )
    ap.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    saved_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(
            prefix="shadow-ablation-", ignore_cleanup_errors=True
        ) as tmp:
            work = Path(tmp)
            for cond in (True, False):
                rows.extend(run_scenario(cond, args.turns, args.branches, work))
            os.chdir(saved_cwd)  # release tempdir handles before cleanup (Windows)
    finally:
        os.chdir(saved_cwd)

    out_csv = args.output / "reload_ablation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")

    # Markdown summary
    print("\n## Reload ablation - divergence by step\n")
    print("| step | reload=on | reload=off |")
    print("|---|---:|---:|")
    steps = []
    for r in rows:
        if r["condition"] == "on" and r["step"] not in steps:
            steps.append(r["step"])
    by_step = {(r["step"], r["condition"]): r["divergence"] for r in rows}
    for step in steps:
        on = by_step.get((step, "on"), "-")
        off = by_step.get((step, "off"), "-")
        print(f"| `{step}` | {on} | {off} |")

    print(
        "\nClaim: with reload=on, every divergence cell is 0; "
        "with reload=off, divergence is non-zero after the first branch op.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
