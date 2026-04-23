"""Aar extension: git checkpoint -- auto-commit/stash at turn boundaries."""
from __future__ import annotations

import asyncio
import subprocess
from typing import Any


def _run_git(*args: str, cwd: str | None = None) -> tuple[int, str]:
    """Run a git command and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


def _is_git_repo(cwd: str | None = None) -> bool:
    """Check if the current directory is inside a git repo."""
    rc, _ = _run_git("rev-parse", "--is-inside-work-tree", cwd=cwd)
    return rc == 0


def _has_changes(cwd: str | None = None) -> bool:
    """Check if there are uncommitted changes."""
    rc, output = _run_git("status", "--porcelain", cwd=cwd)
    return rc == 0 and bool(output)


def _create_checkpoint(turn: int, cwd: str | None = None) -> str | None:
    """Create an auto-commit checkpoint. Returns the commit hash or None."""
    rc, _ = _run_git("add", "-A", cwd=cwd)
    if rc != 0:
        return None
    rc, output = _run_git(
        "commit",
        "-m", f"aar: checkpoint after turn {turn}",
        "--no-verify",
        cwd=cwd,
    )
    if rc != 0:
        return None
    rc, sha = _run_git("rev-parse", "--short", "HEAD", cwd=cwd)
    return sha if rc == 0 else None


def register(api: Any) -> None:
    """Register the git-checkpoint extension."""

    turn_count = 0
    is_repo = False

    @api.on("session_start")
    def on_start(event: Any, ctx: Any) -> None:
        nonlocal is_repo
        is_repo = _is_git_repo()
        if is_repo:
            ctx.logger.info("git-checkpoint: git repo detected, checkpoints enabled")
        else:
            ctx.logger.info("git-checkpoint: no git repo found, checkpoints disabled")

    @api.on("after_turn")
    def on_after_turn(event: Any, ctx: Any) -> None:
        nonlocal turn_count
        if not is_repo:
            return
        turn_count += 1
        if not _has_changes():
            ctx.logger.debug("git-checkpoint: no changes after turn %d", turn_count)
            return
        sha = _create_checkpoint(turn_count)
        if sha:
            ctx.logger.info("git-checkpoint: created checkpoint %s after turn %d", sha, turn_count)
        else:
            ctx.logger.warning("git-checkpoint: failed to create checkpoint after turn %d", turn_count)

    @api.tool(
        name="git_rollback",
        description="Roll back to the last git checkpoint created by the git-checkpoint extension",
        input_schema={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of checkpoints to roll back (default: 1)",
                    "default": 1,
                }
            },
        },
    )
    def rollback(steps: int = 1, **kwargs: Any) -> str:
        if not is_repo:
            return "Error: not in a git repository"
        rc, _ = _run_git("reset", "--hard", f"HEAD~{steps}")
        if rc != 0:
            return f"Error: git reset failed (maybe not enough checkpoints?)"
        rc, sha = _run_git("rev-parse", "--short", "HEAD")
        return f"Rolled back {steps} checkpoint(s). Now at {sha}"

    api.append_system_prompt(
        "The git-checkpoint extension is active. "
        "Changes are auto-committed after each turn. "
        "Use the git_rollback tool to undo recent changes."
    )
