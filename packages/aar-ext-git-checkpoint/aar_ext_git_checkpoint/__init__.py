"""Aar extension: git checkpoint -- auto-commit/stash at turn boundaries.

Registers an `after_turn` hook that creates a git commit ("checkpoint") after
every agent turn that leaves uncommitted changes.  If git is not configured
with a user identity (name + email) the extension logs a clear one-time
warning instead of silently failing on every turn.

Tool exposed to the LLM:
  - ``git_rollback`` — reset HEAD back N checkpoints
"""

from __future__ import annotations

import subprocess
from typing import Any

# ---------------------------------------------------------------------------
# Internal git helpers
# ---------------------------------------------------------------------------


def _run_git(*args: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "git command timed out"
    except FileNotFoundError:
        return 1, "", "git executable not found"


def _is_git_repo(cwd: str | None = None) -> bool:
    rc, _, _ = _run_git("rev-parse", "--is-inside-work-tree", cwd=cwd)
    return rc == 0


def _has_changes(cwd: str | None = None) -> bool:
    rc, output, _ = _run_git("status", "--porcelain", cwd=cwd)
    return rc == 0 and bool(output)


def _has_identity(cwd: str | None = None) -> bool:
    """Return True if git user.name and user.email are configured."""
    rc_name, name, _ = _run_git("config", "user.name", cwd=cwd)
    rc_email, email, _ = _run_git("config", "user.email", cwd=cwd)
    return rc_name == 0 and bool(name) and rc_email == 0 and bool(email)


def _create_checkpoint(turn: int, cwd: str | None = None) -> tuple[str | None, str]:
    """
    Stage all changes and create an auto-commit.

    Returns ``(sha, error_message)``.  On success ``sha`` is the short commit
    hash and ``error_message`` is empty.  On failure ``sha`` is ``None`` and
    ``error_message`` contains the git stderr for diagnosis.
    """
    rc, _, stderr = _run_git("add", "-A", cwd=cwd)
    if rc != 0:
        return None, f"git add -A failed: {stderr or '(no output)'}"

    rc, _, stderr = _run_git(
        "commit",
        "-m",
        f"aar: checkpoint after turn {turn}",
        "--no-verify",
        cwd=cwd,
    )
    if rc != 0:
        return None, f"git commit failed: {stderr or '(no output)'}"

    rc, sha, stderr = _run_git("rev-parse", "--short", "HEAD", cwd=cwd)
    if rc != 0:
        return None, f"git rev-parse failed: {stderr or '(no output)'}"
    return sha, ""


# ---------------------------------------------------------------------------
# Extension entry-point
# ---------------------------------------------------------------------------


def register(api: Any) -> None:
    """Register the git-checkpoint extension."""

    turn_count = 0
    is_repo = False
    identity_ok = True  # set to False once; suppresses repeated warnings
    _identity_warned = False  # guard: only log the identity warning once

    @api.on("session_start")
    def on_start(event: Any, ctx: Any) -> None:
        nonlocal is_repo, identity_ok, _identity_warned, turn_count

        # Reset turn counter for new sessions
        turn_count = 0

        is_repo = _is_git_repo()
        if not is_repo:
            ctx.logger.info("git-checkpoint: no git repo found, checkpoints disabled")
            return

        ctx.logger.info("git-checkpoint: git repo detected")

        # Check identity once at session start
        identity_ok = _has_identity()
        if not identity_ok:
            _identity_warned = True
            ctx.logger.warning(
                "git-checkpoint: git user identity not configured — checkpoints disabled.\n"
                "  Fix with:\n"
                '    git config user.name  "Your Name"\n'
                '    git config user.email "you@example.com"\n'
                "  (or set them globally with --global)"
            )
        else:
            ctx.logger.info("git-checkpoint: checkpoints enabled")

    @api.on("after_turn")
    def on_after_turn(event: Any, ctx: Any) -> None:
        nonlocal turn_count

        if not is_repo:
            return

        if not identity_ok:
            # Warn once per session rather than once per turn to avoid log spam
            if not _identity_warned:
                ctx.logger.warning(
                    "git-checkpoint: skipping checkpoint — git identity not configured"
                )
            return

        turn_count += 1

        if not _has_changes():
            ctx.logger.debug(
                "git-checkpoint: no uncommitted changes after turn %d, skipping", turn_count
            )
            return

        sha, err = _create_checkpoint(turn_count)
        if sha:
            ctx.logger.info("git-checkpoint: checkpoint %s created after turn %d", sha, turn_count)
        else:
            ctx.logger.warning(
                "git-checkpoint: failed to create checkpoint after turn %d — %s",
                turn_count,
                err,
            )

    @api.tool(
        name="git_rollback",
        description=(
            "Roll back to the last git checkpoint created by the git-checkpoint extension. "
            "Each 'step' undoes one checkpoint commit (hard reset)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of checkpoint commits to roll back (default: 1).",
                    "default": 1,
                    "minimum": 1,
                }
            },
            "additionalProperties": False,
        },
    )
    def rollback(steps: int = 1, **kwargs: Any) -> str:
        if not is_repo:
            return "Error: not in a git repository."
        if not identity_ok:
            return "Error: git identity not configured — no checkpoints were created."

        rc, _, stderr = _run_git("reset", "--hard", f"HEAD~{steps}")
        if rc != 0:
            reason = stderr or "(no output)"
            return (
                f"Error: git reset --hard HEAD~{steps} failed: {reason}\n"
                "There may not be enough checkpoint commits to roll back that many steps."
            )

        rc, sha, _ = _run_git("rev-parse", "--short", "HEAD")
        current = sha if rc == 0 else "unknown"
        return f"Rolled back {steps} checkpoint(s). Now at {current}."

    api.append_system_prompt(
        "The git-checkpoint extension is active. "
        "All changes are auto-committed after each agent turn (when inside a git repo with "
        "user identity configured). "
        "Use the git_rollback tool to undo recent changes by rolling back N checkpoint commits."
    )
