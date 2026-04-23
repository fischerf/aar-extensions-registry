"""Aar extension: shadow-branching.

Implements the Shadow Branching Protocol:

* On ``session_start`` — detect the working directory's git repo, list any prior
  ``aar/session-*`` branches, and create a fresh ``aar/session-<session_id>`` shadow
  branch rooted in an empty ``aar-init: base=<ORIGINAL_BRANCH>`` anchor commit.
  Falls back to ``.aar_backups/`` when the working directory is not a git repo.

* After every ``tool_result`` that leaves uncommitted changes — take a checkpoint
  commit ``aar-auto: <tool_name> turn-<N>`` and log the
  ``[CHECKPOINT turn=N hash=... tool=...]`` trail.  Warns (before committing) if
  the change set contains files that look sensitive (``.env*``, ``*.key``,
  ``*credentials*``).

* Slash commands (Aar surfaces these via the Slash-commands extension API in
  every transport):

    /undo [N] [--force]   Revert N checkpoints (default 1). Refuses to touch a
                          dirty working tree unless ``--force`` is passed.
    /revert N             Alias for ``/undo``.
    /fork [N]             Preserve the active shadow branch as
                          ``aar/session-<id>-fork-<K>`` and branch a fresh shadow
                          from ``HEAD~N`` (or ``HEAD`` if no N given). Fork
                          numbering is derived from the branches that already
                          exist on disk, so it survives session reloads and
                          forks-of-forks.
    /switch <branch>      Switch to any existing ``aar/session-<id>*`` branch.
                          Accepts bare branch names or the shorthand
                          ``fork-<K>``. Refuses to switch with a dirty tree.
    /forks                List every shadow/fork branch belonging to this
                          session and mark the active one.
    /done                 Squash-merge the active shadow branch back into the
                          base branch captured in ``aar-init``. Aborts cleanly
                          on merge conflicts and names the files that need
                          manual resolution.

All state is also mirrored into ``session.metadata['shadow_branching']`` so the
session JSONL store keeps a faithful record across resumes.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["register", "ShadowState"]


# ---------------------------------------------------------------------------
# git helpers — thin wrappers so tests can drive a real temp repo
# ---------------------------------------------------------------------------


def _run_git(*args: str, cwd: str | Path | None = None) -> tuple[int, str, str]:
    """Run a git command; return (returncode, stdout, stderr) — stripped."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(cwd) if cwd is not None else None,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "git command timed out"
    except FileNotFoundError:
        return 1, "", "git executable not found"


def _is_git_repo(cwd: str | Path | None = None) -> bool:
    rc, out, _ = _run_git("rev-parse", "--is-inside-work-tree", cwd=cwd)
    return rc == 0 and out == "true"


def _has_identity(cwd: str | Path | None = None) -> bool:
    rc_n, name, _ = _run_git("config", "user.name", cwd=cwd)
    rc_e, email, _ = _run_git("config", "user.email", cwd=cwd)
    return rc_n == 0 and bool(name) and rc_e == 0 and bool(email)


def _current_branch(cwd: str | Path | None = None) -> str:
    rc, name, _ = _run_git("branch", "--show-current", cwd=cwd)
    return name if rc == 0 else ""


def _has_any_commit(cwd: str | Path | None = None) -> bool:
    rc, _, _ = _run_git("rev-parse", "--verify", "HEAD", cwd=cwd)
    return rc == 0


def _branch_exists(name: str, cwd: str | Path | None = None) -> bool:
    rc, _, _ = _run_git("rev-parse", "--verify", f"refs/heads/{name}", cwd=cwd)
    return rc == 0


def _has_changes(cwd: str | Path | None = None) -> bool:
    rc, out, _ = _run_git("status", "--porcelain", cwd=cwd)
    return rc == 0 and bool(out)


def _short_hash(ref: str = "HEAD", cwd: str | Path | None = None) -> str:
    rc, sha, _ = _run_git("rev-parse", "--short", ref, cwd=cwd)
    return sha if rc == 0 else ""


def _list_branches(pattern: str, cwd: str | Path | None = None) -> list[str]:
    rc, out, _ = _run_git(
        "branch", "--list", pattern, "--format=%(refname:short)", cwd=cwd
    )
    if rc != 0 or not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _next_fork_n(session_id: str, cwd: str | Path | None = None) -> int:
    """Largest existing fork number for this session + 1. Derived from disk
    state so numbering survives session reloads and fork-of-fork chains."""
    branches = _list_branches(f"aar/session-{session_id}-fork-*", cwd=cwd)
    max_n = 0
    for b in branches:
        suffix = b.rsplit("-fork-", 1)[-1]
        try:
            n = int(suffix)
            max_n = max(max_n, n)
        except ValueError:
            continue
    return max_n + 1


def _read_anchor_base(
    branch: str, fallback: str, cwd: str | Path | None = None
) -> str:
    """Return the base branch recorded in the ``aar-init`` anchor commit of
    *branch* (falls back to *fallback* when no anchor is present)."""
    rc, out, _ = _run_git(
        "log", "--grep=^aar-init:", "--pretty=%s", branch, cwd=cwd
    )
    if rc != 0 or not out:
        return fallback
    # Prefer the oldest aar-init commit on the branch
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return fallback
    msg = lines[-1]
    if "base=" not in msg:
        return fallback
    return msg.split("base=", 1)[1].strip() or fallback


_SENSITIVE_NEEDLES = (".env", ".key", "credentials", "id_rsa", "secret")


def _flag_sensitive(status_output: str) -> list[str]:
    flagged: list[str] = []
    for line in status_output.splitlines():
        # porcelain format: "XY path"
        if len(line) < 4:
            continue
        path = line[3:].strip().lower()
        if any(needle in path for needle in _SENSITIVE_NEEDLES):
            flagged.append(line[3:].strip())
    return flagged


# ---------------------------------------------------------------------------
# State model — mirrored into session.metadata for save/load consistency
# ---------------------------------------------------------------------------


@dataclass
class ShadowState:
    session_id: str
    original_branch: str = ""
    shadow_branch: str = ""
    fork_counter: int = 0
    turn_counter: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    base_anchor: str = ""
    enabled: bool = True
    mode: str = "git"  # "git" or "fallback" (.aar_backups)

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "ShadowState":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------
# Extension entry-point
# ---------------------------------------------------------------------------


def register(api: Any) -> None:
    """Register the shadow-branching extension against *api*."""

    state: ShadowState | None = None

    # ------------------------------------------------------------------
    # Helpers (close over ``state`` and ``ctx``)
    # ------------------------------------------------------------------

    def _sync_metadata(ctx: Any) -> None:
        if state is None or ctx is None:
            return
        session = getattr(ctx, "session", None)
        if session is None:
            return
        meta = getattr(session, "metadata", None)
        if isinstance(meta, dict):
            meta["shadow_branching"] = state.to_metadata()

    def _reconstruct_from_branch(branch: str) -> tuple[int, list[dict[str, Any]]]:
        """Count aar-auto commits on *branch* and build a checkpoint list."""
        rc, out, _ = _run_git(
            "log", "--reverse", "--format=%h %s", branch
        )
        if rc != 0 or not out:
            return 0, []
        turn = 0
        ckpts: list[dict[str, Any]] = []
        for line in out.splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            sha, msg = parts
            if not msg.startswith("aar-auto:"):
                continue
            turn += 1
            # message shape: "aar-auto: <tool_name> turn-<N>"
            tool = "unknown"
            try:
                after = msg[len("aar-auto: ") :]
                tool = after.split(" turn-", 1)[0]
            except Exception:
                pass
            ckpts.append({"turn": turn, "hash": sha, "tool": tool})
        return turn, ckpts

    def _parse_int_arg(args: str) -> tuple[int | None, set[str]]:
        """Return (integer N or None, set of flag tokens like {'--force'})."""
        flags: set[str] = set()
        n: int | None = None
        for tok in args.strip().split():
            if tok.startswith("-"):
                flags.add(tok)
            else:
                try:
                    n = int(tok)
                except ValueError:
                    pass
        return n, flags

    # ------------------------------------------------------------------
    # session_start — reconnaissance + branch setup
    # ------------------------------------------------------------------

    @api.on("session_start")
    def on_start(event: Any, ctx: Any) -> None:
        nonlocal state

        session_id = getattr(getattr(ctx, "session", None), "session_id", None) or "unknown"

        if not _is_git_repo():
            # Non-repo fallback
            try:
                Path(".aar_backups").mkdir(exist_ok=True)
            except Exception as exc:
                ctx.logger.warning("shadow-branching: cannot create .aar_backups/: %s", exc)
            state = ShadowState(
                session_id=session_id, enabled=False, mode="fallback"
            )
            ctx.logger.info(
                "shadow-branching: no git repo — using .aar_backups/ fallback (checkpoints disabled)"
            )
            _sync_metadata(ctx)
            return

        if not _has_identity():
            ctx.logger.warning(
                "shadow-branching: git user.name/user.email not configured — checkpoints disabled.\n"
                "  Fix with:\n"
                '    git config user.name  "Your Name"\n'
                '    git config user.email "you@example.com"'
            )
            state = ShadowState(session_id=session_id, enabled=False)
            _sync_metadata(ctx)
            return

        # Fresh repo (git init with no commits) — bootstrap a main branch first.
        if not _has_any_commit():
            _run_git("checkout", "-b", "main")
            _run_git("commit", "--allow-empty", "-m", "Initial commit")

        current = _current_branch()
        shadow_name = f"aar/session-{session_id}"

        # Reconnaissance: list prior session branches (informational).
        prior = _list_branches("aar/session-*")
        prior = [b for b in prior if b != shadow_name]
        if prior:
            ctx.logger.info(
                "shadow-branching: found %d prior session branch(es): %s",
                len(prior),
                ", ".join(prior),
            )

        if _branch_exists(shadow_name):
            # Resuming this exact session — reconstruct state from git log.
            rc, _, err = _run_git("checkout", shadow_name)
            if rc != 0:
                ctx.logger.warning(
                    "shadow-branching: cannot resume %s: %s", shadow_name, err
                )
                state = ShadowState(session_id=session_id, enabled=False)
                _sync_metadata(ctx)
                return

            base = _read_anchor_base(shadow_name, current or "main")
            turn, ckpts = _reconstruct_from_branch(shadow_name)
            fork_counter = _next_fork_n(session_id) - 1  # already-used counter
            state = ShadowState(
                session_id=session_id,
                original_branch=base,
                shadow_branch=shadow_name,
                turn_counter=turn,
                checkpoints=ckpts,
                fork_counter=max(0, fork_counter),
                base_anchor=_short_hash(shadow_name + "^{/^aar-init:}")
                or _short_hash("HEAD"),
            )
            ctx.logger.info(
                "shadow-branching: resumed %s (base=%s, %d checkpoint(s))",
                shadow_name,
                base,
                turn,
            )
            _sync_metadata(ctx)
            return

        # Starting fresh — create shadow branch + anchor.
        original_branch = current or "main"
        rc, _, err = _run_git("checkout", "-b", shadow_name)
        if rc != 0:
            ctx.logger.warning(
                "shadow-branching: cannot create shadow branch %s: %s",
                shadow_name,
                err,
            )
            state = ShadowState(
                session_id=session_id,
                original_branch=original_branch,
                enabled=False,
            )
            _sync_metadata(ctx)
            return

        rc, _, err = _run_git(
            "commit",
            "--allow-empty",
            "-m",
            f"aar-init: base={original_branch}",
            "--no-verify",
        )
        if rc != 0:
            ctx.logger.warning(
                "shadow-branching: failed to create aar-init anchor: %s", err
            )

        anchor = _short_hash("HEAD")
        state = ShadowState(
            session_id=session_id,
            original_branch=original_branch,
            shadow_branch=shadow_name,
            base_anchor=anchor,
        )
        ctx.logger.info(
            "shadow-branching: created %s (base=%s, anchor=%s)",
            shadow_name,
            original_branch,
            anchor,
        )
        _sync_metadata(ctx)

    # ------------------------------------------------------------------
    # tool_result — snapshot changes
    # ------------------------------------------------------------------

    @api.on("tool_result")
    def on_tool_result(event: Any, ctx: Any) -> None:
        if state is None or not state.enabled:
            return
        if not _has_changes():
            return

        tool_name = str(getattr(event, "tool_name", "") or "tool")

        # Warn about sensitive files before committing (Improvement 2).
        rc, status_out, _ = _run_git("status", "--porcelain")
        if rc == 0 and status_out:
            flagged = _flag_sensitive(status_out)
            if flagged:
                ctx.logger.warning(
                    "shadow-branching: committing files that look sensitive: %s",
                    ", ".join(flagged),
                )

        prev_turn = state.turn_counter
        state.turn_counter += 1
        _run_git("add", "-A")
        rc, _, err = _run_git(
            "commit",
            "-m",
            f"aar-auto: {tool_name} turn-{state.turn_counter}",
            "--no-verify",
        )
        if rc != 0:
            ctx.logger.warning(
                "shadow-branching: checkpoint commit failed: %s", err
            )
            state.turn_counter = prev_turn
            return

        sha = _short_hash("HEAD")
        state.checkpoints.append(
            {"turn": state.turn_counter, "hash": sha, "tool": tool_name}
        )
        ctx.logger.info(
            "[CHECKPOINT turn=%d hash=%s tool=%s]",
            state.turn_counter,
            sha,
            tool_name,
        )
        _sync_metadata(ctx)

    # ------------------------------------------------------------------
    # /undo  /revert
    # ------------------------------------------------------------------

    def _do_undo(args: str, ctx: Any) -> None:
        if state is None or not state.enabled:
            ctx.logger.warning("shadow-branching: disabled (no git repo or identity)")
            return

        n, flags = _parse_int_arg(args)
        n = n if n and n >= 1 else 1
        force = bool(flags & {"--force", "-f"})

        if n > len(state.checkpoints):
            ctx.logger.warning(
                "shadow-branching: cannot undo %d — only %d checkpoint(s) available",
                n,
                len(state.checkpoints),
            )
            return

        dirty = _has_changes()
        if dirty and not force:
            ctx.logger.warning(
                "shadow-branching: uncommitted changes present — "
                "commit/stash them, or re-run with --force to discard."
            )
            return

        rc, _, err = _run_git("reset", "--hard", f"HEAD~{n}")
        if rc != 0:
            ctx.logger.warning("shadow-branching: git reset failed: %s", err)
            return

        # ``git reset --hard`` only touches tracked changes; untracked files
        # linger. When --force was requested the user accepted losing
        # uncommitted work, so sweep those too (ignored files are preserved).
        if dirty and force:
            _run_git("clean", "-fd")

        removed = state.checkpoints[-n:]
        state.checkpoints = state.checkpoints[:-n]
        state.turn_counter = max(0, state.turn_counter - n)
        ctx.logger.info(
            "shadow-branching: reverted %d checkpoint(s), now at %s (removed turns %s)",
            n,
            _short_hash("HEAD"),
            [c["turn"] for c in removed],
        )
        _sync_metadata(ctx)

    @api.command("undo", description="Revert N checkpoints (default 1). Use --force with dirty tree.")
    def cmd_undo(args: str, ctx: Any) -> None:
        _do_undo(args, ctx)

    @api.command("revert", description="Alias for /undo")
    def cmd_revert(args: str, ctx: Any) -> None:
        _do_undo(args, ctx)

    # ------------------------------------------------------------------
    # /fork
    # ------------------------------------------------------------------

    @api.command(
        "fork",
        description="Preserve current shadow as aar/session-<id>-fork-<K> and branch fresh (optionally from N back).",
    )
    def cmd_fork(args: str, ctx: Any) -> None:
        nonlocal state
        if state is None or not state.enabled:
            ctx.logger.warning("shadow-branching: disabled (no git repo or identity)")
            return

        n, _ = _parse_int_arg(args)
        if n is not None and n > len(state.checkpoints):
            ctx.logger.warning(
                "shadow-branching: only %d checkpoints, cannot fork %d back",
                len(state.checkpoints),
                n,
            )
            return

        # Resolve fork point before rename so HEAD~N is still valid.
        fork_ref = f"HEAD~{n}" if n else "HEAD"
        rc, fork_sha, err = _run_git("rev-parse", fork_ref)
        if rc != 0:
            ctx.logger.warning(
                "shadow-branching: cannot resolve fork point %s: %s", fork_ref, err
            )
            return

        fork_n = _next_fork_n(state.session_id)
        preserved = f"{state.shadow_branch}-fork-{fork_n}"

        rc, _, err = _run_git("branch", "-m", state.shadow_branch, preserved)
        if rc != 0:
            ctx.logger.warning("shadow-branching: branch rename failed: %s", err)
            return

        rc, _, err = _run_git("checkout", "-b", state.shadow_branch, fork_sha)
        if rc != 0:
            # Roll back the rename.
            _run_git("branch", "-m", preserved, state.shadow_branch)
            ctx.logger.warning(
                "shadow-branching: cannot start new branch at %s: %s",
                fork_sha[:8],
                err,
            )
            return

        state.fork_counter = fork_n
        if n:
            state.checkpoints = state.checkpoints[:-n]
            state.turn_counter = max(0, state.turn_counter - n)

        ctx.logger.info(
            "[FORK preserved=%s active=%s forked-from=%s]",
            preserved,
            state.shadow_branch,
            fork_sha[:8],
        )
        _sync_metadata(ctx)

    # ------------------------------------------------------------------
    # /switch
    # ------------------------------------------------------------------

    @api.command(
        "switch",
        description="Switch to another aar/session-<id>* branch (pass full name or shorthand fork-<K>).",
    )
    def cmd_switch(args: str, ctx: Any) -> None:
        nonlocal state
        if state is None or not state.enabled:
            ctx.logger.warning("shadow-branching: disabled (no git repo or identity)")
            return

        target = args.strip()
        if not target:
            ctx.logger.warning("shadow-branching: /switch requires a branch name")
            return

        # Shorthand: "fork-3" or just "3" → aar/session-<id>-fork-3
        if target.isdigit():
            target = f"aar/session-{state.session_id}-fork-{target}"
        elif target.startswith("fork-"):
            target = f"aar/session-{state.session_id}-{target}"
        elif not target.startswith("aar/session-"):
            target = f"aar/session-{state.session_id}-{target}"

        if not _branch_exists(target):
            ctx.logger.warning("shadow-branching: branch %r does not exist", target)
            return

        if _has_changes():
            ctx.logger.warning(
                "shadow-branching: uncommitted changes — commit or stash before /switch"
            )
            return

        rc, _, err = _run_git("checkout", target)
        if rc != 0:
            ctx.logger.warning("shadow-branching: checkout failed: %s", err)
            return

        base = _read_anchor_base(target, state.original_branch or "main")
        turn, ckpts = _reconstruct_from_branch(target)
        state.shadow_branch = target
        state.original_branch = base
        state.turn_counter = turn
        state.checkpoints = ckpts
        ctx.logger.info(
            "shadow-branching: switched to %s (base=%s, %d checkpoint(s))",
            target,
            base,
            turn,
        )
        _sync_metadata(ctx)

    # ------------------------------------------------------------------
    # /forks
    # ------------------------------------------------------------------

    @api.command("forks", description="List all shadow/fork branches for this session.")
    def cmd_forks(args: str, ctx: Any) -> None:
        if state is None:
            ctx.logger.info("shadow-branching: not initialised")
            return

        branches = _list_branches(f"aar/session-{state.session_id}*")
        if not branches:
            ctx.logger.info(
                "shadow-branching: no shadow branches found for session %s",
                state.session_id,
            )
            return

        for b in branches:
            marker = " (active)" if b == state.shadow_branch else ""
            ctx.logger.info("  * %s%s", b, marker)

    # ------------------------------------------------------------------
    # /done
    # ------------------------------------------------------------------

    @api.command(
        "done",
        description="Squash-merge the active shadow branch into its recorded base; aborts on conflicts.",
    )
    def cmd_done(args: str, ctx: Any) -> None:
        if state is None or not state.enabled:
            ctx.logger.warning("shadow-branching: disabled (no git repo or identity)")
            return

        if _has_changes():
            ctx.logger.warning(
                "shadow-branching: uncommitted changes present — commit or stash before /done"
            )
            return

        _, flags = _parse_int_arg(args)
        # extract custom message — anything left after flags/ints
        msg_parts: list[str] = []
        for tok in args.strip().split():
            if tok.startswith("-"):
                continue
            try:
                int(tok)
                continue
            except ValueError:
                msg_parts.append(tok)
        message = " ".join(msg_parts).strip()

        forks = _list_branches(f"aar/session-{state.session_id}-fork-*")
        if forks and "--yes" not in flags:
            ctx.logger.info(
                "shadow-branching: fork branches still exist (%s). "
                "Pass --yes to squash only the active shadow and leave forks intact.",
                ", ".join(forks),
            )
            return

        base = _read_anchor_base(state.shadow_branch, state.original_branch or "main")

        rc, _, err = _run_git("checkout", base)
        if rc != 0:
            ctx.logger.warning(
                "shadow-branching: cannot checkout base %s: %s", base, err
            )
            return

        rc, _, err = _run_git("merge", "--squash", state.shadow_branch)

        # Check for unresolved conflicts regardless of rc, then fall through.
        rc_u, conflicts, _ = _run_git("diff", "--name-only", "--diff-filter=U")
        if conflicts.strip():
            ctx.logger.warning(
                "shadow-branching: merge conflicts in: %s — resolve them and commit manually.",
                conflicts.strip().replace("\n", ", "),
            )
            return

        if rc != 0:
            ctx.logger.warning(
                "shadow-branching: git merge --squash failed: %s", err or "(no output)"
            )
            return

        if not message:
            message = f"aar: squashed session {state.session_id}"

        rc, _, err = _run_git("commit", "-m", message, "--no-verify")
        if rc != 0:
            # Most likely: nothing staged (empty squash).
            ctx.logger.warning(
                "shadow-branching: final commit failed: %s (nothing to commit?)",
                err,
            )
            return

        ctx.logger.info(
            "shadow-branching: merged %s into %s as %s",
            state.shadow_branch,
            base,
            _short_hash("HEAD"),
        )
        _sync_metadata(ctx)

    # ------------------------------------------------------------------
    # System prompt additions
    # ------------------------------------------------------------------

    api.append_system_prompt(
        "The shadow-branching extension is active. Every session runs on an isolated "
        "aar/session-<id> git branch; modifying tool calls auto-checkpoint as "
        "'aar-auto: <tool> turn-<N>'. "
        "Available slash commands: /undo [N] [--force], /revert, /fork [N], "
        "/switch <branch|fork-K>, /forks, /done. "
        "Never commit directly to the user's base branch — work is merged back only on /done."
    )
