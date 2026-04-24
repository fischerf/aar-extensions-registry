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
    /branch [N]           Preserve the active shadow branch as
                          ``aar/session-<id>-branch-<K>`` and start a fresh shadow
                          from ``HEAD~N`` (or ``HEAD`` if no N given). Branch
                          numbering is derived from the branches that already
                          exist on disk, so it survives session reloads and
                          branch-of-branch chains.
    /switch <branch>      Switch to any existing ``aar/session-<id>*`` branch.
                          Accepts bare branch names or the shorthand
                          ``branch-<K>``. Refuses to switch with a dirty tree.
    /branches             List every shadow/branch copy belonging to this
                          session and mark the active one, as a tree.

All state is also mirrored into ``session.metadata['shadow_branching']`` so the
session JSONL store keeps a faithful record across resumes.

A ``before_turn`` hook commits any JSONL or other files written by the transport
after the previous turn completed, keeping the working tree clean so that
``/branch``, ``/switch``, and ``/done`` are never blocked by a dirty tree caused
purely by session bookkeeping.
    /done                 Squash-merge the active shadow branch back into the
                          base branch captured in ``aar-init``. Aborts cleanly
                          on merge conflicts and names the files that need
                          manual resolution.

All state is also mirrored into ``session.metadata['shadow_branching']`` so the
session JSONL store keeps a faithful record across resumes.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["register", "ShadowState"]

_module_logger = logging.getLogger("aar_ext_shadow_branching")


# ---------------------------------------------------------------------------
# git helpers — thin wrappers so tests can drive a real temp repo
# ---------------------------------------------------------------------------


# Default per-call timeout. `git add -A` on a working tree that includes a
# Python ``venv/`` or a JS ``node_modules/`` can easily take much longer than
# the 15 s originally allowed here — when that timeout tripped silently the
# extension happily created a /branch that did not actually contain the user's
# work. 120 s covers realistic large-tree cases while still bounding hangs.
_GIT_TIMEOUT = 120


def _run_git(
    *args: str, cwd: str | Path | None = None, timeout: int | None = None
) -> tuple[int, str, str]:
    """Run a git command; return (returncode, stdout, stderr) — stripped."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else _GIT_TIMEOUT,
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
    rc, out, _ = _run_git("branch", "--list", pattern, "--format=%(refname:short)", cwd=cwd)
    if rc != 0 or not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _next_branch_n(session_id: str, cwd: str | Path | None = None) -> int:
    """Largest existing branch number for this session + 1. Derived from disk
    state so numbering survives session reloads and branch-of-branch chains."""
    branches = _list_branches(f"aar/session-{session_id}-branch-*", cwd=cwd)
    max_n = 0
    for b in branches:
        suffix = b.rsplit("-branch-", 1)[-1]
        try:
            n = int(suffix)
            max_n = max(max_n, n)
        except ValueError:
            continue
    return max_n + 1


def _read_anchor_base(branch: str, fallback: str, cwd: str | Path | None = None) -> str:
    """Return the base branch recorded in the ``aar-init`` anchor commit of
    *branch* (falls back to *fallback* when no anchor is present)."""
    rc, out, _ = _run_git("log", "--grep=^aar-init:", "--pretty=%s", branch, cwd=cwd)
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
    branch_counter: int = 0
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
# SessionStore.save hook — commit JSONL immediately after it is written
# ---------------------------------------------------------------------------
#
# ``agent.core.loop`` fires ``session_end`` BEFORE the transport calls
# ``store.save(session)``, so our ``session_end`` handler runs at a moment when
# the JSONL file has not yet been written — the sweep finds nothing to commit
# and the tree is left dirty the instant ``save`` returns.  The next
# ``before_turn`` or slash-command does catch up, but if the user inspects the
# repo (or a tool outside the extension touches it) between turns the tree
# looks dirty for no good reason.
#
# To make "save" feel atomic from git's point of view we wrap
# ``SessionStore.save`` once per process with a post-save hook that runs
# ``_commit_pending`` after the original ``save`` completes. The hook only
# fires when a ShadowState is registered for the matching ``session_id`` and
# is ``enabled`` — so other processes / sessions are unaffected.


# session_id -> live ShadowState, populated by register()/session_start.
_active_states: dict[str, "ShadowState"] = {}
# Guard so repeated register() calls (tests, reload) don't stack wrappers.
_save_hook_installed = False


def _commit_pending(label: str, logger: logging.Logger | None = None) -> bool:
    """Module-level twin of the inner ``_auto_commit_pending`` closure — used
    by the ``SessionStore.save`` hook, which has no event-loop ``ctx`` to pass.

    Returns True iff an ``aar-meta: <label>`` commit was created.
    """
    log = logger or _module_logger
    if not _has_changes():
        return False
    rc_add, out_add, err_add = _run_git("add", "-A")
    if rc_add != 0:
        detail = (err_add or out_add or "(no output)").strip()
        log.warning("shadow-branching: git add -A failed during %s: %s", label, detail)
        return False
    rc_diff, _, _ = _run_git("diff", "--cached", "--quiet")
    if rc_diff == 0:
        return False
    rc, out, err = _run_git("commit", "-m", f"aar-meta: {label}", "--no-verify")
    if rc != 0:
        detail = (out or err or "(no output)").strip()
        log.warning("shadow-branching: auto-commit of pending changes failed: %s", detail)
        return False
    log.debug(
        "shadow-branching: auto-committed pending changes (%s) → %s",
        label,
        _short_hash("HEAD"),
    )
    return True


def _install_save_hook() -> None:
    """Monkey-patch ``SessionStore.save`` so that every save is immediately
    followed by an ``aar-meta: session-saved`` commit when shadow-branching is
    active for the session being saved.

    Idempotent: only installs once per process.  Safe to call at every
    ``register()`` because the sentinel attribute ``_aar_shadow_wrapped`` on the
    method object pins the installed state even across re-imports of this
    module.
    """
    global _save_hook_installed
    if _save_hook_installed:
        return
    try:
        from agent.memory.session_store import SessionStore
    except Exception as exc:  # pragma: no cover — aar must be importable
        _module_logger.debug("shadow-branching: cannot import SessionStore: %s", exc)
        return

    if getattr(SessionStore.save, "_aar_shadow_wrapped", False):
        _save_hook_installed = True
        return

    original_save = SessionStore.save

    def save_with_shadow_commit(self: Any, session: Any) -> Any:
        path = original_save(self, session)
        state = _active_states.get(getattr(session, "session_id", ""))
        if state is not None and state.enabled:
            _commit_pending("session-saved", logger=_module_logger)
        return path

    save_with_shadow_commit._aar_shadow_wrapped = True  # type: ignore[attr-defined]
    SessionStore.save = save_with_shadow_commit  # type: ignore[assignment]
    _save_hook_installed = True


# ---------------------------------------------------------------------------
# Extension entry-point
# ---------------------------------------------------------------------------


def register(api: Any) -> None:
    """Register the shadow-branching extension against *api*."""

    # Wrap SessionStore.save so that every save triggers an immediate commit.
    # Idempotent: safe to call once per register() invocation.
    _install_save_hook()

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
        # Keep the module-level registry in sync so the SessionStore.save hook
        # can find the right state to sweep when save(session_id=...) fires.
        if state.session_id:
            _active_states[state.session_id] = state

    def _reconstruct_from_branch(branch: str) -> tuple[int, list[dict[str, Any]]]:
        """Count aar-auto commits on *branch* and build a checkpoint list."""
        rc, out, _ = _run_git("log", "--reverse", "--format=%h %s", branch)
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

    def _auto_commit_pending(ctx: Any, label: str = "session sync") -> bool:
        """Commit any uncommitted changes with an aar-meta message.

        This sweeps up session JSONL writes (and any other minor file changes that
        haven't been captured by a checkpoint yet) so that commands like /branch,
        /switch, and /done don't get blocked by a dirty working tree caused purely
        by the session store being updated outside of git.

        *ctx* may be ``None`` (e.g. when called from the SessionStore.save hook,
        which runs outside any event dispatch); in that case the module logger
        is used.

        Returns True if a commit was made, False if the tree was already clean or
        the commit failed.  On failure a warning is logged so callers can decide
        whether to abort — silent ``False`` only means "nothing needed committing".
        """
        logger = getattr(ctx, "logger", None) or _module_logger
        if not _has_changes():
            return False
        rc_add, out_add, err_add = _run_git("add", "-A")
        if rc_add != 0:
            # git add -A can fail for a number of reasons — most importantly here,
            # it times out on large working trees (venv/, node_modules/).  Surface
            # it loudly so the caller (/branch, /switch, /done) can refuse to
            # silently proceed on a tree that still holds un-staged work.
            detail = (err_add or out_add or "(no output)").strip()
            logger.warning(
                "shadow-branching: git add -A failed during %s: %s", label, detail
            )
            return False
        # After staging, check whether anything actually made it into the index.
        # git add -A can succeed yet stage nothing (e.g. all changes were to
        # files outside the work-tree or the content was identical to HEAD).
        rc_diff, _, _ = _run_git("diff", "--cached", "--quiet")
        if rc_diff == 0:
            # Index is clean — nothing to commit; silently skip.
            return False
        rc, out, err = _run_git(
            "commit",
            "-m",
            f"aar-meta: {label}",
            "--no-verify",
        )
        if rc != 0:
            detail = (out or err or "(no output)").strip()
            logger.warning(
                "shadow-branching: auto-commit of pending changes failed: %s", detail
            )
            return False
        logger.debug(
            "shadow-branching: auto-committed pending changes (%s) → %s",
            label,
            _short_hash("HEAD"),
        )
        return True

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
            state = ShadowState(session_id=session_id, enabled=False, mode="fallback")
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
                ctx.logger.warning("shadow-branching: cannot resume %s: %s", shadow_name, err)
                state = ShadowState(session_id=session_id, enabled=False)
                _sync_metadata(ctx)
                return

            base = _read_anchor_base(shadow_name, current or "main")
            turn, ckpts = _reconstruct_from_branch(shadow_name)
            branch_counter = _next_branch_n(session_id) - 1  # already-used counter
            state = ShadowState(
                session_id=session_id,
                original_branch=base,
                shadow_branch=shadow_name,
                turn_counter=turn,
                checkpoints=ckpts,
                branch_counter=max(0, branch_counter),
                base_anchor=_short_hash(shadow_name + "^{/^aar-init:}") or _short_hash("HEAD"),
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
            ctx.logger.warning("shadow-branching: failed to create aar-init anchor: %s", err)

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
    # session_end — sweep up any dirty files left after agent.run() + store.save()
    # ------------------------------------------------------------------

    @api.on("before_turn")
    def on_before_turn(event: Any, ctx: Any) -> None:
        """Sweep any pending changes before each new agent turn starts.

        The transport writes the session JSONL after every completed turn (outside
        the agent loop).  By the time the *next* turn's ``before_turn`` fires the
        file is already on disk, so we commit it here with an ``aar-meta: turn sync``
        message.  This keeps the working tree clean throughout the session so that
        ``/branch``, ``/switch``, and ``/done`` are never blocked by a dirty tree
        caused purely by session bookkeeping."""
        if state is None or not state.enabled:
            return
        _auto_commit_pending(ctx, "turn sync")

    @api.on("session_end")
    def on_session_end(event: Any, ctx: Any) -> None:
        """Commit any files that were written after the last tool_result checkpoint.

        The session JSONL is saved by the transport *after* the agent loop ends
        (and therefore after every tool_result event). This means each completed
        turn leaves the JSONL dirty in git. We commit it here with an
        ``aar-meta: session sync`` message so the working tree stays clean and
        commands like /branch and /switch never block on pending changes."""
        if state is None or not state.enabled:
            return
        _auto_commit_pending(ctx, "session sync")

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
        rc_add, out_add, err_add = _run_git("add", "-A")
        if rc_add != 0:
            # Most commonly a timeout on a huge working tree (venv/,
            # node_modules/). Log loudly — a silently-skipped checkpoint means
            # the user's next /branch or /undo will operate on stale state.
            detail = (err_add or out_add or "(no output)").strip()
            ctx.logger.warning(
                "shadow-branching: git add -A failed for tool_result %s: %s — checkpoint skipped",
                tool_name,
                detail,
            )
            state.turn_counter = prev_turn
            return
        # Guard against "nothing to commit" after staging — can happen when the
        # tool wrote a file whose content is identical to the tracked version.
        rc_diff, _, _ = _run_git("diff", "--cached", "--quiet")
        if rc_diff == 0:
            state.turn_counter = prev_turn
            return
        rc, out, err = _run_git(
            "commit",
            "-m",
            f"aar-auto: {tool_name} turn-{state.turn_counter}",
            "--no-verify",
        )
        if rc != 0:
            detail = (out or err or "(no output)").strip()
            ctx.logger.warning("shadow-branching: checkpoint commit failed: %s", detail)
            state.turn_counter = prev_turn
            return

        sha = _short_hash("HEAD")
        state.checkpoints.append({"turn": state.turn_counter, "hash": sha, "tool": tool_name})
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

    def _do_undo(args: str, ctx: Any) -> str | None:
        if state is None or not state.enabled:
            ctx.logger.warning("shadow-branching: disabled (no git repo or identity)")
            return "✗ shadow-branching disabled (no git repo or identity)"

        n, flags = _parse_int_arg(args)
        n = n if n and n >= 1 else 1
        force = bool(flags & {"--force", "-f"})

        if n > len(state.checkpoints):
            ctx.logger.warning(
                "shadow-branching: cannot undo %d — only %d checkpoint(s) available",
                n,
                len(state.checkpoints),
            )
            return f"✗ only {len(state.checkpoints)} checkpoint(s) available, cannot undo {n}"

        dirty = _has_changes()
        if dirty and not force:
            ctx.logger.warning(
                "shadow-branching: uncommitted changes present — "
                "commit/stash them, or re-run with --force to discard."
            )
            return "✗ uncommitted changes present — commit/stash or use --force"

        rc, _, err = _run_git("reset", "--hard", f"HEAD~{n}")
        if rc != 0:
            ctx.logger.warning("shadow-branching: git reset failed: %s", err)
            return f"✗ git reset failed: {err}"

        # ``git reset --hard`` only touches tracked changes; untracked files
        # linger. When --force was requested the user accepted losing
        # uncommitted work, so sweep those too (ignored files are preserved).
        if dirty and force:
            _run_git("clean", "-fd")

        removed = state.checkpoints[-n:]
        state.checkpoints = state.checkpoints[:-n]
        state.turn_counter = max(0, state.turn_counter - n)
        sha = _short_hash("HEAD")
        ctx.logger.info(
            "shadow-branching: reverted %d checkpoint(s), now at %s (removed turns %s)",
            n,
            sha,
            [c["turn"] for c in removed],
        )
        _sync_metadata(ctx)
        return f"↩ reverted {n} checkpoint(s) → {sha}"

    @api.command(
        "undo", description="Revert N checkpoints (default 1). Use --force with dirty tree."
    )
    def cmd_undo(args: str, ctx: Any) -> str | None:
        return _do_undo(args, ctx)

    @api.command("revert", description="Alias for /undo")
    def cmd_revert(args: str, ctx: Any) -> str | None:
        return _do_undo(args, ctx)

    # ------------------------------------------------------------------
    # /branch
    # ------------------------------------------------------------------

    @api.command(
        "branch",
        description="Preserve current shadow as aar/session-<id>-branch-<K> and start a fresh branch (optionally from N back).",
    )
    def cmd_branch(args: str, ctx: Any) -> str | None:
        nonlocal state
        if state is None or not state.enabled:
            ctx.logger.warning("shadow-branching: disabled (no git repo or identity)")
            return "✗ shadow-branching disabled (no git repo or identity)"

        # Sweep up any session-store writes (JSONL etc.) that happened outside
        # of tool_result checkpoints so the working tree is clean before we
        # rename branches and create a new one.
        _auto_commit_pending(ctx, "pre-branch sync")

        # Refuse to branch on a still-dirty tree: if the sweep couldn't commit
        # the pending work (e.g. ``git add -A`` timed out on a huge venv/),
        # preserving the current shadow would freeze a branch that silently
        # omits the user's actual work.  Better to stop and let the user
        # resolve than to hand them an empty-looking "preserved" branch.
        if _has_changes():
            rc, status_out, _ = _run_git("status", "--porcelain")
            preview = ""
            if rc == 0 and status_out:
                files = [ln[3:] for ln in status_out.splitlines() if len(ln) > 3][:5]
                more = "" if len(status_out.splitlines()) <= 5 else " …"
                preview = f" ({', '.join(files)}{more})"
            ctx.logger.warning(
                "shadow-branching: refusing to /branch — working tree is still dirty%s",
                preview,
            )
            return (
                "✗ working tree still dirty after pre-branch sync"
                f"{preview} — commit manually or add noisy paths (venv/, "
                "node_modules/, __pycache__/, .web_mcp_cache/) to .gitignore, "
                "then retry /branch"
            )

        n, _ = _parse_int_arg(args)
        if n is not None and n > len(state.checkpoints):
            ctx.logger.warning(
                "shadow-branching: only %d checkpoints, cannot branch %d back",
                len(state.checkpoints),
                n,
            )
            return (
                f"✗ only {len(state.checkpoints)} checkpoint(s) available, cannot branch {n} back"
            )

        # Resolve branch point before rename so HEAD~N is still valid.
        branch_ref = f"HEAD~{n}" if n else "HEAD"
        rc, branch_sha, err = _run_git("rev-parse", branch_ref)
        if rc != 0:
            ctx.logger.warning(
                "shadow-branching: cannot resolve branch point %s: %s", branch_ref, err
            )
            return f"✗ cannot resolve branch point {branch_ref}: {err}"

        branch_n = _next_branch_n(state.session_id)
        preserved = f"{state.shadow_branch}-branch-{branch_n}"

        rc, _, err = _run_git("branch", "-m", state.shadow_branch, preserved)
        if rc != 0:
            ctx.logger.warning("shadow-branching: branch rename failed: %s", err)
            return f"✗ branch rename failed: {err}"

        rc, _, err = _run_git("checkout", "-b", state.shadow_branch, branch_sha)
        if rc != 0:
            # Roll back the rename.
            _run_git("branch", "-m", preserved, state.shadow_branch)
            ctx.logger.warning(
                "shadow-branching: cannot start new branch at %s: %s",
                branch_sha[:8],
                err,
            )
            return f"✗ cannot start new branch at {branch_sha[:8]}: {err}"

        state.branch_counter = branch_n
        if n:
            state.checkpoints = state.checkpoints[:-n]
            state.turn_counter = max(0, state.turn_counter - n)

        ctx.logger.info(
            "[BRANCH preserved=%s active=%s branched-from=%s]",
            preserved,
            state.shadow_branch,
            branch_sha[:8],
        )
        _sync_metadata(ctx)
        suffix = f" (rewound {n} checkpoint(s))" if n else ""
        return f"⑂ branch-{branch_n} preserved as {preserved}{suffix} — now on fresh {state.shadow_branch}"

    # ------------------------------------------------------------------
    # /switch
    # ------------------------------------------------------------------

    @api.command(
        "switch",
        description=(
            "Switch to another aar/session-<id>* branch. "
            "Shorthands: branch-<K> or bare <K> for a branch, "
            "'main'/'active'/'shadow' for the canonical shadow branch. "
            "No args: show current branch and available targets."
        ),
    )
    def cmd_switch(args: str, ctx: Any) -> str | None:
        nonlocal state
        if state is None or not state.enabled:
            ctx.logger.warning("shadow-branching: disabled (no git repo or identity)")
            return "✗ shadow-branching disabled (no git repo or identity)"

        target = args.strip()

        # No args → show current branch + available targets as a hint.
        if not target:
            branches = _list_branches(f"aar/session-{state.session_id}*")
            canonical = f"aar/session-{state.session_id}"
            lines = [f"Current branch: {state.shadow_branch}"]
            lines.append("Available targets:")
            for b in branches:
                marker = " ◀ active" if b == state.shadow_branch else ""
                lines.append(f"  {b}{marker}")
            lines.append("Usage: /switch <branch-K | K | main | full-branch-name>")
            return "\n".join(lines)

        # Sweep up session-store writes before checking for a clean tree.
        _auto_commit_pending(ctx, "pre-switch sync")

        # Canonical shadow branch shorthands
        canonical = f"aar/session-{state.session_id}"
        if target in {"main", "active", "shadow"}:
            target = canonical

        # Numeric / branch-K / other shorthands → aar/session-<id>-branch-K
        elif target.isdigit():
            target = f"{canonical}-branch-{target}"
        elif target.startswith("branch-"):
            target = f"{canonical}-{target}"
        elif not target.startswith("aar/session-"):
            target = f"{canonical}-{target}"

        if not _branch_exists(target):
            ctx.logger.warning("shadow-branching: branch %r does not exist", target)
            return f"✗ branch {target!r} does not exist"

        if _has_changes():
            ctx.logger.warning(
                "shadow-branching: uncommitted changes — commit or stash before /switch"
            )
            return "✗ uncommitted changes present — commit/stash or let auto-commit run first"

        rc, _, err = _run_git("checkout", target)
        if rc != 0:
            ctx.logger.warning("shadow-branching: checkout failed: %s", err)
            return f"✗ checkout failed: {err}"

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
        return f"⇄ switched to {target} (base={base}, {turn} checkpoint(s))"

    # ------------------------------------------------------------------
    # /branches
    # ------------------------------------------------------------------

    @api.command(
        "branches", description="List all shadow/branch copies for this session as a tree."
    )
    def cmd_branches(args: str, ctx: Any) -> str | None:
        if state is None:
            ctx.logger.info("shadow-branching: not initialised")
            return "✗ shadow-branching not initialised"

        branches = _list_branches(f"aar/session-{state.session_id}*")
        if not branches:
            ctx.logger.info(
                "shadow-branching: no shadow branches found for session %s",
                state.session_id,
            )
            return f"no branches found for session {state.session_id}"

        canonical = f"aar/session-{state.session_id}"
        fork_prefix = f"{canonical}-branch-"
        root_marker = " ◀ active" if state.shadow_branch == canonical else ""
        lines: list[str] = [f"  {canonical}{root_marker}"]
        ctx.logger.info("  * %s%s", canonical, " (active)" if root_marker else "")

        children = sorted(b for b in branches if b.startswith(fork_prefix))
        for b in children:
            marker = " ◀ active" if b == state.shadow_branch else ""
            lines.append(f"      {b}{marker}")
            ctx.logger.info("  * %s%s", b, " (active)" if marker else "")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # /done
    # ------------------------------------------------------------------

    @api.command(
        "done",
        description="Squash-merge the active shadow branch into its recorded base; aborts on conflicts.",
    )
    def cmd_done(args: str, ctx: Any) -> str | None:
        if state is None or not state.enabled:
            ctx.logger.warning("shadow-branching: disabled (no git repo or identity)")
            return "✗ shadow-branching disabled (no git repo or identity)"

        # Sweep up session-store writes before checking for a clean tree.
        _auto_commit_pending(ctx, "pre-done sync")

        if _has_changes():
            ctx.logger.warning(
                "shadow-branching: uncommitted changes present — commit or stash before /done"
            )
            return "✗ uncommitted changes present — commit/stash or let auto-commit run first"

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

        forks = _list_branches(f"aar/session-{state.session_id}-branch-*")
        if forks and "--yes" not in flags:
            ctx.logger.info(
                "shadow-branching: preserved branches still exist (%s). "
                "Pass --yes to squash only the active shadow and leave them intact.",
                ", ".join(forks),
            )
            return f"⚠ preserved branches still exist: {', '.join(forks)} — pass --yes to proceed"

        base = _read_anchor_base(state.shadow_branch, state.original_branch or "main")

        rc, _, err = _run_git("checkout", base)
        if rc != 0:
            ctx.logger.warning("shadow-branching: cannot checkout base %s: %s", base, err)
            return f"✗ cannot checkout base branch {base!r}: {err}"

        rc, _, err = _run_git("merge", "--squash", state.shadow_branch)

        # Check for unresolved conflicts regardless of rc, then fall through.
        rc_u, conflicts, _ = _run_git("diff", "--name-only", "--diff-filter=U")
        if conflicts.strip():
            conflict_list = conflicts.strip().replace("\n", ", ")
            ctx.logger.warning(
                "shadow-branching: merge conflicts in: %s — resolve them and commit manually.",
                conflict_list,
            )
            return f"✗ merge conflicts in: {conflict_list} — resolve manually then commit"

        if rc != 0:
            ctx.logger.warning(
                "shadow-branching: git merge --squash failed: %s", err or "(no output)"
            )
            return f"✗ git merge --squash failed: {err or '(no output)'}"

        if not message:
            message = f"aar: squashed session {state.session_id}"

        rc, _, err = _run_git("commit", "-m", message, "--no-verify")
        if rc != 0:
            # Most likely: nothing staged (empty squash).
            ctx.logger.warning(
                "shadow-branching: final commit failed: %s (nothing to commit?)",
                err,
            )
            return f"✗ final commit failed: {err} (nothing to commit?)"

        sha = _short_hash("HEAD")
        ctx.logger.info(
            "shadow-branching: merged %s into %s as %s",
            state.shadow_branch,
            base,
            sha,
        )
        _sync_metadata(ctx)
        return f"✓ squashed {state.shadow_branch} → {base} as {sha}"

    # ------------------------------------------------------------------
    # System prompt additions
    # ------------------------------------------------------------------

    api.append_system_prompt(
        "The shadow-branching extension is active. Every session runs on an isolated "
        "aar/session-<id> git branch; modifying tool calls auto-checkpoint as "
        "'aar-auto: <tool> turn-<N>'. "
        "Available slash commands: /undo [N] [--force], /revert, /branch [N], "
        "/switch <branch|branch-K>, /branches, /done. "
        "Never commit directly to the user's base branch — work is merged back only on /done."
    )
