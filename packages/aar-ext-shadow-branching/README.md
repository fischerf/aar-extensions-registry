# aar-ext-shadow-branching

Session-scoped git isolation for Aar.

Every Aar session gets its own throw-away `aar/session-<id>` branch. Every
modifying tool call produces a numbered `aar-auto:` checkpoint commit. The
user's base branch stays untouched until a `/done` squash-merge at the end of
the session.

Built on top of Aar's extension API — `register(api)` is the only integration
point, and every command is registered through the Slash-commands extension
surface so it is available in every transport (CLI chat, inline TUI, fixed
TUI, and ACP-surfaced clients such as the Zed editor).

**Cross-session safety:** if the working directory is left on a shadow branch
from a *previous* session (e.g. the user ran `aar chat` without `--session`),
`session_start` automatically checks out the recorded base branch before
creating a new shadow branch — preventing the new session from inheriting
stale file state. `/switch` is scoped to branches of the *current* session
only; attempting to switch to another session's branch is rejected with a
descriptive message.

---

## Features

* Auto-creates a shadow branch `aar/session-<SESSION_ID>` rooted in an empty
  `aar-init: base=<ORIGINAL_BRANCH>` anchor commit, so `/done` always knows
  where to merge back — no out-of-band state to lose.
* Auto-commits after every modifying tool call as
  `aar-auto: <tool_name> turn-<N>`. Uses `git add -A` so side-effects from
  `bash` tool calls are captured, and warns (in the log) before staging files
  that look sensitive (`.env*`, `*.key`, `*credentials*`, `id_rsa*`).
* **Auto-commits pending changes before branch operations.** Before `/branch`,
  `/switch`, and `/done`, any uncommitted files (e.g. the session `.jsonl`
  written by the transport after `agent.run()`) are swept into an
  `aar-meta: pre-* sync` commit so branch operations are never blocked by a
  dirty working tree caused by session bookkeeping.
* **Post-turn sweep.** A `session_end` hook commits any files that are still
  dirty after the agent loop finishes (the session JSONL is saved by the
  transport after all extension events fire). These commits use
  `aar-meta: session sync` and do not increment the turn counter or appear
  in the checkpoint list.
* **Session reload on `/switch` and `/branch`.** When switching between branches
  or creating a new branch, the extension reloads the session's conversation
  history (events, step count, metadata) from the JSONL file on disk. After
  `/branch`, the new branch's HEAD points at the fork commit whose JSONL
  reflects only the conversation up to that point — without reloading, the
  in-memory session would still contain events about work that now lives
  exclusively on the preserved branch, causing the next `store.save()` to
  overwrite the fork-point JSONL with stale history. After `/switch`, the
  target branch's JSONL is loaded for the same reason. If the target branch
  has no JSONL (e.g. an early fork before any session save), events are
  cleared to avoid stale history. This keeps the LLM's context in sync with
  the files — true time-travel within a session.
* **Cross-session safety.** Starting a new session while the repo HEAD sits on
  another session's shadow branch automatically checks out the recorded base
  branch first. `/switch` rejects targets belonging to a different session.
* Branch-aware: `/branch` preserves the current branch as
  `aar/session-<id>-branch-<K>` and starts a fresh shadow — branch numbering
  is derived from the branches on disk, so it survives session resumes and
  arbitrarily deep branch-of-branch chains.
* Safe `/undo`: refuses to touch a dirty working tree unless you pass
  `--force`.
* Graceful `/done`: reads the base from the `aar-init` anchor, aborts cleanly
  on merge conflicts, and tells you which files to resolve manually.
* All commands return a one-liner feedback string displayed directly in the
  TUI/CLI — not only in the log.
* Mirrors the full shadow-branching state into `session.metadata`, so
  resumed sessions pick up exactly where they left off.
* Falls back to `.aar_backups/` when the project directory is not a git repo
  (checkpoints are disabled in this mode — the directory is created as a
  signal and hook point for future snapshot support).

---

## Slash commands

| Command | Description |
|---|---|
| `/undo [N] [--force]` | Revert N checkpoints (default 1). Refuses to run with a dirty tree unless `--force` is passed. Returns `↩ reverted N checkpoint(s) → <sha>`. |
| `/revert [N] [--force]` | Alias for `/undo`. |
| `/branch [N]` | Preserve active shadow as `aar/session-<id>-branch-<K>` and start a fresh branch from `HEAD~N` (or `HEAD` if N is omitted). Reloads session events from the fork point's JSONL so the LLM context matches the new branch. Multiple branches are allowed. Returns `⑂ branch-K preserved as <branch> — now on fresh <branch>`. |
| `/switch [<target>]` | Switch to any shadow/branch copy for **this** session. Reloads the session's conversation history from the target branch's JSONL so the LLM context matches the files on disk. Rejects branches belonging to other sessions. See **Switch shorthands** below. Returns `⇄ switched to <branch> (base=<base>, N checkpoint(s), M events)`. |
| `/branches` | List every shadow/branch copy for this session as a tree, with the active branch marked `◀ active`. The canonical shadow is shown as the root; preserved copies are indented beneath it. |
| `/done [message] [--yes]` | Squash-merge the active shadow back into the base branch recorded in the `aar-init` anchor. If preserved branches still exist it refuses unless `--yes` is passed. Conflicts abort the merge and print the conflicting paths. Returns `✓ squashed <shadow> → <base> as <sha>`. |

Error and warning returns use `✗` and `⚠` prefixes respectively.

---

## Switch shorthands

`/switch` accepts several forms:

| Input | Resolves to |
|---|---|
| `/switch` *(no args)* | Shows current branch and all available targets — does not switch. |
| `/switch main` | The canonical shadow branch `aar/session-<id>` (no branch suffix). |
| `/switch active` | Same as `main`. |
| `/switch shadow` | Same as `main`. |
| `/switch 3` | `aar/session-<id>-branch-3` |
| `/switch branch-3` | `aar/session-<id>-branch-3` |
| `/switch aar/session-<id>-branch-3` | Exact branch name — verbatim. |

Use `main` / `active` / `shadow` to return to the canonical shadow branch
after visiting a preserved branch.

> **Note:** `/switch` only works within the current session's branches.
> Passing a branch name that belongs to a different session (e.g.
> `aar/session-OTHER-branch-1`) returns an error. To resume another session's
> work, restart Aar with `--session <id>`.

---

## Commit taxonomy

The extension uses three commit message prefixes:

| Prefix | When | Counted as checkpoint? |
|---|---|---|
| `aar-init: base=<branch>` | Once per session — the empty anchor commit that records the base branch. | No |
| `aar-auto: <tool> turn-<N>` | After every modifying tool call. | **Yes** — appears in `/undo` counts. |
| `aar-meta: <label>` | Pre-command sweeps and post-turn JSONL sync. | No |

---

## Installation

Development (editable):

```bash
pip install -e aar-extensions-registry/packages/aar-ext-shadow-branching
```

Published:

```bash
pip install aar-ext-shadow-branching
```

Aar auto-discovers installed extensions via the `aar_extensions` entry-point
group — no configuration changes needed.

---

## Notes

* The extension operates on the working directory. Use Aar's default project
  sandbox or run from your repo root.
* **Session events are reloaded after `/branch` and `/switch`.** The in-memory
  `Session.events` are replaced with the events from the on-disk JSONL so the
  LLM never sees conversation history that doesn't match the files. After
  `/branch`, the fork-point JSONL is loaded; after `/switch`, the target
  branch's JSONL is loaded. If no JSONL exists on the target branch, events
  are cleared.
* **Starting a new session in a shadow-branching project:** if the repo HEAD
  is on `aar/session-<OLD>` when a new session starts, the extension switches
  to the old branch's recorded base (typically `main`) first. A warning is
  logged with a hint to use `--session` to resume instead. This prevents the
  new session from being rooted on the old session's work-in-progress.
* `/done` does not delete the shadow or preserved branches — cleanup is left to
  the user (`git branch -D aar/session-<id>*`) so nothing is lost silently.
* If `git user.name` / `user.email` are not configured, checkpoints are
  disabled and a one-time warning is logged.
* `aar-meta:` commits are intentionally excluded from the checkpoint list and
  do not affect `/undo` counts. They exist solely to keep the working tree
  clean for branch operations.

---

## License

See LICENSE or the `pyproject.toml` for packaging metadata.

---
