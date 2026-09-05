# aar-ext-shadow-branching

Session-scoped git isolation for Aar.

Every Aar session gets its own throw-away `shadow/session-<id>` branch. Every
modifying tool call produces a numbered `shadow-auto:` checkpoint commit. The
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

* Auto-creates a shadow branch `shadow/session-<SESSION_ID>` rooted in an empty
  `shadow-init: base=<ORIGINAL_BRANCH>` anchor commit, so `/done` always knows
  where to merge back — no out-of-band state to lose.
* Auto-commits after every modifying tool call as
  `shadow-auto: <tool_name> turn-<N>`. Uses `git add -A` so side-effects from
  `bash` tool calls are captured, and warns (in the log) before staging files
  that look sensitive — the lower-cased path containing any of `.env`, `.key`,
  `credentials`, `id_rsa`, `secret`.
* **Auto-commits pending changes before branch operations.** Before `/branch`,
  `/switch`, and `/done`, any uncommitted files (e.g. the session `.jsonl`
  written by the transport after `agent.run()`) are swept into an
  `shadow-meta: pre-* sync` commit so branch operations are never blocked by a
  dirty working tree caused by session bookkeeping.
* **Post-turn sweep.** A `session_end` hook commits any files that are still
  dirty after the agent loop finishes (the session JSONL is saved by the
  transport after all extension events fire). These commits use
  `shadow-meta: session sync` and do not increment the turn counter or appear
  in the checkpoint list.
* **Session reload on `/switch`, `/branch`, and `/undo`.** When switching
  between branches, creating a new branch, or reverting checkpoints, the
  extension reloads the session's conversation history (events, step count,
  metadata) from the JSONL file on disk. After `/branch`, the new branch's
  HEAD points at the fork commit whose JSONL reflects only the conversation up
  to that point — without reloading, the in-memory session would still contain
  events about work that now lives exclusively on the preserved branch, causing
  the next `store.save()` to overwrite the fork-point JSONL with stale history.
  After `/switch`, the target branch's JSONL is loaded for the same reason.
  After `/undo`, reloaded events forget reverted work. If the target timeline
  has no JSONL (e.g. an early fork before any session save), events are cleared
  to avoid stale history. This keeps the LLM's context in sync with the files —
  true time-travel within a session.
* **Cross-session safety.** Starting a new session while the repo HEAD sits on
  another session's shadow branch automatically checks out the recorded base
  branch first. `/switch` rejects targets belonging to a different session.
* Branch-aware: `/branch` preserves the current branch as
  `shadow/session-<id>-branch-<K>` and starts a fresh shadow — branch numbering
  is derived from the branches on disk, so it survives session resumes and
  arbitrarily deep branch-of-branch chains.
* Safe `/undo`: refuses to touch a dirty working tree unless you pass
  `--force`, counts logical `shadow-auto` checkpoints rather than raw commits,
  and skips transparent `shadow-meta` housekeeping commits.
* Graceful `/done`: reads the base from the `shadow-init` anchor, aborts cleanly
  on merge conflicts, and tells you which files to resolve manually.
* All commands return a one-liner feedback string displayed directly in the
  TUI/CLI — not only in the log.
* Mirrors the full shadow-branching state into `session.metadata`, so
  resumed sessions pick up exactly where they left off.
* Falls back to `.shadow_backups/` when the project directory is not a git repo
  (checkpoints are disabled in this mode — the directory is created as a
  signal and hook point for future snapshot support).

---

## Slash commands

| Command | Description |
|---|---|
| `/undo [N] [--force]` | Revert N logical checkpoints (default 1), skipping `shadow-meta` commits and reloading session events to the restored timeline. Refuses to run with a dirty tree unless `--force` is passed. Returns `↩ reverted N checkpoint(s) → <sha>`. |
| `/revert [N] [--force]` | Alias for `/undo`. |
| `/branch [N]` | Preserve active shadow as `shadow/session-<id>-branch-<K>` and start a fresh branch from N logical checkpoints back (or `HEAD` if N is omitted), skipping `shadow-meta` commits when counting. Reloads session events from the fork point's JSONL so the LLM context matches the new branch. Multiple branches are allowed. Returns `⑂ branch-K preserved as <branch> — now on fresh <branch>`. |
| `/switch [<target>]` | Switch to any shadow/branch copy for **this** session. Reloads the session's conversation history from the target branch's JSONL so the LLM context matches the files on disk. Rejects branches belonging to other sessions. See **Switch shorthands** below. Returns `⇄ switched to <branch> (base=<base>, N checkpoint(s), M events)`. |
| `/branches` | List every shadow/branch copy for this session as an indented listing, with the active branch marked `◀ active`. The canonical shadow is shown as the root; preserved copies are listed below it. |
| `/done [message] [--yes]` | Squash-merge the active shadow back into the base branch recorded in the `shadow-init` anchor. If preserved branches still exist it refuses unless `--yes` is passed. Conflicts abort the merge and print the conflicting paths. Returns `✓ squashed <shadow> → <base> as <sha>`. |

Error and warning returns use `✗` and `⚠` prefixes respectively.

---

## Switch shorthands

`/switch` accepts several forms:

| Input | Resolves to |
|---|---|
| `/switch` *(no args)* | Shows current branch and all available targets — does not switch. |
| `/switch main` | The canonical shadow branch `shadow/session-<id>` (no branch suffix). |
| `/switch active` | Same as `main`. |
| `/switch shadow` | Same as `main`. |
| `/switch 3` | `shadow/session-<id>-branch-3` |
| `/switch branch-3` | `shadow/session-<id>-branch-3` |
| `/switch shadow/session-<id>-branch-3` | Exact branch name — verbatim. |

Use `main` / `active` / `shadow` to return to the canonical shadow branch
after visiting a preserved branch.

> **Note:** `/switch` only works within the current session's branches.
> Passing a branch name that belongs to a different session (e.g.
> `shadow/session-OTHER-branch-1`) returns an error. To resume another session's
> work, restart Aar with `--session <id>`.

---

## Commit taxonomy

The extension uses three commit message prefixes:

| Prefix | When | Counted as checkpoint? |
|---|---|---|
| `shadow-init: base=<branch>` | Once per session — the empty anchor commit that records the base branch. | No |
| `shadow-auto: <tool> turn-<N>` | After every modifying tool call. | **Yes** — appears in `/undo` counts. |
| `shadow-meta: <label>` | Pre-command sweeps and post-turn JSONL sync. | No |

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

## TUI panel (`aar tui --fixed`, ctrl+b)

Since 0.3.0 the extension registers a **UI panel** (Aar's `UIPanel` contract,
`agent.extensions.api`). In the fixed TUI press `ctrl+b` to open it in the
right column; it shows the session's git shadow tree and runs every operation
without typing slash commands:

```
⎇ Shadow
session a1b2
├─ main @ 3f9c1e2
├─ shadow/session-a1b2  ● active
│  ├─ e71a9d0  turn  7  edit_file ●
│  ├─ b02c4f1  turn  6  bash
│  ├─ 9a8d33c  turn  5  write_file ⚠
│  └─ …
├─ shadow/session-a1b2-branch-1  (4 cp)
└─ 2 modified · 1 untracked (pending)
───────────────────────────────────────
[u] undo to here  [b] fork here  [d] diff  [r] refresh  [D] squash → base
```

| Key | Node | Action |
|-----|------|--------|
| `u` | checkpoint | `/undo` back to that checkpoint (confirm; `f` in the dialog = `--force`). Disabled on the tip. |
| `b` | checkpoint / active branch | `/branch N` from that checkpoint, or `/branch` from HEAD |
| `s` | branch | `/switch` to that branch |
| `d` | checkpoint | `git show --stat` for the checkpoint (read-only) |
| `x` | branch | `git branch -D` (confirm). Refuses the active shadow and the base branch. |
| `D` | root / base / branch | `/done --yes` with the message typed in the dialog (confirm) |
| `r` | any | re-read the tree |

* Newest checkpoint on top; `●` marks the tip, `⚠` a checkpoint that touched a
  sensitive-looking path, `(N cp)` a collapsed branch.
* Action keys only work while the panel has focus (`ctrl+b` toggles focus,
  `esc` hides). Mutating actions are refused while the agent is running;
  `diff` / `refresh` stay available.
* Every action prints the same result line the slash command would, and the
  transcript is re-rendered after `undo` / `switch` / `branch`.
* After `/done` the panel shows *Shadow branching inactive*.
* The header shows a chip `⎇ session-<id> · N cp` while the panel is armed.

The same tree and actions are available to editors over ACP stdio
(`_aar/panel_list`, `_aar/panel_snapshot`, `_aar/panel_action`,
`_aar/panel_changed`) — see `docs/acp.md` in the Aar repo. On an Aar core that
predates the panel contract the extension still loads with slash commands only.

## Notes

* The extension operates on the working directory. Use Aar's default project
  sandbox or run from your repo root.
* **Session events are reloaded after `/branch`, `/switch`, and `/undo`.** The
  in-memory `Session.events` are replaced with the events from the on-disk
  JSONL so the LLM never sees conversation history that doesn't match the
  files. After `/branch`, the fork-point JSONL is loaded; after `/switch`, the
  target branch's JSONL is loaded; after `/undo`, the restored checkpoint
  boundary's JSONL is loaded. If no JSONL exists on the target timeline,
  events are cleared.
* **Starting a new session in a shadow-branching project:** if the repo HEAD
  is on `shadow/session-<OLD>` when a new session starts, the extension switches
  to the old branch's recorded base (typically `main`) first. A warning is
  logged with a hint to use `--session` to resume instead. This prevents the
  new session from being rooted on the old session's work-in-progress.
* **After `/done` the extension is inactive for the rest of the session.** The
  squash-merge leaves HEAD on your base branch, so shadow-branching disarms
  itself (`mode: done`) — no further checkpoints, sweeps or session-save commits
  are made, and `/undo`, `/branch`, `/switch` and `/done` report
  `session already merged via /done`. Start a new session to get a fresh shadow
  branch. Resuming the merged session with `--session <id>` also stays disarmed.
* `/done` does not delete the shadow or preserved branches — cleanup is left to
  the user (`git branch -D shadow/session-<id>*`) so nothing is lost silently.
* The session JSONL under `.agent/sessions/` lives inside the work tree, so it
  is checkpointed along with your changes and squashed into the base branch by
  `/done`. Add `.agent/` to `.gitignore` if you don't want session transcripts
  in your history.
* If `git user.name` / `user.email` are not configured, checkpoints are
  disabled and a one-time warning is logged.
* `shadow-meta:` commits are intentionally excluded from the checkpoint list and
  do not affect `/undo` counts. They exist solely to keep the working tree
  clean for branch operations.

---

## License

See LICENSE or the `pyproject.toml` for packaging metadata.

---
