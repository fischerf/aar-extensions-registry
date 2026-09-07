# aar-ext-shadow-branching

Session-scoped git isolation for Aar.

In a Git repository with a configured commit identity, each Aar session gets a
dedicated `shadow/session-<id>` branch. A tool result that leaves stageable
changes produces a numbered `shadow-auto:` checkpoint commit. The user's base
branch stays untouched until a `/done` squash-merge at the end of the session.

Built on top of Aar's extension API — `register(api)` is the only integration
point, and every command is registered through the Slash-commands extension
surface. Commands are available in CLI chat, the inline and fixed TUIs, and ACP
stdio clients such as Zed. ACP HTTP does not currently parse extension slash
commands.

The harness-neutral [Shadow Branching Protocol Specification](docs/git_shadow_branching.md)
defines the behavior required for implementations in other coding-agent
harnesses. This extension uses one working tree and does not support concurrent
sessions in the same checkout; the specification describes worktrees as an
optional deployment architecture.

**Cross-session safety:** if the working directory is left on a shadow branch
from a *previous* session (e.g. the user ran `aar chat` without `--session`),
`session_start` attempts to check out the recorded base branch before creating
a new shadow branch, preventing stale-state inheritance when the checkout
succeeds. If it fails, the extension warns and continues from the current
`HEAD`. `/switch` is scoped to branches of the *current* session only;
attempting to switch to another session's branch is rejected with a descriptive
message.

---

## Features

* Auto-creates `shadow/session-<SESSION_ID>` from the current branch and adds an
  empty `shadow-init: base=<ORIGINAL_BRANCH>` anchor as the first
  session-specific commit, so `/done` can recover the merge target from Git
  history.
* After any tool result that leaves changes visible to normal Git status,
  auto-commits all stageable changes as `shadow-auto: <tool_name> turn-<N>`.
  It uses `git add -A` so side-effects from shell commands are captured, but
  this also means a checkpoint may include unrelated pending edits. It warns in the log when
  paths contain sensitive-looking names such as `.env`, `.key`, `credentials`,
  `id_rsa`, or `secret`; the warning does not block the commit.
* **Commits session persistence immediately.** Aar writes the session JSONL
  after the agent-loop lifecycle events have fired, so the extension installs
  an idempotent post-save hook around `SessionStore.save()`. A matching active
  session is committed as `shadow-meta: session-saved` as soon as the save
  finishes.
* **Sweeps pending changes defensively.** `before_turn` and `session_end` hooks
  create `shadow-meta: turn sync` or `shadow-meta: session sync` commits when
  needed. Before `/branch`, `/switch`, and `/done`, remaining changes are swept
  into `shadow-meta: pre-* sync`; the operation stops if the tree is still
  dirty. Meta commits do not increment the turn counter or appear in logical
  checkpoint counts.
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
  to avoid stale history. A reload error is logged but does not abort the
  command, so the live context can remain stale on that failure path.
* **Cross-session safety.** Starting a new session while the repo HEAD sits on
  another session's shadow branch attempts to check out the recorded base
  first. A failed checkout is warned about but does not stop initialization.
  `/switch` rejects targets belonging to a different session.
* Branch-aware: `/branch` preserves the current branch with a `-branch-<K>`
  suffix and recreates its previous name at the selected fork point. From the
  canonical shadow this produces `shadow/session-<id>-branch-<K>`; branching
  while a preserved branch is active can produce nested names such as
  `...-branch-1-branch-2`. Numbering is derived from refs on disk, so it
  survives session resumes and branch-of-branch chains.
* Safe `/undo`: refuses to touch a dirty working tree unless you pass
  `--force`, counts logical `shadow-auto` checkpoints rather than raw commits,
  and skips transparent `shadow-meta` housekeeping commits.
* Graceful `/done`: reads the base from the `shadow-init` anchor and stops
  without committing when a squash merge conflicts. It reports the conflicting
  paths and leaves the base branch's unresolved merge state for manual repair.
* Commands return feedback displayed directly in the TUI/CLI, not only in the
  log. `/switch` without arguments and `/branches` return multiline listings.
* Mirrors the full shadow-branching state into `session.metadata` as a durable
  audit record. Process restarts reconstruct the canonical session timeline
  from Git history; the metadata is not currently used to restore a previously
  active preserved branch.
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
| `/done [message] [--yes]` | Squash-merge the active shadow back into the base branch recorded in the `shadow-init` anchor. If preserved branches still exist it refuses unless `--yes` is passed. On conflicts it stops without committing, prints the paths, and leaves unresolved merge state on the base. Message parsing drops flags and integer-only tokens. Returns `✓ squashed <shadow> → <base> as <sha>`. |

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
| `shadow-auto: <tool> turn-<N>` | After a tool result when staged changes exist. | **Yes** — appears in `/undo` counts. |
| `shadow-meta: <label>` | Session saves and housekeeping sweeps, including `session-saved`, `turn sync`, `session sync`, and `pre-* sync`. | No |

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
| `u` | checkpoint | `/undo` back to that checkpoint (confirm; `f` in the dialog = `--force`). Selecting the tip returns a no-op message. |
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
`_aar/panel_changed`) and through the ACP HTTP panel endpoints and
`panel_changed` SSE events — see `docs/acp.md` in the Aar repo. ACP HTTP still
does not parse slash commands directly. On an Aar core that predates the panel
contract the extension loads with slash commands only.

## Notes

* The extension operates on the working directory. Use Aar's default project
  sandbox or run from your repo root.
* **Session events are reloaded after `/branch`, `/switch`, and `/undo`.** The
  extension attempts to replace in-memory `Session.events` with the events from
  the on-disk JSONL. After `/branch`, the fork-point JSONL is loaded; after
  `/switch`, the target branch's JSONL is loaded; after `/undo`, the restored
  checkpoint boundary's JSONL is loaded. If no JSONL exists on the target
  timeline, events are cleared. Other reload failures are logged but do not
  abort the command.
* **Starting a new session in a shadow-branching project:** if the repo HEAD
  is on `shadow/session-<OLD>` when a new session starts, the extension switches
  to the old branch's recorded base (typically `main`) first. A warning is
  logged with a hint to use `--session` to resume instead. If checkout fails,
  initialization continues from the stale `HEAD` after another warning.
* **After `/done` the extension is inactive for the rest of the session.** The
  squash-merge leaves `HEAD` on the base branch, so shadow branching disarms
  itself (`mode: done`). No further checkpoints, sweeps, or session-save commits
  are made, and `/undo`, `/branch`, `/switch`, and `/done` report that the
  session was already merged. Starting a new session creates a fresh shadow;
  resuming the merged session with `--session <id>` keeps it disarmed.
* `/done` does not delete the shadow or preserved branches — cleanup is left to
  the user (`git branch -D shadow/session-<id>*`) so nothing is lost silently.
* If `/done` stops on a merge conflict or final-commit failure, `HEAD` is already
  on the base branch but the extension remains enabled. Resolve the state before
  another turn or session save; an automatic save can otherwise create a
  `shadow-meta:` commit on the base. Successful `/done` does disarm safely.
* The session JSONL under `.agent/sessions/` lives inside the work tree, so it
  is checkpointed along with other changes and squashed into the base branch by
  `/done`. Add `.agent/` to `.gitignore` if you do not want session transcripts
  in project history.
* If `git user.name` / `user.email` are not configured, checkpoints are
  disabled and a warning is logged on each repeated initialization attempt.
* `shadow-meta:` commits are intentionally excluded from the checkpoint list and
  do not affect `/undo` counts. They keep session context versioned and the
  working tree clean for branch operations.
* Untracked ignored paths are not staged by `git add -A`, do not produce a
  checkpoint by themselves, and cannot be restored by `/undo`. Changes and
  deletions to already tracked paths are still staged even if a later ignore
  rule matches them.
* This implementation operates in one working tree. Starting concurrent agent
  sessions in the same checkout is unsupported; use separate clones or a
  worktree-aware host integration.

---

## License

See LICENSE or the `pyproject.toml` for packaging metadata.

---
