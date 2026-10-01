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
| `/undo [N \| tN \| pN \| <sha>] [--force]` | Revert N logical checkpoints (default 1), or everything from the named checkpoint / prompt onwards (see **References** below), skipping `shadow-meta` commits and reloading session events to the restored timeline. Refuses to run with a dirty tree unless `--force` is passed. Returns `↩ reverted N checkpoint(s) (t3 and later) → <sha>`. |
| `/revert …` | Alias for `/undo`. |
| `/undo … --dry-run` | Preview only: `• /undo would drop 2 checkpoint(s) (p2 and later) · 3 files +40 −12 — nothing changed`. A real `/undo` reports the same `dropped …` stat. |
| `/redo [--force]` | Re-apply what the last `/undo` removed (repeatable for several undos). Only while nothing new happened since — a new checkpoint, `/branch`, `/switch`, `/restore` or `/done` clears it, and a new prompt makes it refuse (`✗ cannot redo — the conversation moved on since the /undo`). Reloads the session events like `/undo`. |
| `/restore <path> [tN \| pN \| <sha>] [--force]` | Put **one file** back to how it was just before the named checkpoint / prompt — by default before the last checkpoint that changed it. A file that did not exist then is removed. Recorded as its own checkpoint (`/restore <path>` group, tool `restore`), so `/undo` takes it back. Refuses a file with uncommitted changes unless `--force`. |
| `/branch [N \| tN \| pN \| <sha>]` | Preserve active shadow as `shadow/session-<id>-branch-<K>` and start a fresh branch from N logical checkpoints back — or from just before the named checkpoint / prompt, i.e. "retry p2 differently" — (or `HEAD` if nothing is given), skipping `shadow-meta` commits when counting. Reloads session events from the fork point's JSONL so the LLM context matches the new branch. Multiple branches are allowed. Returns `⑂ branch-K preserved as <branch> — now on fresh <branch>`. |
| `/switch [<target>]` | Switch to any shadow/branch copy for **this** session. Reloads the session's conversation history from the target branch's JSONL so the LLM context matches the files on disk. Rejects branches belonging to other sessions. See **Switch shorthands** below. Returns `⇄ switched to <branch> (base=<base>, N checkpoint(s), M events)`. |
| `/branches` | Tree of every shadow/branch copy for this session (canonical first, active marked `◀ active`) with its newest checkpoints — turn, tool, SHA, diffstat, age, `⚠` / `◀ tip` markers — the base anchor, and pending changes. Plain text, so it reads the same in every transport. |
| `/diff [tN \| pN \| <sha>] [--patch]` | What a checkpoint (default: the latest) or a whole prompt changed, as `git --stat`; `--patch` adds the unified diff (coloured in the TUIs, a `diff` code block in editors). Session files under `.agent/` are left out of prompt diffs. |
| `/done [message] [--yes] [--dry-run]` | Squash-merge the active shadow back into the base branch recorded in the `shadow-init` anchor. If preserved branches still exist it refuses unless `--yes` is passed. On conflicts it stops without committing, prints the paths, and leaves unresolved merge state on the base. Message parsing drops flags and integer-only tokens. `--dry-run` previews the squash (checkpoints, diffstat vs base, preserved branches) without sweeping, committing or checking out anything. Returns `✓ squashed <shadow> → <base> as <sha>`. |

Error and warning returns use `✗` and `⚠` prefixes respectively.

---

## References: `tN`, `pN`, `<sha>`

Checkpoints are grouped by the **prompt** that produced them. Commands accept
the names the tree shows:

| Name | Means | Stable? |
|---|---|---|
| `t4` | 4th checkpoint on the active line | Positions — renumbered after `/undo` / `/branch` |
| `p2` | all checkpoints made while answering your 2nd file-changing prompt | Never reused, not even after `/undo` |
| `e2ebcdc` | the checkpoint with that SHA (≥ 7 hex chars) | Always unambiguous |
| `3` | the last 3 checkpoints (unchanged behaviour) | — |

`/undo` and `/branch` rewind to just **before** the named checkpoint or prompt —
the named work is what gets dropped (`/branch` keeps it on the preserved line).
Unknown names are refused without touching the branch
(`✗ no prompt p7 on shadow — see /branches`).

Prompt numbers and excerpts live in the session state and as
`Shadow-Prompt: <n>` / `Shadow-Prompt-Text: <excerpt>` trailers on each
`shadow-auto` commit, so resumed sessions and preserved branches keep the
grouping. Loop-internal nudges are not counted as prompts; sessions recorded
before 0.5.0 keep the flat checkpoint list.

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

## TUI panel

Since 0.3.0 the extension registers a **UI panel** (Aar's `UIPanel` contract,
`agent.extensions.api`); 0.4.0 adds per-node details and a detail pane, 0.5.0 groups
checkpoints under the prompt that produced them.

### Fixed TUI (`aar tui --fixed`) — sidebar + zoom

The panel lives in a **sidebar left of the chat**, visible from the start, so
checkpoints can be followed while the agent works. Labels stay short enough for
the narrow column; the title carries the status (`⎇ Shadow · shadow · 4 cp ⊞`).

```
⎇ Shadow · shadow · 4 cp  ⊞
main @ 3f9c1e2
▼ shadow ● active
  ▼ p2 "rename login …" ●
    ├ t4 edit_file ●
    └ t3 write_file ⚠
  ▼ p1 "add a login fo…"
    ├ t2 edit_file
    └ t1 write_file
▶ branch-1 (3 cp)
✎ 1 untracked (pending)
[u] undo to here [b] fork here
[d] diff [r] refresh [z] zoom
```

Press `z` in the sidebar (or click its title) to **zoom** into a near-full-screen
window: every node gets its detail (`e71a9d0 · 2 files +12 −4 · 5m ago`, the full
branch name, …), a detail pane on the right shows `git show --stat` for a
checkpoint, the diff vs. base and recent checkpoints for a branch, the base's
last commit, or `git status --short` for pending changes — and each action is a
**clickable button** (keys work too). `esc`, `z`, `ctrl+b` or a click on the
title close it; the sidebar follows the selection you made in the window.

| Key | Node | Action |
|-----|------|--------|
| `u` | checkpoint | `/undo` back to that checkpoint (confirm; `f` in the dialog = `--force`). Selecting the tip returns a no-op message. |
| `u` | prompt | undo the whole prompt and everything after it (confirm) |
| `b` | checkpoint / active branch | `/branch N` from that checkpoint, or `/branch` from HEAD |
| `b` | prompt | retry from before the prompt: preserve the line, fresh branch without it |
| `s` | branch | `/switch` to that branch |
| `d` | checkpoint / prompt | `git show --stat` for the checkpoint, combined `--stat` for a prompt (read-only) |
| `x` | branch | `git branch -D` (confirm). Refuses the active shadow and the base branch. |
| `D` | root / base / branch | `/done --yes` with the message typed in the dialog (confirm) |
| `r` | any | re-read the tree |
| `z` | any | zoom the sidebar into the window / close the window |

* Newest checkpoint on top; `●` marks the tip, `⚠` a checkpoint that touched a
  sensitive-looking path, `(N cp)` a collapsed branch, `✎` uncommitted changes.
* Destructive actions show a preview in their confirm dialog — what an undo
  drops (`drops 3 checkpoint(s) · 4 files +60 −12 · /redo restores`), what a
  deleted branch holds, what `/done` would squash.
* Action keys only work while the panel has focus (`ctrl+b` focuses the
  sidebar, again returns to the input, `esc` hides it). Mutating actions are
  refused while the agent is running; `diff` / `refresh` stay available.
* Every action prints the same result line the slash command would, and the
  transcript is re-rendered after `undo` / `switch` / `branch`.
* After `/done` the panel shows *Shadow branching inactive*.
* The header shows a chip `⎇ shadow · N cp` (or `⎇ branch-K · N cp`) while armed.
* Hide the sidebar by default with
  `"tui": {"layout": {"extensions": {"shadow_branching": {"visible": false}}}}`.

### Checkpoint notes (every interface)

Each checkpoint is announced on the tool result that caused it — in the result
panel's bottom border in both TUIs, and as an extra line on the tool-call card in
editors over ACP (Zed):

```
╰──────── ⎇ checkpoint t4 (p3) · e2ebcdc · 1 file +2 −0 · /undo t4 ─╯
```

A checkpoint touching a sensitive-looking path adds `· ⚠ sensitive path`. The
`/undo t4` hint is a position: after an `/undo`, use the SHA from an older note
instead.

### Inline TUI (`aar tui`)

`/panel` prints the same tree as a Rich panel with details. After a slash command
that moved the shadow state (`/undo`, `/branch`, …) a compact version (5 newest
checkpoints per branch) is printed automatically — the layout setting above turns
that off. `/branches` gives the plain-text tree:

```
⎇ session a1b2 · base main @ 3f9c1e2
├─ ● shadow/session-a1b2  (4 cp)  ◀ active
│  ├─ p2 "rename login to sign_in"  2 cp · +2 −2
│  │  ├─ t4   edit_file   e2ebcdc  1 file +1 −1  3m ago  ◀ tip
│  │  └─ t3   write_file  31870a4  1 file +1 −1  4m ago  ⚠
│  └─ p1 "add a login form"  2 cp · +3 −0
│     ├─ t2   edit_file   7b5554a  1 file +1 −0  6m ago
│     └─ t1   write_file  f0da1c6  1 file +2 −0  9m ago
├─ ○ shadow/session-a1b2-branch-1  (3 cp)
│  └─ …
└─ ✎ 1 untracked (pending)
```

### Editors over ACP (Zed)

Commands appear in the editor's `/` menu with argument placeholders
(`/undo [N | tN | pN] [--force]`, `/diff [tN | pN] [--patch]`, `/switch [main | K | branch-K]`, …).
`/branches` and other tree replies arrive in a code block, so the layout survives
the editor's Markdown rendering.

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
