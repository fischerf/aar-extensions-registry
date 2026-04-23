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

---

## Features

* Auto-creates a shadow branch `aar/session-<SESSION_ID>` rooted in an empty
  `aar-init: base=<ORIGINAL_BRANCH>` anchor commit, so `/done` always knows
  where to merge back — no out-of-band state to lose.
* Auto-commits after every modifying tool call as
  `aar-auto: <tool_name> turn-<N>`. Uses `git add -A` so side-effects from
  `bash` tool calls are captured, and warns (in the log) before staging files
  that look sensitive (`.env*`, `*.key`, `*credentials*`, `id_rsa*`).
* **Auto-commits pending changes before branch operations.** Before `/fork`,
  `/switch`, and `/done`, any uncommitted files (e.g. the session `.jsonl`
  written by the transport after `agent.run()`) are swept into an
  `aar-meta: pre-* sync` commit so branch operations are never blocked by a
  dirty working tree caused by session bookkeeping.
* **Post-turn sweep.** A `session_end` hook commits any files that are still
  dirty after the agent loop finishes (the session JSONL is saved by the
  transport after all extension events fire). These commits use
  `aar-meta: session sync` and do not increment the turn counter or appear
  in the checkpoint list.
* Fork-aware: `/fork` preserves the current branch as
  `aar/session-<id>-fork-<K>` and starts a fresh shadow — fork numbering
  is derived from the branches on disk, so it survives session resumes and
  arbitrarily deep fork-of-fork chains.
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
| `/fork [N]` | Preserve active shadow as `aar/session-<id>-fork-<K>` and start a fresh branch from `HEAD~N` (or `HEAD` if N is omitted). Multiple forks are allowed. Returns `⑂ fork-K preserved as <branch> — now on fresh <branch>`. |
| `/switch [<target>]` | Switch to any shadow/fork branch for this session. See **Switch shorthands** below. Returns `⇄ switched to <branch> (base=<base>, N checkpoint(s))`. |
| `/forks` | List every shadow/fork branch for this session, with the active branch marked `◀ active`. |
| `/done [message] [--yes]` | Squash-merge the active shadow back into the base branch recorded in the `aar-init` anchor. If fork branches still exist it refuses unless `--yes` is passed. Conflicts abort the merge and print the conflicting paths. Returns `✓ squashed <shadow> → <base> as <sha>`. |

Error and warning returns use `✗` and `⚠` prefixes respectively.

---

## Switch shorthands

`/switch` accepts several forms:

| Input | Resolves to |
|---|---|
| `/switch` *(no args)* | Shows current branch and all available targets — does not switch. |
| `/switch main` | The canonical shadow branch `aar/session-<id>` (no fork suffix). |
| `/switch active` | Same as `main`. |
| `/switch shadow` | Same as `main`. |
| `/switch 3` | `aar/session-<id>-fork-3` |
| `/switch fork-3` | `aar/session-<id>-fork-3` |
| `/switch aar/session-<id>-fork-3` | Exact branch name — verbatim. |

Use `main` / `active` / `shadow` to return to the canonical shadow branch
after visiting a preserved fork.

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
* `/done` does not delete the shadow or fork branches — cleanup is left to
  the user (`git branch -D aar/session-<id>*`) so nothing is lost silently.
* If `git user.name` / `user.email` are not configured, checkpoints are
  disabled and a one-time warning is logged.
* `aar-meta:` commits are intentionally excluded from the checkpoint list and
  do not affect `/undo` counts. They exist solely to keep the working tree
  clean for branch operations.

---

## License

MIT.