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
* Fork-aware: `/fork` preserves the current branch as
  `aar/session-<id>-fork-<K>` and starts a fresh shadow — fork numbering
  is derived from the branches on disk, so it survives session resumes and
  arbitrarily deep fork-of-fork chains.
* Safe `/undo`: refuses to touch a dirty working tree unless you pass
  `--force`.
* Graceful `/done`: reads the base from the `aar-init` anchor, aborts cleanly
  on merge conflicts, and tells you which files to resolve manually.
* Mirrors the full shadow-branching state into `session.metadata`, so
  resumed sessions pick up exactly where they left off.
* Falls back to `.aar_backups/` when the project directory is not a git repo
  (checkpoints are disabled in this mode — the directory is created as a
  signal and hook point for future snapshot support).

---

## Slash commands

| Command | Description |
|---|---|
| `/undo [N] [--force]` | Revert N checkpoints (default 1). Refuses to run with a dirty tree unless `--force` is passed. |
| `/revert N` | Alias for `/undo`. |
| `/fork [N]` | Preserve active shadow as `aar/session-<id>-fork-<K>` and start a fresh branch from `HEAD~N` (or `HEAD` if N is omitted). Multiple forks are allowed. |
| `/switch <branch \| fork-K \| K>` | Switch to any shadow/fork branch belonging to this session. Shorthand accepted: bare number or `fork-K` expands to the matching fork branch. Refuses to switch with a dirty tree. |
| `/forks` | List every shadow/fork branch for this session. |
| `/done [message] [--yes]` | Squash-merge the active shadow back into the base branch recorded in the `aar-init` anchor. If fork branches still exist it refuses unless `--yes` is passed. Conflicts abort the merge and print the conflicting paths. |

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

---

## License

MIT.
