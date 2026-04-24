# aar-ext-git-checkpoint

Aar extension that creates git checkpoints after each agent turn for safe rollback.

## Install

```bash
pip install aar-ext-git-checkpoint
# or
aar install aar-ext-git-checkpoint
```

## What it does

- **After each turn**: auto-commits all changes with message `aar: checkpoint after turn N`
- **Rollback tool**: provides a `git_rollback` tool the agent can use to undo changes
- Automatically detects if the working directory is a git repo
- Uses `--no-verify` to skip pre-commit hooks for speed
- Skips checkpoints when there are no changes

## Notes

Checkpoints are regular git commits. You can squash them later with `git rebase -i`.

---

## License

See LICENSE or the `pyproject.toml` for packaging metadata.

---
