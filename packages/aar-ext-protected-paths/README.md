# aar-ext-protected-paths

Aar extension that blocks writes to sensitive files and credentials.

## Install

```bash
pip install aar-ext-protected-paths
# or
aar install aar-ext-protected-paths
```

## What it does

Hooks into `tool_call` events for `write_file` and `edit_file`, blocking writes to:
- `.env` and `.env.*` files
- `credentials`, `secrets` files
- Private keys (`.pem`, `.key`, `.p12`, `.pfx`)
- SSH keys (`id_rsa`, `id_ed25519`, etc.)
- Cloud credentials (`.aws/`, `.azure/`, `.config/gcloud/`)
- Package manager tokens (`.netrc`, `.npmrc`, `.pypirc`)

## Customization

Fork and edit `PROTECTED_PATTERNS` in `__init__.py` to add/remove patterns.

---

## License

See LICENSE or the `pyproject.toml` for packaging metadata.

---
