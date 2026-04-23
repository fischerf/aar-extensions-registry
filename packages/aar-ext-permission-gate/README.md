# aar-ext-permission-gate

Aar extension that blocks dangerous bash commands before execution.

## Install

```bash
pip install aar-ext-permission-gate
# or
aar install aar-ext-permission-gate
```

## What it does

Hooks into `tool_call` events and blocks bash commands containing dangerous patterns:
- `rm -rf /`, `rm -rf ~`, `sudo rm`
- `mkfs`, `dd if=`, `> /dev/sda`
- `chmod 777`, `chmod -R 777`
- `shutdown`, `reboot`, `halt`, `poweroff`
- Fork bombs

## Customization

Fork and edit `DANGEROUS_PATTERNS` in `__init__.py` to add/remove patterns.
