"""Aar extension: permission gate — block dangerous bash commands."""
from __future__ import annotations

from typing import Any

# Dangerous command patterns (substring match)
DANGEROUS_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "sudo rm",
    "mkfs",
    "dd if=",
    "> /dev/sda",
    "chmod 777",
    "chmod -R 777",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
]


def register(api: Any) -> None:
    """Register the permission-gate extension."""

    @api.on("tool_call")
    def guard_dangerous(event: Any, ctx: Any) -> Any:
        if event.tool_name != "bash":
            return None
        command = event.arguments.get("command", "")
        for pattern in DANGEROUS_PATTERNS:
            if pattern in command:
                ctx.logger.warning(
                    "permission-gate: blocked dangerous command: %s",
                    command[:100],
                )
                return api.block(f"Dangerous command blocked: contains '{pattern}'")
        return None

    @api.on("session_start")
    def on_start(event: Any, ctx: Any) -> None:
        ctx.logger.info("permission-gate extension loaded (%d patterns)", len(DANGEROUS_PATTERNS))

    api.append_system_prompt(
        "The permission-gate extension is active. "
        "Dangerous shell commands (rm -rf, sudo rm, mkfs, etc.) will be blocked automatically."
    )
