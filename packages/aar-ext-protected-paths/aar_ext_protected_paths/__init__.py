"""Aar extension: protected paths — block writes to sensitive files."""
from __future__ import annotations

import fnmatch
from typing import Any

# Glob patterns for protected files (matched against the path argument)
PROTECTED_PATTERNS: list[str] = [
    "**/.env",
    "**/.env.*",
    "**/credentials",
    "**/credentials.*",
    "**/secrets",
    "**/secrets.*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/.ssh/*",
    "**/id_rsa",
    "**/id_dsa",
    "**/id_ecdsa",
    "**/id_ed25519",
    "**/.aws/*",
    "**/.azure/*",
    "**/.config/gcloud/*",
    "**/.netrc",
    "**/.npmrc",
    "**/.pypirc",
]

WRITE_TOOLS: set[str] = {"write_file", "edit_file"}


def _is_protected(path: str) -> str | None:
    """Return the matching pattern if path is protected, else None."""
    normalized = path.replace("\\", "/")
    # Try matching as-is and also with a dummy prefix so **/ patterns work on bare names
    candidates = [normalized]
    if "/" not in normalized:
        candidates.append("_/" + normalized)
    for candidate in candidates:
        for pattern in PROTECTED_PATTERNS:
            if fnmatch.fnmatch(candidate, pattern):
                return pattern
    return None


def register(api: Any) -> None:
    """Register the protected-paths extension."""

    @api.on("tool_call")
    def guard_writes(event: Any, ctx: Any) -> Any:
        if event.tool_name not in WRITE_TOOLS:
            return None
        path = event.arguments.get("path", "")
        if not path:
            return None
        matched = _is_protected(path)
        if matched:
            ctx.logger.warning(
                "protected-paths: blocked write to %s (matches %s)", path, matched
            )
            return api.block(f"Write blocked: '{path}' matches protected pattern '{matched}'")
        return None

    @api.on("session_start")
    def on_start(event: Any, ctx: Any) -> None:
        ctx.logger.info(
            "protected-paths extension loaded (%d patterns)", len(PROTECTED_PATTERNS)
        )

    api.append_system_prompt(
        "The protected-paths extension is active. "
        "Writes to sensitive files (.env, credentials, SSH keys, etc.) will be blocked."
    )
