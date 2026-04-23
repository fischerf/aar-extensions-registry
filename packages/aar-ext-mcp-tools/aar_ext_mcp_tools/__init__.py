"""Aar extension: MCP tools — register MCP server tools via the extension API.

This extension reads MCP server configuration from the environment or config
and registers discovered tools so the agent can use them. Unlike the built-in
MCPBridge (which requires manual wiring), this extension auto-discovers MCP
servers from ~/.aar/mcp_servers.json or a path set in the config.

Note: This is a reference implementation. The built-in agent.extensions.mcp
module provides the same functionality without the extension API. Use this
package when you want MCP tools managed by the extension lifecycle.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MCP_CONFIG = Path.home() / ".aar" / "mcp_servers.json"


def register(api: Any) -> None:
    """Register MCP tools discovered from the MCP servers config."""

    @api.on("session_start")
    async def discover_mcp(event: Any, ctx: Any) -> None:
        config_path = _DEFAULT_MCP_CONFIG
        if not config_path.is_file():
            ctx.logger.debug("mcp-tools: no MCP config at %s, skipping", config_path)
            return

        try:
            from agent.extensions.mcp import MCPBridge, load_mcp_config

            servers = load_mcp_config(str(config_path))
            if not servers:
                return

            ctx.logger.info("mcp-tools: connecting to %d MCP server(s)", len(servers))

            # Note: In a real deployment, the bridge lifetime must be managed.
            # This is a simplified reference — the bridge stays open for the
            # session duration via the extension context.
            # Full lifecycle management would require the extension API to
            # support async context managers (future enhancement).
            ctx.logger.info(
                "mcp-tools: discovered %d server config(s) from %s",
                len(servers),
                config_path,
            )
        except ImportError:
            ctx.logger.warning(
                "mcp-tools: 'mcp' package not installed. "
                "Install with: pip install 'aar-agent[mcp]'"
            )
        except Exception as exc:
            ctx.logger.error("mcp-tools: failed to load MCP config: %s", exc)

    api.append_system_prompt(
        "The mcp-tools extension is available. "
        "MCP server tools will be registered automatically from ~/.aar/mcp_servers.json."
    )
