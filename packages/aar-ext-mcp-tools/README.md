# aar-ext-mcp-tools

Aar extension that discovers and registers MCP (Model Context Protocol) server tools.

## Install

```bash
pip install aar-ext-mcp-tools
# or
aar install aar-ext-mcp-tools
```

## What it does

On session start, reads `~/.aar/mcp_servers.json` and registers all discovered
MCP tools so the agent can use them natively.

## Configuration

Create `~/.aar/mcp_servers.json`:

```json
{
  "servers": [
    {
      "name": "filesystem",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  ]
}
```

## Requirements

Requires the `mcp` package: `pip install "aar-agent[mcp]"`

## Note

This is a reference implementation. Aar's built-in `agent.extensions.mcp` module
provides the same functionality without the extension API. Use this extension when
you want MCP tools managed by the extension lifecycle.

---

## License

See LICENSE or the `pyproject.toml` for packaging metadata.

---
