# aar-ext-inspect

Slash-commands extension for Aar that provides a `/inspect` command to analyse
the current session and emit a human-readable report.

This extension is intentionally lightweight and defensive: it reads the session
object and recent events where available and prints a best-effort summary via
the extension logger (so output appears in TUI consoles, log viewers, and ACP
clients that surface extension log output).

---

## Features

- Registers a `/inspect` slash-command (entry-point `inspect`).
- Produces a concise session report including:
  - session id, trace id, step count
  - counts of messages / events
  - counts of user / assistant messages
  - tool-call / tool-result breakdown
  - token usage aggregated from provider metadata (if present)
  - truncated last assistant message
- Supports a `verbose` mode: `/inspect verbose` prints last ~20 events with short summaries.
- Defensive: works with different transports/providers by falling back gracefully if fields are missing.

---

## Installation

From the repository root (development install):

```bash
pip install -e packages/aar-ext-inspect
```

From PyPI (when published):

```bash
pip install aar-ext-inspect
```

The package exposes the `aar_extensions` entry point named `inspect` — Aar will
auto-discover and load it from installed packages, the user extensions directory,
or per-project `.agent/extensions/` when present.

---

## Usage

Note: slash-command availability depends on the transport/client you run Aar
with. ACP-capable editor integrations and some clients will surface extension
slash-commands directly in their command palette or input UI. When available,
type the command into the input:

- Quick report (best-effort):
```
/inspect
```

- Verbose report (includes short summaries of last ~20 events):
```
/inspect verbose
```

If your transport does not directly dispatch extension slash-commands, you can
still use the extension in other ways:

- From the LLM: the LLM can call registered extension tools (if used).
- For debugging: `aar extensions inspect inspect` shows the extension metadata
  (what it registers) from the CLI.

Output is emitted via the extension logger (scoped to `aar.ext.<name>`), so it
appears in the TUI console output or in the log viewer in the fixed Textual TUI.
The report is intentionally textual to be readable in terminal environments.

---

## Example output

A simplified example the extension might log:

```
=== Session Inspect Report ===
Session ID: 7a12f3b2
Trace ID: abcde-12345
Step count: 6
Total events: 18
Total stored messages: 6
User messages: 3
Assistant messages: 3
Reasoning (stream) chunks seen: 4
Errors recorded: 0

Tool calls summary:
  • read_file: 2 call(s)
Tool results summary:
  • read_file: 2 result(s)

Token usage (aggregated from ProviderMeta events):
  • Input tokens: 120
  • Output tokens: 340

Last assistant message (truncated):
<assistant reply text...>
```

---

## Development notes

- Entry-point: `aar_ext_inspect:register` (declared in package `pyproject.toml`).
- The extension registers the `/inspect` command using `api.command(...)`. It
  reads `ctx.session` and `ctx.config` and writes results to `ctx.logger`.
- The extension is defensive: it tolerates missing attributes on session/events
  so it can run across different providers and transports.

Suggested improvements:
- Add a UI panel adapter if you want a richer visual representation in the
  fixed Textual TUI.
- Add unit tests that construct a synthetic Session with events and assert the
  produced log text contains expected summary lines.

---

## License

See LICENSE or the `pyproject.toml` for packaging metadata.

---
