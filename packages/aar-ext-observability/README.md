# aar-ext-observability

Aar extension that emits structured metrics and logging for each agent turn.

## Install

```bash
pip install aar-ext-observability
```

## What it does

- **After each turn**: logs structured JSON with token counts, cost, duration
- **On session end**: logs a summary with totals
- **On errors**: logs structured error records
- **Event bus**: emits metrics:turn, metrics:session, metrics:error events
- **Tool**: provides session_stats tool for the agent to query metrics

## Output format

See the extension source code for the full JSON metrics schema.
