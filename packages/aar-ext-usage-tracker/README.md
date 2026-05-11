# aar-ext-usage-tracker

Aar extension that tracks per-provider API quota and usage, persisted to `~/.aar/usage.json`.

## Install

```bash
pip install aar-ext-usage-tracker
```

Or place the package in `~/.aar/extensions/` for auto-discovery.

## What it does

- **Tracks monthly requests** per provider (incremented on each API call)
- **Tracks token usage** (input / output tokens per month)
- **Warns** when approaching the monthly request limit (configurable threshold)
- **Logs errors** when the monthly limit is reached
- **Persists** all data to `~/.aar/usage.json` (survives restarts)
- **`/usage` command** — show usage stats in any TUI/CLI session
- **`usage_status` tool** — the LLM can check its own remaining quota

## Configuration

Add `quota` to the provider's `extra` config in `~/.aar/config.json`:

```json
{
  "providers": {
    "claude": {
      "name": "anthropic",
      "model": "claude-sonnet-4-6",
      "extra": {
        "quota": {
          "monthly_requests": 1000,
          "warn_at_percent": 80.0
        }
      }
    }
  }
}
```

| Field | Default | Description |
|---|---|---|
| `monthly_requests` | `0` (unlimited) | Max API requests per calendar month (UTC) |
| `warn_at_percent` | `80.0` | Log a warning when usage reaches this % |

## Usage data format (`~/.aar/usage.json`)

```json
{
  "version": 1,
  "providers": {
    "claude": {
      "quota": {
        "monthly_requests": 1000,
        "warn_at_percent": 80.0
      },
      "months": {
        "2025-07": {
          "requests": 42,
          "input_tokens": 150000,
          "output_tokens": 50000,
          "first_request_at": "2025-07-17T09:00:00+00:00",
          "last_request_at": "2025-07-17T14:30:00+00:00"
        }
      }
    }
  }
}
```

Months reset automatically — new calendar months start a fresh counter.

## Example: Gateway quota mapping

| Gateway Quota | Extension Config | Notes |
|---|---|---|
| Monthly requests: 1000/user | `monthly_requests: 1000` | ✅ Fully tracked |
| Max input tokens: 1,000,000 | `context_window` in provider config | Not tracked here (per-request limit) |
| Max output tokens: 128,000 | `max_tokens` in provider config | Not tracked here (per-request limit) |
| QPM (shared) | — | Shared across all users; not enforceable per-client |
| Input TPM (shared) | — | Shared; tracked but not enforced |
| Output TPM (shared) | — | Shared; tracked but not enforced |

## `/usage` command output

```
── claude (2025-07) ──
  requests:  42 / 1000 (4.2%) — 958 remaining
  input:     150,000 tokens
  output:    50,000 tokens
  total:     200,000 tokens
  first:     2025-07-17T09:00:00+00:00
  last:      2025-07-17T14:30:00+00:00
```

## License

Apache License 2.0 — same as Aar.
