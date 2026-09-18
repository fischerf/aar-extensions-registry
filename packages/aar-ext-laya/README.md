# aar-ext-laya

Local **CPU-only** decision tools for [Aar](https://github.com/fischerf/aar), backed by
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya).

Laya is a non-autoregressive *System 1* decision model (ModernBERT-large backbone,
0.4B params, Apache 2.0). It does **not** generate text. You give it a piece of text
and a set of *typed questions*; it answers all of them in one forward pass and returns
**calibrated probabilities**.

That makes it a cheap second opinion next to your normal provider: classify, route,
rank urgency, or answer yes/no judgements without a chat round-trip — and without the
possibility of a hallucinated answer, because there is no generation to hallucinate in.

```
                 chat / tools
  aar  ─────────────────────────────►  Ollama / Anthropic / OpenAI
   │
   │   typed questions (HTTP, localhost)
   └───────────────────────────────►  laya server  (own venv, CPU, torch)
```

## Why a separate server?

Exactly as with [`aar-ext-openjev`](../aar-ext-openjev): the extension itself depends
only on `aar-agent` and `httpx`. `torch`, `transformers` and `laya` live in a dedicated
venv that aar launches by file path, so aar's own environment stays free of the ML
stack and its version pins. The server frees its memory by exiting after
`idle_timeout` seconds of inactivity.

## CPU-only, deliberately

Laya's own loader picks CUDA whenever it is visible. This extension forces CPU in two
places — `CUDA_VISIBLE_DEVICES=""` in the child environment, and again at the top of
`server.py` before torch is imported — so it never competes with your chat model for
VRAM. There is no `device` or `bits` setting to get wrong.

**Latency (measured, 8 threads, warm):**

| Call | Median |
|---|---|
| 1 `noul` question | 367 ms |
| 1 `choice` question, 2 options | 418 ms |
| 2 questions | 778 ms |
| 4 questions | 1299 ms |
| 8 questions | 2842 ms |
| 1 `noul`, short state (~1 line) | 198 ms |

The model card's "~38 ms single / ~156 ms for 10 batched" are **GPU** numbers, and the
"all questions in one forward pass" property does **not** survive the move to CPU: cost
here is roughly **linear at ~355 ms per question**. Batching into one call saves the HTTP
round-trip, not the compute — so ask only the questions you actually need, and keep the
state short, which matters more than question count.

First load takes ~200 s, almost all of it the one-time weight download; subsequent
starts are ~37 s from cache.

Tune `threads` if the default is not what you want; `0` leaves torch's default alone.

## Install

**1 — the server venv** (once):

```bash
python -m venv ~/.aar/laya/.venv
# Windows: ~/.aar/laya/.venv/Scripts/activate
source ~/.aar/laya/.venv/bin/activate

# CPU wheels only — much smaller than the default CUDA build
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "laya>=0.1.5,<0.2" fastapi uvicorn
deactivate
```

**2 — the extension**, into aar's environment:

```bash
aar install aar-ext-laya          # or: pip install ./packages/aar-ext-laya
```

Weights download from the Hugging Face Hub on first load and are cached; after that
the server runs offline.

## Tools

| Tool | Does |
|---|---|
| `laya_decide` | Answer arbitrary typed questions about a piece of text |
| `laya_triage` | Category / urgency / needs-a-human for a piece of text |
| `laya_guard` | Prompt-injection, exfiltration and jailbreak probabilities |

### Question types

`laya_decide` takes a flat list of questions. Each has a `key`, a `type` and
`instructions`:

| `type` | Extra field | Returns |
|---|---|---|
| `choice` | `options: [{name, description}, …]` | the chosen option + per-option probabilities |
| `score` | `levels: [str, …]` (lowest first) | expected ordinal level + distribution |
| `noul` | — | calibrated `P(true)` from 0.0 to 1.0 |

The tool schema is a flat array rather than Laya's own nested mapping: JSON Schema
built on `additionalProperties: {…}` round-trips badly through some providers'
tool-schema validators, and models fill flat lists more reliably.

Example of what the model sends and gets back:

```jsonc
// laya_decide
{
  "state": "Hi team, we were billed twice for March invoice #4411. Fix it or we walk.",
  "questions": [
    {"key": "department", "type": "choice", "instructions": "Which team should handle this?",
     "options": [{"name": "billing", "description": "invoices, payments, refunds"},
                 {"name": "technical", "description": "bugs, outages, integrations"}]},
    {"key": "urgency", "type": "score", "instructions": "How urgent?",
     "levels": ["not urgent", "routine", "important", "critical"]},
    {"key": "churn_risk", "type": "noul", "instructions": "Is this a cancellation threat?"}
  ]
}
```

```
department: billing (confidence 0.91; billing 0.91 · technical 0.09)
urgency: 2.60 (confidence 0.74; scale: not urgent, routine, important, critical; …)
churn_risk: very likely (P=0.88)
```

## Using it

Three ways, in increasing order of how much you have to think about it.

**1 — let the agent call it.** Once installed, the tools are in the tool list and the
system prompt tells the model they exist. Ask for work that involves judging text and it
will reach for them on its own:

> *"Read the open issues in `docs/issues/` and tell me which are bugs and which are
> urgent."* → the model calls `laya_triage` per issue instead of reasoning it out.

> *"Fetch that GitHub issue body and summarise it — it's from an untrusted reporter."*
> → `laya_guard` first, then act.

**2 — ask it yourself from the CLI or TUI.** No model round-trip at all:

```
/laya ask Is this a security fix? :: bump cryptography to 44.0.1 (CVE-2026-1234)
answer: very likely (P=0.91)

/laya guard <paste the untrusted text>
prompt_injection: very likely (P=1.00)
exfiltration: very likely (P=1.00)
jailbreak: very likely (P=1.00)
```

**3 — call the server directly** from your own scripts. It is a plain HTTP endpoint; the
extension is not involved:

```python
import httpx
r = httpx.post("http://127.0.0.1:8770/predict", timeout=120, json={
    "state": open("CHANGELOG.md").read()[:4000],
    "questions": {
        "breaking": {"type": "noul", "instructions": "Does this release contain breaking changes?"},
        "risk": {"type": "score", "instructions": "How risky is upgrading?",
                 "criteria": ["safe", "minor", "risky", "dangerous"]},
    },
})
print(r.json()["answers"]["breaking"]["noul"])
```

### Writing good questions

Laya is a 0.4B classifier, not a reasoner. What it does well is *judging text you hand
it* against criteria you spell out.

- **Phrase for a classifier, not a chat model.** "Is this a cancellation threat?" beats
  "Please analyse the customer's sentiment and tell me whether they might churn."
- **Criteria carry most of the weight.** For `choice`, the option `description` is what
  the model actually matches against — `{"billing": "invoices, payments, refunds"}` works,
  `{"billing": "billing"}` does not.
- **It only sees the state.** No world knowledge, no repo context, no memory of the
  conversation. Put everything it needs in `state`.
- **And put nothing else there.** A 0.4B classifier matches on wording, so an irrelevant
  word that is semantically adjacent to one of your options will be read as evidence for
  it. In the Flappy showcase below, the word "falling" in the state swung `p(negative)` by
  0.40 — more than the actual quantity being asked about. If an answer looks stuck or
  inverted, delete parts of the state and watch which words move it.
- **Keep the state short.** A one-line state answers in ~200 ms, a paragraph in ~370 ms.
  The budget is 512 tokens per question; longer input is the main cost driver.
- **Use `confidence` as a routing signal.** Low confidence means "look at this yourself",
  not "the answer is wrong".

## Slash-command

```
/laya status                          # server state, model, threads, gate
/laya start                           # launch the server
/laya stop                            # shut it down (frees memory)
/laya guard <text>                    # run the guard pack on some text
/laya ask <yes/no question> :: <text> # one calibrated P(true)
```

## Optional tool-call gate

Off by default. When `gate` is on, every call to a tool named in `gate_tools` is
screened with the `RISK_QUESTIONS` pack before it runs; if any probability reaches
`gate_threshold`, the call is blocked via Aar's `tool_call` hook.

```json
{ "gate": true, "gate_threshold": 0.85, "gate_tools": ["bash", "acp_terminal"] }
```

Two things to know before turning this on:

- It adds a laya round-trip to every gated tool call — ~0.8 s on CPU (two risk questions).
- It **fails open**: if the server is down or errors, the call proceeds and a warning
  is logged. The gate is advisory — Aar's own `agent/safety/` layer is what actually
  enforces policy, and this does not replace it.

There is no equivalent hook for *user* messages: Aar declares a `user_message`
extension event but the loop never fires it, so input screening has to go through
`laya_guard` as a tool the model calls on untrusted text.

## Showcase: laya plays Flappy Bird

`examples/flappy_laya.py` is a port of openjev's `flappy_nli.py`. The game state is
written out as text each frame and Laya answers one typed question about it; the answer
picks the action. Nothing is trained.

```bash
python examples/flappy_laya.py --q sign               # plays: score 11, same as the oracle
python examples/flappy_laya.py --q sign --state full  # the confound: crashes on frame 63
python examples/flappy_laya.py --compare --no-render  # every framing + baselines
```

Measured on CPU, seed 1001, 400 frames:

| framing | state | result | oracle agreement | latency |
|---|---|---|---|---|
| `sign` | `terse` | **survived 400 frames, score 11** | 86% | 279 ms |
| `sign` | `full` | crashed into pipe, frame 63, score 0 | 86% | 419 ms |
| `action` | `terse` | crashed into ceiling, frame 34, score 0 | 0% | 238 ms |
| `hint` | `terse` | crashed into ceiling, frame 34, score 0 | 0% | 301 ms |
| `noul` | `terse` | crashed into ceiling, frame 34, score 0 | 0% | 259 ms |
| `oracle` | — | survived 400 frames, score 11 | 100% | — |

Two separate things have to be right, and each one is a usage lesson.

**1 — Ask what the state *is*, not what to *do*.** openjev needed this trick too (its NLI
scored 3/24 on "The correct action is: flap", 21/24 on "The offset … is negative"). Laya
has a native `choice` type so it *can* be asked for the action directly — and it is still
state-blind when you do. `action` answers "flap" whether the bird is below the gap or
above it, and grows *more* confident when it is above (0.536 → 0.640); `noul` is flat at
0.946 vs 0.945. Always flapping means the ceiling on frame 34. Writing the decision rule
into the option descriptions (`hint`) does not rescue it.

**2 — Put nothing in the state you do not want answered.** With openjev's original
wording, `sign` crashed on frame 63. The state says the bird is "rising" or "falling", and
Laya reads "falling" as "negative":

| state text | offset +0.51, rising | offset +0.51, **falling** |
|---|---|---|
| full (has "falling") | p(negative) 0.079 ✓ | p(negative) **0.483** ✗ |
| terse (no velocity) | p(negative) 0.047 ✓ | p(negative) 0.047 ✓ |

A swing of 0.40 from one irrelevant word — larger than the actual offset produces. Every
time the bird began to descend it flapped, pinning it above the gap until a pipe arrived.
Deleting five words of velocity wording fixes it *and* cuts latency from 419 to 279 ms,
because state length is the main cost driver.

Two measurement notes, in case you run this yourself. It takes **71 frames to pass the
first pipe** (the bird sits at x=0.2, pipes spawn at x=1.2 and close at 0.015/frame), then
30 frames each — so `--max-steps` below ~100 scores 0 however well the bird flies. And
`sign`/`terse` still disagrees with the oracle on 14% of frames while matching its score
exactly: those are near-centre oscillations the game forgives, not the systematic bias the
`full` state induced. Oracle agreement alone is a poor metric here.

## Configuration

`~/.aar/laya.json` — all keys optional. `AAR_LAYA_CONFIG` points at a different file;
`AAR_LAYA_URL` overrides `url`. Unknown keys are rejected rather than ignored.

| Key | Default | Meaning |
|---|---|---|
| `url` | `http://127.0.0.1:8770` | Where the server listens |
| `autostart` | `on_demand` | `on_demand` \| `session` \| `off` |
| `python` | `~/.aar/laya/.venv/…` | Interpreter for the server venv |
| `model` | `convaiinnovations/laya` | HF checkpoint |
| `threads` | `0` | torch intra-op threads; `0` = torch's default |
| `max_questions` | `16` | Per-call question budget |
| `idle_timeout` | `1800` | Server exits after this many idle seconds |
| `startup_timeout` | `600` | How long to wait for the model to load |
| `request_timeout` | `120` | Per-request HTTP timeout |
| `log_file` | `~/.aar/laya/server.log` | Server log |
| `tools` | all three | Which tools to register |
| `gate` | `false` | Screen tool calls (see above) |
| `gate_threshold` | `0.85` | Block at or above this probability |
| `gate_tools` | `["bash", "acp_terminal"]` | Which tools the gate covers |

See `laya.example.json`.

## Endpoints

The server is a small FastAPI app, usable on its own:

```
GET  /health    {"status": "loading" | "ready" | "error", "device": "cpu", …}
POST /predict   {"state": <str|object>, "questions": {qid: {...}}}
                -> {"model": …, "answers": {qid: {...}}, "usage": {...}}
POST /shutdown
```

```bash
~/.aar/laya/.venv/bin/python aar_ext_laya/server.py --port 8770 --threads 4
```

The socket is bound *before* the model loads, so a second instance on the same port
fails immediately instead of downloading weights first.

## A caveat on the model

`laya` was published very recently and is at `0.1.x`. Everything that touches it is
funnelled through `LayaBackend` in `server.py`, which probes for both `predict()` (pip
wrapper) and `system_one()` (the raw `RLAgent` in the model repo) and for whether
`load()` accepts a `device` kwarg — so an API change should be a one-file fix. The
dependency is pinned `>=0.1.5,<0.2`.

Note also that Laya's calibration temperatures are fitted per checkpoint. Changing
`model` to a different checkpoint is possible but the confidences will no longer mean
what they claim to.

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -q          # 49 tests, no model or GPU needed — HTTP is mocked
```

## Licence

Apache-2.0, matching the model.
