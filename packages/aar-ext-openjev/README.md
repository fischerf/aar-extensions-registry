# aar-ext-openjev

Aar extension that gives the agent **local NLI verification tools** backed by
[AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev), a Qwen3.5‑4B
cross-encoder that labels a *(premise, hypothesis)* pair as **entailment**,
**contradiction** or **neutral**.

openjev is not a chat model, so it doesn't replace your provider. It runs next to it
(e.g. `qwen3.8` on Ollama) as a small HTTP server in its own Python environment,
ideally on a spare GPU. aar never imports torch.

```
aar ──(chat + tools)──► Ollama / any provider
 └──(nli_check …)─────► openjev server (venv, torch + bitsandbytes, 4-bit on a 6 GB GPU)
```

## Tools

| Tool | Input | Output |
|---|---|---|
| `nli_check` | `premise`, `claims[]` | per claim: SUPPORTED / CONTRADICTED / NOT ESTABLISHED + probabilities |
| `nli_rerank` | `question`, `options[]` | options ranked by P(entailment) of "The correct answer is: …" |
| `nli_grade` | `question`, `reference`, `answer` | CORRECT / INCORRECT / UNVERIFIED |

The verifier only judges what the premise says, not world knowledge. Keep claims short and
self-contained. Long premises are cut from the end so the hypothesis always fits (the tool
output says when this happened).

## Slash-command

```
/openjev                               status (model, device, VRAM, pid)
/openjev start                         launch the server in the background
/openjev stop                          stop it and free VRAM
/openjev check <premise> => <claim>    quick manual test
```

## Setup

### 1. Server environment (once)

The server needs `transformers>=5.15` (Qwen3.5) and a CUDA build of torch. Keep it out of
aar's environment:

```powershell
python -m venv $HOME\.aar\openjev\.venv
$py = "$HOME\.aar\openjev\.venv\Scripts\python.exe"
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install "transformers>=5.15" accelerate "bitsandbytes>=0.45" fastapi uvicorn
# optional: pre-download the weights (~9 GB, only the model subfolder)
& "$HOME\.aar\openjev\.venv\Scripts\hf.exe" download AlexWortega/openjev --include "qwen3.5-4b-nli/*"
```

Linux/macOS: use `~/.aar/openjev/.venv/bin/python`. The extension looks for exactly these
default paths.

### 2. Extension (into aar's environment)

```bash
pip install aar-ext-openjev            # or: pip install -e packages/aar-ext-openjev
aar extensions list                    # should show "openjev (entrypoint)"
```

Or copy the `aar_ext_openjev/` directory into `~/.aar/extensions/`.

### 3. Use it

```bash
aar chat
> /openjev start          # optional — the first tool call starts it anyway
> /openjev status         # wait for "ready" (first load: weights are read + quantized)
> /openjev check A man is playing a guitar on stage. => Someone is making music.
```

Then ask the agent things like *"Read CHANGELOG.md and verify with nli_check whether
version 0.4 dropped Python 3.10 support."*

### Running the server yourself

Set `"autostart": "off"` in `~/.aar/openjev.json` so aar only connects and never launches
anything, then start the server in its own terminal:

```powershell
& "$HOME\.aar\openjev\.venv\Scripts\python.exe" path\to\aar_ext_openjev\server.py `
  --bits 4 --device cuda:0 --idle-timeout 0      # Ctrl+C to stop and free VRAM
```

`server.py` is standalone and only needs the server venv. `--help` lists every flag (`--port`,
`--max-length`, `--batch-size`, …). Set `HF_HUB_OFFLINE=1` to start without contacting the
Hugging Face Hub.

## Configuration

`~/.aar/openjev.json`: every key is optional, and unknown keys are rejected. JSON has no
comments, so start from [`openjev.example.json`](openjev.example.json), which lists every
key with its default except `autostart`, set to `off` for a self-managed server:

```bash
cp packages/aar-ext-openjev/openjev.example.json ~/.aar/openjev.json
```

```json
{
  "url": "http://127.0.0.1:8765",
  "autostart": "off",
  "python": "~/.aar/openjev/.venv/Scripts/python.exe",
  "model": "AlexWortega/openjev",
  "subfolder": "qwen3.5-4b-nli",
  "bits": 4,
  "device": "cuda:0",
  "cuda_visible_devices": null,
  "max_length": 2048,
  "batch_size": 4,
  "idle_timeout": 1800,
  "startup_timeout": 240,
  "request_timeout": 120,
  "log_file": "~/.aar/openjev/server.log",
  "tools": ["nli_check", "nli_rerank", "nli_grade"]
}
```

| Key | Default | Notes |
|---|---|---|
| `autostart` | `on_demand` | `on_demand`: first tool call starts the server; `session`: start in the background when a session starts; `off`: only `/openjev start` |
| `bits` | `4` | `4` = NF4 (~3–3.5 GB VRAM), `8` = LLM.int8 (~5 GB), `16` = bf16 (~9 GB) |
| `device` | `cuda:0` | index *after* `cuda_visible_devices` filtering |
| `cuda_visible_devices` | `null` | set `"0"` if you hid NVIDIA GPUs globally (e.g. to keep Ollama on another card) |
| `idle_timeout` | `1800` | server exits after this many idle seconds (0 = never), like Ollama's `keep_alive` |
| `max_length` | `2048` | token cap per pair; lower it if you run out of VRAM on long premises |
| `tools` | all | subset of tools to expose to the model |

Environment overrides: `AAR_OPENJEV_CONFIG` (config file path), `AAR_OPENJEV_URL`.

## Notes

- **Server lifetime.** The server is started detached, so it survives the aar process and
  later `aar run` calls reuse the loaded model. It stops after `idle_timeout` or `/openjev stop`.
  Its stdin/stdout are never inherited, which keeps `aar acp` (JSON-RPC over stdio) safe.
- **Safety.** The tools only talk to the configured loopback URL (`side_effects: network`). The
  server command comes from your config, never from the model.
- **Multi-GPU laptops.** If Ollama also sees the NVIDIA card, it may place layers there and
  compete for VRAM. Check Ollama's server log, and hide the card from Ollama with
  `CUDA_VISIBLE_DEVICES=-1` if needed. Then set `"cuda_visible_devices": "0"` here.
- **Speed.** On Windows, Qwen3.5's linear-attention layers use the pure-PyTorch fallback
  (`flash-linear-attention` / `causal-conv1d` kernels aren't installed). This is fine for short
  NLI pairs, but slower on multi-thousand-token premises.

## Showcase: openjev plays Flappy Bird

[`examples/flappy_nli.py`](examples/flappy_nli.py) is a zero-shot game demo adapted from the model
authors' `code/flappy.py`. It uses the same physics, state text and oracle. Each frame, the
game state is written as a premise, and the model scores one hypothesis per action. Nothing is
trained. It runs in aar's environment against the running server and redraws the game live in
the terminal:

```bash
python examples/flappy_nli.py                        # live, "sign" hypotheses
python examples/flappy_nli.py --hyp action           # the phrasing that fails
python examples/flappy_nli.py --policy oracle --baselines --episodes 5 --no-render
python examples/flappy_nli.py --max-steps 900 --replay flappy.json
```

| `--hyp` | flap if entailed … | do nothing if entailed … |
|---|---|---|
| `sign` | The offset relative to the gap centre is negative. | …is positive. |
| `position` | The bird is below the centre of the gap. | …above the centre of the gap. |
| `action` | The correct action is: flap | The correct action is: do nothing |

The phrasing decides whether it works. Statements about the *state* play the game; action names
don't. On an RTX 3060 Laptop (4-bit), `sign` and `position` each agreed with the oracle on 21 of
24 sampled states, and `action` on 3 of 24. A decision takes about 400 ms, so the game is
turn-based: it waits for the model instead of running at 15 fps. The oracle scores 28 in 900
frames, while random and do-nothing crash at 0.

`--replay` writes the upstream replay format, so openjev's `code/flappy_video.py --json flappy.json
--policy nli --out flappy.mp4` can render an MP4 (needs `imageio[ffmpeg]`, `pillow`, `matplotlib`).

## Tests

```bash
cd packages/aar-ext-openjev
python -m pytest tests/ -v                        # mocked HTTP + fake backend, no GPU
AAR_OPENJEV_LIVE=1 python -m pytest tests/ -v     # plus a live check against the real server
```

## License

Apache License 2.0 — same as Aar. The openjev model is MIT-licensed.
