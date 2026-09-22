# aar-ext-qwen-image

Aar extension that gives the agent **local text-to-image and image editing tools**
backed by [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) — a 7B
diffusion model that generates *and* edits, renders legible text inside pictures,
produces transparent (RGBA) layers and accepts up to ten reference images.

Qwen-Image-2.1 is not a chat model, so it doesn't replace your provider. It runs next
to it (e.g. `qwen3.8` on Ollama) as a small HTTP server in its own Python environment,
ideally on the GPU with the most VRAM. aar never imports torch or diffusers.

```
aar ──(chat + tools)──► Ollama / any provider
 └──(image_generate …)─► qwen-image server (venv, torch + diffusers, ~31 GiB of weights)
```

The transformer can be swapped for a [GGUF quantization](#quantized-transformer-gguf)
of the same weights (`quant` / `/qwenimage quant`), which drops the pipeline from
~31 GiB to ~22 GiB and is the difference between a comfortable and a strained 24 GB
card.

## Tools

| Tool | Input | Output |
|---|---|---|
| `image_generate` | `prompt`, optional `negative_prompt`, `width`, `height`, `steps`, `seed`, `transparent`, `out` | path of the saved PNG + size, seed, steps, duration |
| `image_edit` | `prompt`, `images[]` (up to 10 references), same options | path of a **new** PNG — originals are never modified |

Both tools return a **file path**, because aar tool results are plain strings — the
model never sees the pixels. To let it look at its own output, attach the file back
with `@foo.png` on a vision-capable provider.

Output names are constrained: `out` must be a bare file name, it always lands inside
the configured `out_dir`, the suffix is forced to `.png`, and an existing file is
never overwritten (`sunset.png` → `sunset-1.png`).

### Where images are saved

`out_dir` is empty by default, which means **aar's current working directory** — run
`aar` in a project and `image_generate` drops the PNG right there. Set it in
`~/.aar/qwen-image.json` to change that:

| `out_dir` | images land in |
|---|---|
| `""` *(default)* | aar's current working directory |
| `"images"` | `./images` under aar's cwd |
| `"assets/renders"` | nested subdirectory of aar's cwd |
| `"~/.aar/qwen-image/out"` | that fixed directory, whatever the cwd is |
| `"D:/art/out"` | that fixed directory, whatever the cwd is |

```json
{ "out_dir": "assets/renders" }
```

Anything that is not absolute (and does not start with `~`) is taken relative to the
working directory aar was started in, and is created on the first render. The cwd is
read at render time, not at config load, so the same relative setting follows you from
project to project — start aar in a game repo and the sprites land in that repo.

Reference images for `image_edit` are accepted in the three forms a model actually
produces:

| form | example |
|---|---|
| full or `~` path | `~/pics/neon.png` |
| aar's attachment syntax | `@~/pics/neon.png` — the leading `@` is stripped |
| bare file name | `neon.png` — resolved against `out_dir`, where `image_generate` writes |

The second matters because when a user writes `@some/pic.png` the model passes that
string straight through; the third because the model refers back to its own output by
the name it just saved. A file that genuinely does not exist is still reported as
`reference image not found`.

> **`transparent: true` produces a real RGBA cutout.** Qwen-Image-2.1 generates
> transparency natively, but it is requested in the *prompt text* — there is no
> pipeline argument for it — so the tool wraps your prompt in the model card's
> wording (`RGBA_PROMPT_PREFIX` / `RGBA_PROMPT_SUFFIX`). Measured at 1024x512:
> alpha `0..255`, 20.7% of pixels fully transparent. See
> [the sprite sheet recipe](#recipe-a-transparent-sprite-sheet-for-a-platformer).

## Slash-command

```
/qwenimage                      status (model, device, VRAM, pid)
/qwenimage devices              every GPU torch can see, and which one is in use
/qwenimage quant                list the quantizations, marking the active one
/qwenimage quant Q4_K_M         switch weights (applies on the next server start)
/qwenimage start                launch the server in the background
/qwenimage stop                 stop it and free VRAM
/qwenimage generate <prompt>    quick manual test
```

## Choosing a GPU

torch names devices by **backend**, not by vendor — a ROCm build reports an AMD card
as `cuda:0` as well — so `device` takes:

| `device` | Means |
|---|---|
| `auto` (default) | the accelerator with the most memory |
| `cuda:N` | NVIDIA, or AMD on a ROCm build of torch |
| `xpu:N` / `mps` | Intel GPU / Apple silicon |
| `dml:N` | DirectML — a last resort for AMD/Intel cards with no ROCm build |
| `cpu` | works, measured in many minutes per image |

`/qwenimage devices` lists what the server sees. Before the server runs, ask torch
directly:

```bash
python path/to/aar_ext_qwen_image/server.py --list-devices
```

**VRAM.** The bf16 weights are ~31 GiB on disk — transformer 13.3 GiB, text encoder
16.3 GiB, VAE 1.3 GiB — so the whole pipeline does *not* fit in 24 GB at once. With
`offload: "model"` each component is staged onto the GPU as it is needed, which is the
fast path on a 24 GB card; `offload: "sequential"` streams individual layers and fits
on ~8 GB, at a large speed cost. On a 6 GB card, prefer 512–768px with 20–25 steps.

Every figure below is for the unquantized transformer. Setting
[`quant`](#quantized-transformer-gguf) takes 6–9 GiB off all of them.

### Windows: size your page file before using `model` offload

`offload: "model"` keeps all ~31 GiB of bf16 weights in CPU RAM and stages one whole
component onto the GPU at a time, so a render's real cost is **committed host memory**,
not just VRAM. (A [GGUF transformer](#quantized-transformer-gguf) shrinks the resident
set by 6–9 GiB, which moves every number in this section down — but does not change the
shape of the problem, so size the page file anyway.) Windows' commit limit is RAM **plus the page file**, and when a render
asks for more than is left, the allocation fails inside `c10::alloc_cpu` and the process
dies with exception `0xC0000005` and **no Python traceback** — the client just sees the
connection close mid-render.

Measured on a 63.7 GB RAM / RX 7900 XTX box, `offload: "model"`, Qwen-Image-2.1:

| | committed host memory |
|---|---|
| after the model loads | ~48 GB |
| steady-state floor between renders | ~78 GB |
| peak during `image_generate` 1024px | ~87 GB |
| peak during `image_edit` 1024px (1 reference) | **~102 GB** |

The floor climbs over the first few renders and then plateaus; the *peak* is what kills
you, and editing costs far more transient memory than generating. On the test machine:

| commit limit | result |
|---|---|
| 83 GB (64 GB RAM + 20 GB page file) | crashed on the **second** render |
| 104 GB (+ 40 GB page file) | five renders, then an `image_edit` crashed at 102.2 GB |
| 110 GB (+ 46 GB page file) | stable, including `image_edit` |

So budget **~110 GB** (RAM + page file) if you use `image_edit`, or ~96 GB if you only
ever call `image_generate`. Raising the page file does not eliminate the ceiling — it
moves it, and editing sits close enough to the line that a few GB matter.

Check what you have:

```powershell
$os = Get-CimInstance Win32_OperatingSystem
"commit limit: {0:N1} GB" -f ($os.TotalVirtualMemorySize / 1MB)
Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize
```

To raise it, set an explicit page file size (elevated, then **reboot** — the file does
not grow until restart). 46 GB on top of 64 GB of RAM gives a ~110 GB limit:

```powershell
$pf = Get-CimInstance Win32_PageFileSetting -Filter "Name='c:\pagefile.sys'"
Set-CimInstance -InputObject $pf -Property @{InitialSize = 46960; MaximumSize = 46960}
```

The page file needs that much free disk. If the system drive is tight, put the extra
page file on another volume rather than shrinking headroom on `C:`.

Letting Windows manage the page file automatically is *not* reliable here: on the test
machine it produced a fixed 20 GB file, which was not enough.

If you cannot raise the limit, use `offload: "sequential"`. It streams individual layers
and so never makes the large allocation, at a large speed cost — measured at **72 s/step**
for 768px on an RX 7900 XTX, about 24 minutes for a 20-step image.

### Measured speed (RX 7900 XTX, native ROCm, `offload: "model"`)

| render | wall clock |
|---|---|
| model load (weights cached on disk) | 23-39 s |
| 512px, 8 steps | ~170 s |
| 1024px, 30 steps | ~214 s |
| 1024px, 30 steps, `image_edit` with 1 reference | 228-385 s |

Two things dominate and are easy to misread:

- **The first step of a fresh process costs ~92 s** — AOTriton compiles its attention
  kernels once. Steady-state denoising is ~**1.4 s/step** at 1024px, so a per-step
  average taken from one short render is wildly pessimistic.
- **Encode and VAE decode add minutes**, independent of step count. At 8 steps they are
  most of the wall clock; at 30 steps they are roughly half.

Size timeouts from the *total*, not from s/step. `request_timeout` must exceed the whole
render, and aar's own `tools.command_timeout` (default **300 s**) must exceed it too —
otherwise the executor cancels the tool call while the GPU keeps working, and the
finished image is thrown away. For `image_edit` at 1024px, 300 s is not enough:

```jsonc
// ~/.aar/config.json
{ "tools": { "command_timeout": 1800 } }
```

### AMD on Windows: native ROCm (no WSL)

AMD ships **native Windows PyTorch wheels** built on ROCm, so a Radeon card needs no
WSL and no launcher — `device: "auto"` finds it and reports it as `cuda:0`, because a
ROCm torch build uses the CUDA device names. Requirements for the 7.2.1 release:
**Python 3.12** (the wheels are `cp312`) and Adrenalin **26.2.2 or newer**. Check the
driver with:

```powershell
Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' |
  ForEach-Object { (Get-ItemProperty $_.PSPath).RadeonSoftwareVersion }
```

Then, into the server venv — the ROCm SDK first, PyTorch second:

```powershell
$py = "$HOME\.aar\qwen-image\.venv\Scripts\python.exe"
& $py -m pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
& $py -m pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
& $py -m pip install git+https://github.com/huggingface/diffusers `
    transformers accelerate safetensors pillow fastapi uvicorn "gguf>=0.10"
```

Version numbers move — take the current ones from
[AMD's Windows PyTorch page](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html).
Only PyTorch is supported natively on Windows; the full ROCm stack remains Linux-only.

### AMD fallback: ROCm inside WSL2

If native Windows wheels don't cover your card, run the server in a WSL2 distro and
let aar launch it through a *launcher* prefix. AMD ships ROCm WSL packages for
**Ubuntu 22.04 (jammy) and 24.04 (noble) only** — a 26.04 distro won't work, and its
Python 3.14 has no torch wheels either.

1. Install ROCm inside the distro per
   [AMD's WSL instructions](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html)
   (`amdgpu-install` from `repo.radeon.com`, then
   `amdgpu-install --usecase=wsl,rocm --no-dkms`).
2. Build the venv with `python3.12` and install the Linux ROCm torch wheels plus
   `diffusers` (from git, see below), `transformers`, `accelerate`, `safetensors`,
   `pillow`, `fastapi`, `uvicorn` and — for `quant` — `gguf>=0.10`.
3. Point the extension at it from Windows (`~/.aar/qwen-image.json`):

   ```json
   {
     "launcher": ["wsl.exe", "-d", "aar-ubuntu", "--"],
     "python": "/home/you/.aar/qwen-image/.venv/bin/python",
     "server_script": "/mnt/b/.../aar_ext_qwen_image/server.py",
     "device": "auto",
     "offload": "model"
   }
   ```

   `server_script` is the path **as WSL sees it**; the Windows checkout under `/mnt/…`
   works, nothing needs copying. WSL2's localhost forwarding makes the server
   reachable at `127.0.0.1:8770` from Windows. If that proves flaky, run the server
   with `--host 0.0.0.0` and set `url` to the distro's IP.

## Quantized transformer (GGUF)

`quant` loads the transformer from a GGUF quantization of the *same* Qwen-Image-2.1
weights instead of the published bf16 tensors. Only the transformer is quantized —
the text encoder and VAE are unchanged, and still come from `Qwen/Qwen-Image-2.1`.

| `quant` | file | size | note |
|---|---|---|---|
| `none` (default) | base checkpoint | 13.3 GiB | published bf16 weights |
| `Q8_0` | `qwen-image-2.1-Q8_0.gguf` | 7.1 GiB | closest to bf16 |
| `Q6_K` | `qwen-image-2.1-Q6_K.gguf` | 5.5 GiB | |
| `Q5_K_M` | `qwen-image-2.1-Q5_K_M.gguf` | 4.9 GiB | |
| `Q4_K_M` | `qwen-image-2.1-Q4_K_M.gguf` | 4.3 GiB | **recommended** |
| `Q4_0` | `qwen-image-2.1-Q4_0.gguf` | 3.8 GiB | smallest |

Sizes are GiB, to match how VRAM is counted everywhere else here;
[the model card](https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF) lists the same
files in decimal GB, so its numbers run ~7% higher.

```
/qwenimage quant                 # what is available, and what is active
/qwenimage quant Q4_K_M          # write it to ~/.aar/qwen-image.json
/qwenimage stop                  # then restart to load it
```

The switch is persisted to the config file; a server that is already running keeps
the weights it started with, so `/qwenimage status` reports the *server's* quant, not
the config's, while one is loaded.

Equivalent config, if you prefer to edit the file:

```json
{ "quant": "Q4_K_M" }
```

The file is fetched into the normal Hugging Face cache on first load. To use a
quantization you built yourself, point `quant_file` at it (an existing path wins over
any download); a bare file name is looked up in `quant_repo` instead.

### Why it helps on a 24 GB card

Weights only have to *be* in VRAM — getting them there is the expensive part. On an
RX 7900 XTX the card itself does ~770 GB/s, but host↔device transfer runs at about
2.5 GB/s:

```
Vector Addition: 772.88 GB/s          float16  : 102.03 TFLOPS
Memory Copy:     710.93 GB/s          bfloat16 :  97.78 TFLOPS
CPU -> GPU:      2.56 GB/s
GPU -> CPU:      2.80 GB/s
```

`offload: "model"` stages one whole component onto the GPU at a time, so every render
pays that 2.5 GB/s for the transformer. At bf16 that is 13.3 GiB ≈ **5.6 s** of pure
copying per render; at `Q4_K_M` it is 4.3 GiB ≈ **1.8 s**. The same 9 GiB comes off the
committed host memory the page-file section below is about.

With a quantized transformer the whole pipeline is ~22 GiB instead of ~31 GiB, which
puts `offload: "none"` within reach on a 24 GB card — everything stays resident and
renders stop paying for staging at all. It is tight (the bf16 text encoder alone is
16.3 GiB), so treat it as worth trying rather than a safe default: if it OOMs, go back
to `offload: "model"`, which is still meaningfully faster and lighter than it was with
bf16 weights.

`dtype` stays `bfloat16` — it is the compute dtype the dequantized blocks are
promoted to. The benchmark above shows fp16 marginally ahead of bf16 on this card
(102 vs 98 TFLOPS), but the margin is well inside noise for a render dominated by
attention and VAE decode, and bf16's exponent range is what the model was trained in.

### Requirements

The `gguf` package must be in the *server* venv (it is in the `server` extra):

```bash
pip install "gguf>=0.10"
```

diffusers reads the file with its own `GGUFQuantizationConfig` — this is not
ComfyUI-GGUF and needs no custom nodes. The server loads the transformer with
`from_single_file`, then hands it to `from_pretrained`, which skips downloading the
base transformer entirely.

One wrinkle is handled for you. diffusers dispatches `from_single_file` through a
table of known classes, and `QwenImage21Transformer2DModel` does not subclass the 1.0
transformer, so the lookup misses it and loading would fail. The server registers the
same identity entry the 1.0 class uses (`ensure_single_file_loadable`), which is exact
rather than a guess: all 297 tensors in these files already carry diffusers' own
parameter names — `transformer_blocks.N.attn.to_q.weight`, `modulation.1.weight`,
`img_mlp.gate_layer.weight` — so nothing is converted. It becomes a no-op the day
diffusers ships an entry of its own.

The text encoder and VAE in that repo are ComfyUI-packaged single files and are *not*
used; `transformers` wants the sharded `Qwen3VLForConditionalGeneration` layout from
the base checkpoint, so the 16.3 GiB text encoder is still downloaded once.

## diffusers version

Qwen-Image-2.1 landed on **diffusers main** before any tagged release carried it —
0.40.0 ships `QwenImagePipeline`, `QwenImageEditPipeline`, `QwenImageEditPlusPipeline`
and `QwenImageLayeredPipeline`, but no `QwenImage21Pipeline`. So install diffusers
from git:

```bash
pip install -U git+https://github.com/huggingface/diffusers
```

The server does not hardcode a class name. With `pipeline: "auto"` (the default) it
hands the checkpoint to `DiffusionPipeline.from_pretrained`, which reads
`model_index.json` and instantiates whatever class the weights were published with;
`/qwenimage status` then reports the class that was actually used. For editing, the
server prefers that same pipeline when its `__call__` accepts `image=` (the unified
case); only if it doesn't does it fall back to a separate Edit class built from the
**already-loaded weights** — never a second copy in VRAM. Set `pipeline` explicitly to
override, e.g. `"QwenImageLayeredPipeline"` for transparent-layer work.

Verified against diffusers `0.41.0.dev0`: the checkpoint's `model_index.json` declares
`QwenImage21Pipeline`, and that class takes `prompt`, `image`, `negative_prompt`,
`true_cfg_scale`, `height`, `width`, `num_inference_steps`, `generator` — so
Qwen-Image-2.1 is genuinely unified and `image_edit` drives the same pipeline with
`image=`. It has **no** transparency *keyword* — but that is not a missing feature:
Qwen-Image-2.1 produces RGBA natively when the prompt asks for it, so `transparent:
true` wraps the prompt rather than passing an argument the pipeline would reject.

## Setup

### 1. Server environment (once)

Keep the ML stack out of aar's environment. NVIDIA, native Windows or Linux:

```powershell
python -m venv $HOME\.aar\qwen-image\.venv
$py = "$HOME\.aar\qwen-image\.venv\Scripts\python.exe"
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install git+https://github.com/huggingface/diffusers `
    transformers accelerate safetensors pillow fastapi uvicorn
& $py -m pip install "gguf>=0.10"       # only needed for quant != "none"
# optional: pre-download the weights (~31 GiB) instead of waiting on the first render
& "$HOME\.aar\qwen-image\.venv\Scripts\hf.exe" download Qwen/Qwen-Image-2.1
# with quant: the base transformer is never fetched, so pull the GGUF instead
& "$HOME\.aar\qwen-image\.venv\Scripts\hf.exe" download `
    abenzerps/Qwen-Image-2.1-GGUF qwen-image-2.1-Q4_K_M.gguf
```

For AMD, use the ROCm wheels from *Choosing a GPU* above instead of the `cu128` index.
Linux/macOS use `~/.aar/qwen-image/.venv/bin/python`.

### 2. Extension (into aar's environment)

```bash
pip install aar-ext-qwen-image           # or: pip install -e packages/aar-ext-qwen-image
aar extensions list                      # should show "qwen_image (entrypoint)"
```

Or copy the `aar_ext_qwen_image/` directory into `~/.aar/extensions/`.

### 3. Use it

```bash
aar chat
> /qwenimage start            # optional — the first tool call starts it anyway
> /qwenimage devices          # confirm it picked the card you meant
> /qwenimage status           # wait for "ready" (first load reads ~31 GB of weights)
> /qwenimage generate a vintage travel poster for Reykjavik, bold lettering
```

Then ask the agent in plain language. Three prompts that exercise the whole surface:

```
# legible in-image text — the model's headline strength
> Use image_generate to make a 1024x1024 neon shop sign that reads "QWEN IMAGE 2.1",
  rainy night, reflections on wet pavement. 30 steps, seed 42, save as neon.png

# native RGBA, no keying step
> Use image_generate with transparent=true to make a 768x768 cute cartoon dragon
  sticker with bold outlines. 25 steps, seed 99, save as dragon.png

# editing an existing picture — @path, a plain path or the bare name all work
> Take @neon.png and use image_edit to change it to a bright snowy morning with the
  sign switched off. 30 steps, save as neon-winter.png
```

The edit keeps the original's composition — same sign, same camera — and changes only
what you asked for. If you instead get an unrelated fresh image, the model called
`image_generate`; `~/.aar/qwen-image/server.log` shows `0 refs` on that render
rather than `1 refs`.

## Driving the sidecar from a normal aar agent

The image tools are *ordinary aar tools*. Your chat provider (`qwen3.8` on Ollama, an
Anthropic model, anything) keeps doing the talking and the coding; it just gains two
extra tools that happen to be backed by a 7B diffusion model on your GPU. Nothing
special is needed to combine them with `write_file` and `bash`.

### A. One agent, both jobs (start here)

Installed as an entrypoint (`pip install aar-ext-qwen-image`), the extension loads into
**every** aar process, so `image_generate` sits next to the built-in tools in the same
registry. Start aar in the project you are building and ask for both halves in one go:

```bash
cd ~/games/dino
aar tui
```

```
> Make a game asset and then use it:
  1. image_generate, transparent=true, 1024x512, seed 2024, save as player.png —
     pixel-art side view of a small green dinosaur running, bold dark outlines,
     flat colours, no background.
  2. image_generate, transparent=true, 512x256, seed 2025, save as cactus.png —
     matching pixel-art cactus obstacle, same style.
  3. Write index.html + game.js: a canvas endless-runner that loads player.png and
     cactus.png from this directory, space to jump, score counter.
  4. Open index.html with the default browser and tell me what to check.
```

With the default `out_dir` (`""`) both PNGs land in `~/games/dino` — the same directory
the agent is writing `game.js` into — so the `<img src="player.png">` the model writes
just works. That is the whole point of the cwd default: **the picture and the code end
up in the same place**, and the model does not have to reason about
`~/.aar/qwen-image/out` paths it cannot see.

A render takes tens of seconds to minutes. The agent blocks on the tool call, so give
it room: `max_steps` high enough for both images plus the code, and a `request_timeout`
in `qwen-image.json` that covers your slowest size.

### B. A dedicated image sub-agent

Declare an `illustrator` profile and the coding agent can delegate to it with the
built-in [`spawn_agent`](../../../docs/configuration.md#sub-agents-spawn_agent) tool —
one nested agent whose entire tool surface is `image_generate` + `image_edit`.

```json
{
  "subagents": {
    "enabled": true,
    "agents": {
      "illustrator": {
        "description": "Generates a single image from a description and returns its path",
        "tools": [],
        "system_prompt": "You are an image generator. Call image_generate exactly once, then reply with only the saved file path. Never explain.",
        "max_steps": 6,
        "timeout": 900
      }
    }
  }
}
```

`tools: []` strips `read_file` / `bash` / everything else; extension tools are
registered separately and survive, so the child cannot read your files, write anything
but a PNG, or run a command. `timeout: 900` becomes the tool's own `timeout_s`, so a
slow render is not cut short by `tools.command_timeout`.

```
> Use spawn_agent with the illustrator for a pixel-art green dinosaur running,
  transparent, 1024x512, seed 2024, saved as player.png — then build the game
  around whatever path it reports.
```

The child returns only its final message (the saved path), so prompt iterations never
enter the coding agent's context. It runs in the same process and the same working
directory, so `out_dir` still resolves to the project you started aar in.

**Out-of-process variant.** Without `subagents`, the same shape works by shelling out:

```bash
aar run --config ~/.aar/image-agent.json "pixel-art dinosaur, transparent, save as player.png"
```

where that config carries the same `tools.enabled_builtins: []` and a
`system_prompt_override`. Note `system_prompt` in a config file is the *assembled*
prompt and is rebuilt on startup — `system_prompt_override` is the key that replaces
it. Both processes share one sidecar, so the second `aar` finds the already-loaded
model on `http://127.0.0.1:8770`. Pin a different output directory per agent with
`AAR_QWEN_IMAGE_CONFIG=~/.aar/qwen-image-assets.json`, where that file sets
`{"out_dir": "assets"}`.

**When B is worth it:** a long coding session where you don't want image prompts,
renders and retries eating the main context window; or a cheaper model for
prompt-writing than the one doing the coding (`"provider"` on the profile). **When A is
better:** almost everything else — one less moving part, and the coding model keeps the
seeds and file names it just chose in context.

### Letting the agent look at what it made

Tool results are strings, so neither agent sees pixels. On a vision-capable provider,
attach the file back explicitly:

```
> @player.png — is the dinosaur facing right, and is the background actually clear?
  If not, use image_edit to fix it.
```

Under the cwd default that is a short relative path, which is also why `image_edit`
accepts a bare file name and resolves it against `out_dir`.

## Supported sizes

Qwen-Image-2.1 is trained on these aspect ratios; the model card uses the sizes on the
right, and all of them are within the default `max_pixels`:

| ratio | size | | ratio | size |
|---|---|---|---|---|
| 1:1 | 2048 x 2048 | | 3:2 | 2528 x 1696 |
| 4:3 | 2400 x 1792 | | 2:3 | 1696 x 2528 |
| 3:4 | 1792 x 2400 | | 16:9 | 2752 x 1536 |
| | | | 9:16 | 1536 x 2752 |

The tool defaults to **1024 x 1024 at 30 steps** rather than the card's 2048 x 2048 at
40, because a 2048px render on an offloaded 24 GB card takes several minutes and a
small card may not finish at all. Ask for a bigger size explicitly when you want it —
`max_pixels` (default `2400 * 1792`) is the ceiling, and the server independently caps
each side at 4096.

## Recipe: a transparent sprite sheet for a platformer

> For the full end-to-end version of this — sub-agent config, two-GPU pinning, slicing the
> sheet in a canvas game, and the timeouts involved — see
> [`docs/sprite-sheet-workflow.md`](../../../docs/sprite-sheet-workflow.md) in the aar repo.

Qwen-Image-2.1 has **native RGBA output**. There is no `transparent` argument on the
pipeline — transparency is requested in the *prompt*, using the model card's wording:

```
This is an RGBA image with transparency. <your subject>.
The image has alpha channel and the background is transparent.
```

`transparent: true` applies that wrapping for you, so from the TUI (`aar tui`) you can
just ask:

```
> Use image_generate with transparent=true to make a 1024x512 pixel-art sprite sheet:
  one row of 4 frames of a small green dinosaur in a jump cycle (crouch, launch,
  mid-air, land), side view, crisp pixel art, bold dark outlines, evenly spaced
  frames in a single horizontal row. Use 30 steps, seed 2024, save as sprite.png
```

Measured on an RX 7900 XTX, 1024x512, 30 steps, 184 s:

```
mode: RGBA   ignored: []
alpha range: 0..255
fully transparent px: 108465/524288 (20.7%)
```

That is a real cutout — drop it straight into your engine, no keying step.

What actually matters in the prompt:

| say this | why |
|---|---|
| `transparent=true` (or the RGBA wording) | without it you get an opaque background |
| `bold dark outlines` | keeps sprite edges readable against any level art |
| `one row of N frames`, `evenly spaced` | the model drifts into a diagonal arc otherwise |
| *don't* also ask for a background colour | it competes with the alpha channel |

### Known limitations

- **Frames are not on a uniform grid.** The model places them by eye, so slicing at
  `width / N` will not line up, and it sometimes returns a different frame count than
  asked (a 4-frame request came back with 5 plus a stray object). For production
  sheets, generate one pose per render with the *same seed* and composite the grid
  yourself — that also keeps the character consistent frame to frame.
- **It is not true pixel art.** The output is a raster *imitating* pixel art, with
  anti-aliased edges and semi-transparent pixels around the outline (97.8% of pixels
  were non-opaque in the measurement above — only 20.7% were *fully* clear). Downscale
  to your real sprite size with `Image.NEAREST` and threshold the alpha if your engine
  needs a hard 1-bit mask:

  ```python
  from PIL import Image

  im = Image.open("sprite.png").convert("RGBA")
  a = im.getchannel("A").point(lambda v: 255 if v > 128 else 0)
  im.putalpha(a)
  im.resize((256, 128), Image.NEAREST).save("sprite-hard.png")
  ```

- **Chroma-keying is still useful as a fallback** — if a render comes back opaque
  despite the flag, re-render on a `flat solid magenta background` and key it out.

## Configuration

`~/.aar/qwen-image.json`: every key is optional, and unknown keys are rejected. JSON
has no comments, so start from
[`qwen-image.example.json`](qwen-image.example.json), which lists every key with its
default except `autostart`, set to `off` for a self-managed server:

```bash
cp packages/aar-ext-qwen-image/qwen-image.example.json ~/.aar/qwen-image.json
```

| Key | Default | Notes |
|---|---|---|
| `autostart` | `on_demand` | `on_demand`: first tool call starts the server; `session`: start when a session starts; `off`: only `/qwenimage start` |
| `device` | `auto` | see *Choosing a GPU* |
| `dtype` | `bfloat16` | `float16` for cards without bf16; `float32` doubles memory |
| `offload` | `model` | `none` (needs ~31 GiB VRAM, or ~22 GiB with `quant`), `model` (fast path on 24 GB), `sequential` (smallest, slowest) |
| `quant` | `none` | GGUF quantization of the transformer — see [Quantized transformer](#quantized-transformer-gguf) |
| `quant_repo` | `abenzerps/Qwen-Image-2.1-GGUF` | where the `.gguf` files are fetched from |
| `quant_file` | `null` | explicit `.gguf` path, or a file name inside `quant_repo`; overrides `quant` |
| `launcher` | `[]` | argv prefix for the server, e.g. `["wsl.exe", "-d", "Ubuntu-24.04", "--"]` |
| `server_script` | `null` | path to `server.py` as the launcher sees it |
| `cuda_visible_devices` / `hip_visible_devices` | `null` | set to hide other cards from the server |
| `out_dir` | `""` | where generated PNGs land: empty = aar's current working directory, a relative path like `images` or `assets/renders` = that subdirectory of the cwd, an absolute or `~` path = exactly that directory (created on first render) |
| `width` / `height` / `steps` | `1024` / `1024` / `30` | the model card's example uses 2048px and 40 steps |
| `max_pixels` | `4300800` (2400x1792) | requests above this are refused before reaching the GPU; covers every documented aspect ratio |
| `idle_timeout` | `900` | server exits after this many idle seconds (**0 = never** — set this on a slow bus / eGPU, where rebuilding the pipeline costs ~2 minutes) |
| `vae_tiling` / `vae_slicing` / `attention_slicing` | `false` | Opt-in activation-memory reducers, applied after placement. They lower the per-step peak rather than the weight footprint, so they matter with `offload: "none"`. Best-effort: a helper this pipeline lacks logs a warning instead of failing to start |
| `request_timeout` | `900` | a large image on an offloaded GPU takes minutes |
| `tools` | all | subset of tools to expose to the model |

Environment overrides: `AAR_QWEN_IMAGE_CONFIG` (config file path), `AAR_QWEN_IMAGE_URL`.

## Notes

- **Server lifetime.** The server is started detached, so it survives the aar process
  and later `aar run` calls reuse the loaded pipeline. It stops after `idle_timeout`
  or `/qwenimage stop`. Its stdin/stdout are never inherited, which keeps `aar acp`
  (JSON-RPC over stdio) safe.
- **Safety.** The tools only talk to the configured loopback URL and only write inside
  `out_dir` (`side_effects: network, write`; `image_edit` also reads the reference
  images). The server command comes from your config, never from the model.
- **Serialised renders.** One diffusion pipeline cannot run two prompts at once, so
  requests queue behind a lock. Ask for one image at a time.
- **Moving target.** diffusers' Qwen-Image support is new. Unknown pipeline options
  (`true_cfg_scale`, `output_resolution`, …) are filtered against the installed pipeline's
  real signature and reported in the tool result as *ignored* rather than raising, and
  the edit pipeline class is resolved from several candidate names. If your diffusers
  build exposes a different class, set `--pipeline` / `pipeline` explicitly.
- **Multi-GPU boxes.** If Ollama also sees the card you want for rendering, they will
  compete for VRAM. Hide one from the other with `CUDA_VISIBLE_DEVICES` /
  `HIP_VISIBLE_DEVICES`, or put the renderer on the larger card and keep the LLM on
  the smaller one.

## Tests

```bash
cd packages/aar-ext-qwen-image
python -m pytest tests/ -v                          # mocked HTTP + fake backend, no GPU
AAR_QWEN_IMAGE_LIVE=1 python -m pytest tests/ -v    # plus a live render against the real server
```

## License

Apache License 2.0 — same as Aar. The Qwen-Image-2.1 weights are covered by the
[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1), which is *not* a
plain open-source licence — check it before any commercial use. The GGUF files are a
requantization of those same weights and carry the same licence; note also that,
unlike the base checkpoint, that repo advertises itself as having no content filter.
