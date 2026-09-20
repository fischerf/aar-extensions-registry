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
 └──(image_generate …)─► qwen-image server (venv, torch + diffusers, ~31 GB of weights)
```

## Tools

| Tool | Input | Output |
|---|---|---|
| `image_generate` | `prompt`, optional `negative_prompt`, `width`, `height`, `steps`, `seed`, `transparent`, `out` | path of the saved PNG + size, seed, steps, duration |
| `image_edit` | `prompt`, `images[]` (up to 10 paths), same options | path of a **new** PNG — originals are never modified |

Both tools return a **file path**, because aar tool results are plain strings — the
model never sees the pixels. To let it look at its own output, attach the file back
with `@~/.aar/qwen-image/out/foo.png` on a vision-capable provider.

Output names are constrained: `out` must be a bare file name, it always lands inside
the configured `out_dir`, the suffix is forced to `.png`, and an existing file is
never overwritten (`sunset.png` → `sunset-1.png`).

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

**VRAM.** The bf16 weights are ~31 GB on disk — transformer 13.3 GB, text encoder
16.3 GB, VAE 1.3 GB — so the whole pipeline does *not* fit in 24 GB at once. With
`offload: "model"` each component is staged onto the GPU as it is needed, which is the
fast path on a 24 GB card; `offload: "sequential"` streams individual layers and fits
on ~8 GB, at a large speed cost. On a 6 GB card, prefer 512–768px with 20–25 steps.

### Windows: size your page file before using `model` offload

`offload: "model"` keeps all ~31 GB of bf16 weights in CPU RAM and stages one whole
component onto the GPU at a time, so a render's real cost is **committed host memory**,
not just VRAM. Windows' commit limit is RAM **plus the page file**, and when a render
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
you, and editing costs far more transient memory than generating. Budget a commit limit
of **at least 112 GB** (RAM + page file) for comfortable `image_edit` use, or ~96 GB if
you only ever call `image_generate`.

Check what you have:

```powershell
$os = Get-CimInstance Win32_OperatingSystem
"commit limit: {0:N1} GB" -f ($os.TotalVirtualMemorySize / 1MB)
Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize
```

To raise it, set an explicit page file size (elevated, then **reboot** — the file does
not grow until restart). 40 GB on top of 64 GB of RAM gives a ~104 GB limit:

```powershell
$pf = Get-CimInstance Win32_PageFileSetting -Filter "Name='c:\pagefile.sys'"
Set-CimInstance -InputObject $pf -Property @{InitialSize = 40960; MaximumSize = 40960}
```

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
| 1024px, 30 steps, `image_edit` with 1 reference | ~385 s |

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
    transformers accelerate safetensors pillow fastapi uvicorn
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
   `pillow`, `fastapi`, `uvicorn`.
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
# optional: pre-download the weights (~31 GB) instead of waiting on the first render
& "$HOME\.aar\qwen-image\.venv\Scripts\hf.exe" download Qwen/Qwen-Image-2.1
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

Then ask the agent things like *"Generate a 768x768 logo for this project and save it
as logo.png"*, or *"Take `@shot.png` and use image_edit to replace the background with
a plain white studio backdrop."*

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
| `offload` | `model` | `none` (needs ~31 GB VRAM), `model` (fast path on 24 GB), `sequential` (smallest, slowest) |
| `launcher` | `[]` | argv prefix for the server, e.g. `["wsl.exe", "-d", "Ubuntu-24.04", "--"]` |
| `server_script` | `null` | path to `server.py` as the launcher sees it |
| `cuda_visible_devices` / `hip_visible_devices` | `null` | set to hide other cards from the server |
| `out_dir` | `~/.aar/qwen-image/out` | every generated PNG lands here |
| `width` / `height` / `steps` | `1024` / `1024` / `30` | the model card's example uses 2048px and 40 steps |
| `max_pixels` | `4300800` (2400x1792) | requests above this are refused before reaching the GPU; covers every documented aspect ratio |
| `idle_timeout` | `900` | server exits after this many idle seconds (0 = never), like Ollama's `keep_alive` |
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
plain open-source licence — check it before any commercial use.
