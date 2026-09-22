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
/qwenimage                        status (model, device, VRAM, pid)
/qwenimage devices                every GPU torch can see, and which one is in use
/qwenimage quant                  list the quantizations, marking the active one
/qwenimage quant Q4_K_M           switch weights (applies on the next server start)
/qwenimage start                  launch the server in the background
/qwenimage stop                   stop it and free VRAM
/qwenimage generate <prompt>      render without going through the model
/qwenimage edit <image> <prompt>  edit a picture without going through the model
```

### Rendering from the command, not through the model

`generate` and `edit` call the sidecar directly. Nothing about the request is inferred
from prose, so this is the way to render when the chat model keeps dropping your
parameters — see
[When the model drops your parameters](#c-when-the-model-drops-your-parameters).

```
/qwenimage edit old-photo.png --size 1600x960 --steps 30 --seed 42 --out colour.png
  add natural realistic colour, restore detail, clean up noise and compression artefacts
```

| Flag | Meaning |
|---|---|
| `--size WxH` | both dimensions at once; `--width` / `--height` (`--w` / `--h`) also work |
| `--steps N` | denoising steps |
| `--seed N` | fix the seed — the one parameter that has no config key, and the one that makes two renders comparable |
| `--out name.png` | file name inside `out_dir`; same rules as the tool (bare name, `.png`, never overwrites) |
| `--negative "..."` | negative prompt; quote it if it contains spaces |
| `--transparent` | RGBA cutout, same prompt wrapping as `transparent=true` |
| `--image <path>` | an extra reference for `edit`, repeatable up to ten in total |

Flags may appear anywhere in the line and accept `--flag value` or `--flag=value`. Only
flag *values* are unquoted, so an apostrophe in the prompt is safe. Anything you leave
out falls back to the config (`width` / `height` / `steps`), exactly as it does for a
tool call. Runs of whitespace collapse, so the examples here can be typed on one line or
wrapped across several with `shift+enter` in the fixed TUI.

For `edit`, the **first word after the subcommand is the picture** — a path, a `~` path
or a bare name resolved against `out_dir`, the same three forms `image_edit` accepts.
Everything after it is the prompt.

Both subcommands are blocking and refuse to start the server themselves: if it is not
loaded you get `server not ready`, so run `/qwenimage start` first. Each is gated on the
matching entry in `tools`, so a config that exposes only `image_generate` also has no
`/qwenimage edit`.

### Stopping a server that is mid-render

`/qwenimage stop` asks the server to exit, but a render already in flight cannot be
interrupted — uvicorn finishes in-flight requests before shutting down, and a diffusion
step does not check for cancellation. So the server reports what it is up against:

```
> /qwenimage stop
qwen-image: a render is still in flight — the server cannot exit until it finishes,
so it will be killed in 20s. That render's image is lost either way; nothing else was
queued behind it.
```

After `SHUTDOWN_GRACE_S` (20s) the process hard-exits rather than lingering. That
matters more than it sounds: until it exits it keeps the weights on the GPU while no
longer answering on its port, and a `start` issued in that window used to launch a
*second* server onto the same card — at which point neither finished. The server now
writes a pid file next to its log (`server.log` → `server.pid`) and `start` refuses
while that pid is alive:

```
qwen-image: a qwen-image server (pid 26052) is still running and holding the GPU, even
though it is not answering on http://127.0.0.1:8770 — it is probably shutting down
behind a stuck render. Wait for it to exit, or kill that pid, before starting another
```

The pid file is derived from `log_file`, so a second config with its own log (the SDNQ
variant) gets its own pid file and the two servers never mistake each other for orphans.

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

> **Check your environment before blaming the config.** `HIP_VISIBLE_DEVICES` and
> `CUDA_VISIBLE_DEVICES` are inherited by the server, and a persistent one set in your
> user environment will hide the card from torch no matter what `device` says —
> `HIP_VISIBLE_DEVICES=-1` in particular means *no HIP devices at all*, and
> `--list-devices` then prints `Failed to get device count` and lists only `cpu`. The
> config's `hip_visible_devices` / `cuda_visible_devices` keys override the inherited
> value for the server process, which is the reliable way to pin a card on a machine
> where those variables are set for something else:
>
> ```json
> { "device": "cuda:0", "hip_visible_devices": "0" }
> ```
>
> An explicit `device` that torch cannot see is a hard error at startup, not a silent
> fall back to CPU — but `device: "auto"` *will* pick `cpu` when nothing else is
> visible, which looks like a working server rendering very slowly.

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

Size timeouts from the *total*, not from s/step: `request_timeout` must exceed the whole
render. The tools declare `timeout_s = request_timeout + 30` on their `ToolSpec`, so
aar's shared `tools.command_timeout` (default **300 s**) no longer clips them — without
that, the executor would cancel the tool call while the GPU kept working and the
finished image would be thrown away. If you are running an older aar that lacks
per-tool timeouts, raise the shared cap instead:

```jsonc
// ~/.aar/config.json
{ "tools": { "command_timeout": 1800 } }
```

### Keeping the model in VRAM

`offload: "model"` (the default) is diffusers' `enable_model_cpu_offload`: weights live
in system RAM and each component is moved onto the card as the pipeline reaches it, then
back. **VRAM reading near-zero between renders is that working, not a leak.**

On a PCIe x16 slot the shuffling is cheap. Over **Thunderbolt 3 to an eGPU** (~2.5 GB/s
against x16's ~25) it is not, and two separate costs appear:

| cost | fix |
|---|---|
| The server exiting and rebuilding the pipeline (~140 s) | `"idle_timeout": 0` — never exit. Safe with any offload mode, no downside beyond a resident process |
| Re-sending components every render | `offload` / `resident_components` — see below |

Measured on an RX 7900 XTX (24 GB), 1024x512, 30 steps, Q4_K_M:

| configuration | idle VRAM | render |
|---|---|---|
| `offload: "model"`, nothing pinned | 0.04 GB | **153 s** |
| `resident_components: ["transformer", "vae"]` | 5.03 GB | 209 s / 199 s |
| `resident_components: ["text_encoder"]` | 16.38 GB | 202 s / 206 s |
| `offload: "none"` | 21.4 GB | 87 s once, then 336 s/step — not reproducible |

**On a 24 GB card, plain eviction wins.** Everything pinned is headroom the activations
no longer have, and the spill crosses the same slow bus you were trying to avoid. With
`offload: "none"` the pipeline occupies 21.4 of 24 GB and the outcome flips on
fragmentation you cannot see — the same server that rendered in 87 s later took 336 s
*per step* for the same size, with no error, just a crawl.

One more trap specific to `resident_components`: pinning the transformer collapses
diffusers' `model_cpu_offload_seq` to a single entry, and a module is only evicted when
the *next* one in the chain runs. With nothing after it, the ~15 GB text encoder stays on
the card for the whole denoise — the opposite of what pinning was meant to achieve. The
server removes pinned names from the chain so the exclusion list is honoured at all, but
it cannot conjure headroom that is not there.

So: set `idle_timeout: 0`, keep `offload: "model"`, and reach for `resident_components`
only on a card whose resident footprint is under roughly half its VRAM. The two-GPU
setup this pairs with — renderer on one card, chat model on the other — is written up in
[`docs/sprite-sheet-workflow.md`](../../../docs/sprite-sheet-workflow.md#two-gpus-two-jobs).

**The remaining lever is size, not placement.** With Q4_K_M the transformer is ~4.5 GB —
the bulk of the 21.4 GB is the *unquantized text encoder*, which runs once per render and
then idles for 30 steps. GGUF does not help: those checkpoints quantize the transformer
only. See [SDNQ](#sdnq-a-fully-quantized-pipeline) for a checkpoint that quantizes
everything.

### SDNQ: a fully quantized pipeline

[SDNQ](https://github.com/Disty0/sdnq) (SD.Next Quantization) checkpoints are ordinary
diffusers pipelines with **every component quantized**, text encoder included — which is
the part GGUF leaves at bf16.

| checkpoint | text encoder | transformer | vae | total |
|---|---|---|---|---|
| `Qwen/Qwen-Image-2.1` + Q4_K_M GGUF | ~15 GB (bf16) | ~4.5 GB | ~1.3 GB | **~21.4 GB** |
| [`OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit`](https://huggingface.co/OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit) | 6.28 GB | 3.82 GB | 1.26 GB | **11.37 GB** |
| [`OzzyGT/Qwen_Image_2_1_sdnq_dynamic_8bit`](https://huggingface.co/OzzyGT/Qwen_Image_2_1_sdnq_dynamic_8bit) | 9.34 GB | 6.70 GB | 1.26 GB | **17.32 GB** |

At 11.37 GB the int4 pipeline leaves ~12 GB of a 24 GB card free for activations, which
is the headroom `offload: "none"` needs — and once it is resident, nothing crosses the
bus again. Measured on the same RX 7900 XTX, `offload: "none"`, `vae_tiling: true`,
10.87 GB resident and flat between renders:

| render | base + Q4_K_M, `offload: "model"` | SDNQ int4, `offload: "none"` |
|---|---|---|
| model load | 44 s | 64 s |
| 1024x512, 30 steps | 153 s | **25.6 s** (x2 runs, identical) |
| 1024x1024, 30 steps | thrashed — 286 s/step | **48.6 s** |
| 2048x1024, 20 steps | not attempted | **87.5 s** |

Six times faster at 1024x512, reproducible, and the size cliff is gone — 2 MP renders
that were impossible before now finish in under two minutes. Note this was measured
*without* the Triton fast path (see the `cl` warning below), so the memory win dominates
on its own.

> **But int4 degrades the alpha channel.** Transparent output is the one thing it is
> measurably worse at:
>
> | | base + Q4_K_M | SDNQ int4 |
> |---|---|---|
> | 1024x512 sprite sheet, fully transparent px | 25.1% | **5.2%** |
> | 768x768 sticker, fully transparent px | ~20% | **10.2%** |
>
> Instead of a clean cutout the int4 model returns a banded purple background — keyable
> in principle, but noisy enough that it is not a drop-in replacement. Frame uniformity
> across a sprite row was *better* than bf16 (21.0 / 21.0 / 21.0 / 20.9% opaque), so this
> is specifically an alpha problem, not a general quality loss.
>
> **So: use int4 for opaque work and large sizes, keep the bf16 + GGUF pipeline for
> transparent sprites.** The 8-bit SDNQ variant (17.32 GB) is the untested middle ground
> — it may keep the alpha and most of the speed, though 17.32 GB resident leaves much
> less room for activations.

Install the backend into the **server's** venv and point `model` at the repo. The
quantization is baked into the checkpoint, so `quant` stays `"none"` (that setting selects
a GGUF *transformer*, which is a different mechanism):

```bash
~/.aar/qwen-image/.venv/Scripts/python.exe -m pip install "sdnq>=0.2.2"
```

```json
{
  "model": "OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit",
  "quant": "none",
  "offload": "none",
  "vae_tiling": true,
  "idle_timeout": 0
}
```

The server imports `sdnq` before `from_pretrained` when it is installed — SDNQ weights
only deserialize once the backend has registered itself. Without the import the load
fails; with it, nothing else changes.

> **Check that `torch.compile` works first.** SDNQ's fast path
> (`use_quantized_matmul`) needs Triton / `torch.compile`. Without it the backend falls
> back to eager mode and dequantizes weights on every operation, which can be *slower*
> than the unquantized model despite using less memory. On Windows this shows up at
> import time as:
>
> ```
> SDNQ: Torch Compile test failed! Falling back to PyTorch Eager mode.
> Error message: RuntimeError: Compiler: cl is not found.
> ```
>
> `cl` is MSVC — install the Visual Studio Build Tools (C++ workload) and it goes away.
> Benchmark before and after; do not assume int4 is faster just because it is smaller.

#### Trying it side by side

`AAR_QWEN_IMAGE_CONFIG` points the extension at a different settings file, so an
alternative checkpoint can be benchmarked without touching the one that works. Put the
variant in its own file, on its own port and with its own log:

```jsonc
// ~/.aar/qwen-image-sdnq.json — everything else copied from qwen-image.json
{
  "url": "http://127.0.0.1:8771",
  "model": "OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit",
  "quant": "none",
  "offload": "none",
  "vae_tiling": true,
  "idle_timeout": 0,
  "log_file": "~/.aar/qwen-image/server-sdnq.log"
}
```

```bash
# only one server may hold the card at a time
curl -s -X POST http://127.0.0.1:8770/shutdown

AAR_QWEN_IMAGE_CONFIG=~/.aar/qwen-image-sdnq.json aar tui
> /qwenimage status        # check allocated VRAM against the table above
> /qwenimage generate a pixel-art dinosaur, transparent, 1024x512
```

Running `aar` without the variable goes back to the original config, so there is nothing
to revert if the alternative disappoints. Compare against the numbers in
[Keeping the model in VRAM](#keeping-the-model-in-vram), and pull the weights first —
a partial cache fails at load:

```bash
~/.aar/qwen-image/.venv/Scripts/python.exe -c   "from huggingface_hub import snapshot_download; print(snapshot_download('OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit'))"
```

See the [diffusers-recipes scripts](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2_1/README.md)
for the reference usage this is modelled on.

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

Those prompts assume a chat model that reliably turns "30 steps, seed 42, save as
neon.png" into tool arguments. Smaller local models frequently drop them and you get a
default-sized render under a timestamp name instead —
[When the model drops your parameters](#c-when-the-model-drops-your-parameters) covers
what to do about that.

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

Declare a profile and the coding agent can delegate to it with the built-in
[`spawn_agent`](../../../docs/configuration.md#sub-agents-spawn_agent) tool — a nested
agent whose entire tool surface is one image tool.

**Declare two, not one.** Generating and editing are different jobs and the prompt that
makes one reliable makes the other impossible: an illustrator told "call
`image_generate` exactly once" will never edit anything, and it is the *description*
that the parent model reads when it picks an agent.

```json
{
  "subagents": {
    "enabled": true,
    "max_depth": 1,
    "agents": {
      "illustrator": {
        "description": "Generates one image from a description and returns the saved file path",
        "tools": [],
        "extension_tools": ["image_generate"],
        "system_prompt": "You are an image generator. Call image_generate exactly once, passing BOTH width and height explicitly, then reply with only the saved file path. Never explain.",
        "max_steps": 6,
        "timeout": 1800
      },
      "retoucher": {
        "description": "Edits an existing picture and returns the path of the NEW file (the original is never modified). Give it the full path of the image to change, exactly what to change, and the output size in pixels - an edit does not keep the original's aspect ratio unless you ask for it. Pass a seed when you want two attempts to be comparable.",
        "tools": [],
        "extension_tools": ["image_edit"],
        "system_prompt": "You edit existing pictures. Call image_edit exactly once: put the file path you were given in 'images', and pass BOTH width and height explicitly - never only one, or the other falls back to a default and silently changes the aspect ratio. Pass 'seed' and 'out' when you were given them. Describe only the change, not the whole picture. Then reply with only the saved file path. Never explain.",
        "max_steps": 6,
        "timeout": 1800
      }
    }
  }
}
```

Three keys do the real work:

- **`tools: []`** strips `read_file` / `bash` / everything else, so the child cannot read
  your files, write anything but a PNG, or run a command.
- **`extension_tools`** is the one people miss. An installed extension registers into
  *every* agent in the process, sub-agents included, so without this list the child
  inherits `image_generate`, `image_edit` and whatever other extensions you have. Naming
  a single tool is also what makes the split above *structural* rather than a matter of
  the child obeying its prompt: the retoucher has no `image_generate` to call by mistake.
- **`timeout`** becomes the `spawn_agent` tool's own `timeout_s`, so a slow render is not
  cut short by `tools.command_timeout`. Keep it above the sidecar's `request_timeout`.

"Passing BOTH width and height" is not padding: small models routinely pass `width` and
let `height` fall back to the config default, which silently gives you a 512x1024 image
when you asked for 512x512, or squares up a photo you were editing.

Leaving `provider` off the profile makes the child reuse the parent's model. A *smaller*
model per profile saves context, but on a single-GPU box it also makes Ollama swap models
in and out of the same VRAM the renderer wants — worth measuring before assuming it is
cheaper.

```
> Use spawn_agent with the illustrator for a pixel-art green dinosaur running,
  transparent, 1024x512, seed 2024, saved as player.png — then build the game
  around whatever path it reports.

> Use spawn_agent with the retoucher on B:/photos/old.png — add natural realistic
  colour and restore detail, 1600x960, seed 42, saved as old-colour.png.
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

### C. When the model drops your parameters

`width`, `height`, `steps`, `seed` and `out` are *optional* tool arguments, and a small
local chat model will often ignore them — you write "1600x960, 30 steps, seed 42, save as
`foo.png`" in your message and the render comes back at the config default size under a
timestamp name. This is not a prompt-wording problem you can reliably fix by rephrasing:
the parameters have to come from somewhere the model is not involved in.

There are three such places, in increasing order of how much you give up.

**1. The slash-command.** `/qwenimage generate` and
[`/qwenimage edit`](#rendering-from-the-command-not-through-the-model) take the
parameters as flags and call the sidecar directly, so nothing is inferred from prose:

```
/qwenimage edit old-photo.png --size 1600x960 --steps 30 --seed 42 --out colour.png
  add natural realistic colour, restore detail, clean up noise and compression artefacts
```

This is exact and repeatable — change the prompt, keep `--seed 42`, and the two renders
are comparable. What you give up is the agent: the command does one render and returns a
path, it does not then write the code that uses the picture. Reach for it when you are
iterating on an image, and for the tools when the image is a step in a larger task.

**2. A config file, so the tool call needs fewer arguments.** The config can carry some
of them, but not all:

| argument | preconfigurable? |
|---|---|
| `width` / `height` / `steps` | yes — `width` / `height` / `steps` |
| guidance | yes — `true_cfg_scale` |
| output directory | yes — `out_dir` |
| `seed` | **no** — random unless the model or `--seed` supplies it |
| `out` (file name) | **no** — defaults to a timestamp |
| `negative_prompt` | **no** |

Don't move your global defaults for one task — copy them and point
`AAR_QWEN_IMAGE_CONFIG` at the copy:

```powershell
Copy-Item $HOME\.aar\qwen-image.json $HOME\.aar\qwen-image-restore.json
# edit width/height/steps in the copy, e.g. 1600 / 960 / 30
$env:AAR_QWEN_IMAGE_CONFIG = "$HOME\.aar\qwen-image-restore.json"
aar tui
```

Now the message only has to name the tool and the file, which even a small model gets
right, and you keep the agent:

```
> Use image_edit on old-photo.png: add natural realistic colour, restore detail,
  clean up noise and compression artefacts. Keep the composition unchanged.
```

**3. A system prompt.** A `subagents` profile (section B) whose `system_prompt` reads
*"always call `image_edit` with width=1600, height=960, steps=30"* puts the parameters
somewhere that does not compete with the rest of the user's message. Instructions in the
system prompt survive a long conversation; the same sentence typed into a chat turn does
not. Still a model deciding, so still not a guarantee — but a much better bet than prose.

Below all three, the sidecar's HTTP API is always there if you want a script instead:

```python
import base64, httpx, pathlib

src = pathlib.Path("old-photo.png")
payload = {
    "prompt": "add natural realistic colour to the whole photograph and restore it",
    "negative_prompt": "oversaturated, cartoon, watermark, text",
    "images": [base64.b64encode(src.read_bytes()).decode()],
    "width": 1600, "height": 960, "steps": 30, "seed": 42,
}
r = httpx.post("http://127.0.0.1:8770/edit", json=payload, timeout=900)
r.raise_for_status()
pathlib.Path("old-photo-colour.png").write_bytes(base64.b64decode(r.json()["images"][0]))
```

`POST /generate` takes the same body without `images`.

> **Match the aspect ratio when editing.** `image_edit` falls back to the config
> `width` / `height` just like `image_generate` does, so editing an 800x480 photo under
> the default 1024x1024 silently re-renders it square. Either set the ratio in the config
> you are using for that job, or pass `--size 1600x960`.

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
| `width` / `height` / `steps` | `1024` / `1024` / `30` | fallback for both tools when the model omits those arguments — the model card's example uses 2048px and 40 steps. There is no config key for `seed`, `out` or `negative_prompt`; see [When the model drops your parameters](#c-when-the-model-drops-your-parameters) |
| `max_pixels` | `4300800` (2400x1792) | requests above this are refused before reaching the GPU; covers every documented aspect ratio |
| `idle_timeout` | `900` | server exits after this many idle seconds (**0 = never** — set this on a slow bus / eGPU, where rebuilding the pipeline costs ~2 minutes) |
| `resident_components` | `[]` | Pipeline components pinned to the GPU instead of being offloaded, e.g. `["transformer", "vae"]`. Only meaningful with `offload` `model`/`sequential`. Pays off when the card has headroom to spare; on a 24 GB card with this pipeline it is slower than plain eviction |
| `vae_tiling` / `vae_slicing` / `attention_slicing` | `false` | Opt-in activation-memory reducers, applied after placement. They lower the per-step peak rather than the weight footprint, so they matter with `offload: "none"`. Best-effort: a helper this pipeline lacks logs a warning instead of failing to start |
| `request_timeout` | `900` | a large image on an offloaded GPU takes minutes |
| `evict_ollama` | `""` | Base URL of an Ollama instance sharing this GPU. When set, every model loaded there is unloaded before the server starts and before each render — see [Sharing one GPU with Ollama](#sharing-one-gpu-with-ollama). Empty disables it |
| `evict_timeout` | `60` | Seconds to wait for that VRAM to actually come back; Ollama answers the unload before its runner exits |
| `tools` | all | subset of tools to expose to the model |

Environment overrides: `AAR_QWEN_IMAGE_CONFIG` (config file path), `AAR_QWEN_IMAGE_URL`.
`AAR_QWEN_IMAGE_CONFIG` is the clean way to keep one set of render defaults per job —
sprite sizes for a game repo, photo ratios for a restoration pass — without editing the
global file.

## Sharing one GPU with Ollama

The renderer and your chat model both want the biggest card, and on a 24 GB card
they do not both fit: a 16 GiB chat model plus a diffusion pipeline is over the
line before the first step. Worse, `OLLAMA_KEEP_ALIVE=-1` means the chat model
never yields — and the slash-commands never call the model, so nothing triggers
an unload on its own.

`evict_ollama` closes that gap. Point it at the Ollama instance that shares the
card:

```json
{
  "evict_ollama": "http://127.0.0.1:11435",
  "evict_timeout": 60
}
```

Before starting the sidecar and before every render, the extension asks that
instance for its loaded models (`/api/ps`) and unloads each one
(`keep_alive: 0`), then waits for them to actually go. Measured on a 7900 XTX,
releasing a 16.09 GiB `qwen3.8` took **2.6s**; Ollama reloads it by itself on the
next prompt, so the only cost is that reload.

It is deliberately best-effort: an Ollama that is not running, is on a different
GPU, or simply does not answer never blocks a render. And the wait is bounded by
`evict_timeout`, so a model that refuses to go delays a render by at most that
long rather than hanging it.

Two things to get right alongside it:

- **The sidecar has to hand the card back too.** `offload: "none"` keeps the whole
  pipeline resident, so with `idle_timeout: 0` the chat model can never reload.
  Either give `idle_timeout` a finite value (60s survives a batch of renders,
  since the gap between them is only the agent thinking) or use
  `offload: "model"`, which idles near zero VRAM.
- **Pin image sub-agents to the *other* model.** A `spawn_agent` profile that
  inherits the big chat model keeps it loaded on the card for the whole render,
  which puts you back where you started. Give those profiles a `provider` whose
  `base_url` is the small model's instance.

## Notes

- **Server lifetime.** The server is started detached, so it survives the aar process
  and later `aar run` calls reuse the loaded pipeline. It stops after `idle_timeout`
  or `/qwenimage stop`. Its stdin/stdout are never inherited, which keeps `aar acp`
  (JSON-RPC over stdio) safe. A stop that cannot complete gracefully escalates to a
  hard exit after 20s, and a pid file stops a second server being started on top of one
  that is still holding the card — see
  [Stopping a server that is mid-render](#stopping-a-server-that-is-mid-render).
- **A client timeout does not cancel the render.** If `request_timeout` expires the tool
  returns, but the GPU keeps working and the finished image is discarded, because the
  result only exists in the HTTP response. Size `request_timeout` from your slowest
  render, not your average one.
- **Safety.** The tools only talk to the configured loopback URL and only write inside
  `out_dir` (`side_effects: network, write`; `image_edit` also reads the reference
  images). The server command comes from your config, never from the model.
  `image_edit`'s `images` argument is annotated `format: "path"`, so aar's policy engine
  applies `allowed_paths` / `denied_paths` to **every** reference — a picture outside the
  allowed paths is denied before the sidecar is called, exactly as it would be for
  `read_file`. That check is on the *tool*, so it governs the model and any sub-agent;
  `/qwenimage edit` is you typing a path yourself and is not policy-checked. If a render
  is refused, either start aar where the picture lives (which `out_dir: ""` wants anyway)
  or widen `safety.allowed_paths`.
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
