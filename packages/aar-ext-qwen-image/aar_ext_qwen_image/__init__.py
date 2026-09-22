"""Aar extension: qwen-image — local text-to-image and image editing tools.

Backed by `Qwen/Qwen-Image-2.1 <https://huggingface.co/Qwen/Qwen-Image-2.1>`_, a
7B diffusion model for generation *and* editing (transparent RGBA output, up to
ten reference images).  It is not a chat model, so it runs next to the agent's
normal provider as a small local HTTP server (``server.py``) in a dedicated
Python environment with ``torch`` + ``diffusers`` — aar itself never imports the
heavy ML stack.

Tools exposed to the model:

- ``image_generate`` — text to image, saved as a PNG under ``out_dir``
  (aar's current working directory unless ``out_dir`` says otherwise)
- ``image_edit``     — edit / combine up to ten reference images

Slash-command ``/qwenimage [status|devices|quant|start|stop|generate <prompt>|
edit <image> <prompt>]``.  ``generate`` and ``edit`` accept ``--size``/``--width``/
``--height``/``--steps``/``--seed``/``--out``/``--negative``/``--transparent`` flags,
which is the way to pin render parameters that the chat model keeps dropping.

Configuration is read from ``~/.aar/qwen-image.json`` (all keys optional)::

    {
      "url": "http://127.0.0.1:8770",
      "autostart": "on_demand",          // "on_demand" | "session" | "off"
      "python": "~/.aar/qwen-image/.venv/Scripts/python.exe",
      "device": "auto",                  // or "cuda:0" / "xpu" / "dml:0" / "cpu"
      "offload": "model",                // "none" | "model" | "sequential"
      "quant": "none",                   // "none" | "Q8_0" | "Q6_K" | "Q5_K_M" | "Q4_K_M" | "Q4_0"
      "out_dir": "images",              // relative -> under aar's cwd; "" = cwd itself
      "tools": ["image_generate", "image_edit"]
    }

``quant`` swaps the 13.3 GiB bf16 transformer for a GGUF quantization of the
same Qwen-Image-2.1 weights (3.8-7.1 GiB), which is what makes the model
comfortable on a 24 GB card.  The text encoder and VAE are untouched and still
come from the base checkpoint.  Switch at runtime with ``/qwenimage quant
<name>``; it takes effect on the next server start.

The server may also live behind a *launcher* — an argv prefix such as
``["wsl.exe", "-d", "Ubuntu", "--"]`` — which is how an AMD GPU is reached from
Windows: torch only speaks to Radeon cards through ROCm, and ROCm runs in WSL2.
Then ``python`` and ``server_script`` are paths *inside* the launcher's world.

``AAR_QWEN_IMAGE_CONFIG`` points at a different config file and
``AAR_QWEN_IMAGE_URL`` overrides ``url``.

Standalone usage — copy the ``aar_ext_qwen_image`` directory into
``~/.aar/extensions/`` and it will be loaded automatically.

The tools return a *file path*: aar tool results are plain strings, so the model
never sees pixels.  To let it look at what it made, attach the file back with
``@path/to/image.png`` on a vision-capable provider.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

SERVER_SCRIPT = Path(__file__).with_name("server.py")
ALL_TOOLS = ("image_generate", "image_edit")

# GGUF quantizations of the *transformer* only, largest first.  Sizes are the
# files in GGUF_REPO in GiB, to match how VRAM is counted everywhere else here —
# the model card lists the same files in decimal GB, so its numbers run ~7%
# higher.  They replace a 13.3 GiB bf16 transformer, so the saving is 6-9 GiB of
# VRAM (and of host memory under `offload: "model"`).  The text encoder and VAE
# are unquantized and still come from the base checkpoint.
GGUF_QUANTS: dict[str, tuple[float, str]] = {
    "Q8_0": (7.1, "closest to bf16"),
    "Q6_K": (5.5, ""),
    "Q5_K_M": (4.9, ""),
    "Q4_K_M": (4.3, "recommended"),
    "Q4_0": (3.8, "smallest"),
}
GGUF_REPO = "abenzerps/Qwen-Image-2.1-GGUF"
BF16_TRANSFORMER_GIB = 13.3
MAX_REF_IMAGES = 10  # Qwen-Image-2.1 accepts up to ten reference images
MAX_INPUT_BYTES = 24 * 1024 * 1024  # per reference image
INPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Qwen-Image-2.1 generates RGBA natively, but the request is made in the *prompt*
# text — there is no pipeline argument for it.  This is the wording from the model
# card; straying from it noticeably weakens the effect.
RGBA_PROMPT_PREFIX = "This is an RGBA image with transparency. "
RGBA_PROMPT_SUFFIX = " The image has alpha channel and the background is transparent."

# Qwen-Image-2.1's documented aspect ratios, at the sizes the model card uses.
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}


def config_path() -> Path:
    """Where the extension's settings live (``AAR_QWEN_IMAGE_CONFIG`` wins)."""
    env_path = os.environ.get("AAR_QWEN_IMAGE_CONFIG")
    return Path(env_path) if env_path else Path.home() / ".aar" / "qwen-image.json"


def write_config_key(path: Path, key: str, value: Any) -> None:
    """Persist one setting into the JSON config, leaving every other key alone."""
    raw: dict[str, Any] = {}
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
    raw[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def normalise_quant(name: str) -> str | None:
    """Map a user-typed quantization label onto a config value, or ``None``.

    Accepts the labels as the model card writes them (``Q4_K_M``) in any case
    and with dashes, plus the obvious ways of asking for the unquantized
    weights.
    """
    key = name.strip().upper().replace("-", "_")
    if key in ("", "NONE", "OFF", "FULL", "BF16"):
        return "none"
    return key if key in GGUF_QUANTS else None


def describe_quant(quant: str) -> str:
    """One-line description of a quantization setting, for status output."""
    if quant == "none":
        return f"none (full bf16 transformer, {BF16_TRANSFORMER_GIB} GiB)"
    if quant not in GGUF_QUANTS:
        return quant
    size, note = GGUF_QUANTS[quant]
    return f"{quant} (GGUF, {size} GiB{', ' + note if note else ''})"


def _default_python() -> str:
    venv = Path.home() / ".aar" / "qwen-image" / ".venv"
    return str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QwenImageConfig:
    """Extension settings, loaded from ``~/.aar/qwen-image.json``."""

    url: str = "http://127.0.0.1:8770"
    autostart: str = "on_demand"  # "on_demand" | "session" | "off"
    python: str = field(default_factory=_default_python)
    model: str = "Qwen/Qwen-Image-2.1"
    device: str = "auto"  # "auto" | "cuda:N" (NVIDIA or AMD/ROCm) | "xpu" | "mps" | "dml:N" | "cpu"
    dtype: str = "bfloat16"
    offload: str = "model"  # "none" | "model" | "sequential"
    # Activation-memory reducers. They matter with ``offload="none"``, where the
    # weights hold most of the card and the per-step activations decide whether a
    # larger image fits or spills into host memory.
    vae_tiling: bool = False
    vae_slicing: bool = False
    attention_slicing: bool = False
    # Components pinned to the GPU while the rest stay CPU-offloaded. The
    # transformer runs once per denoising *step*, the text encoder once per
    # *render* — so on a slow bus (eGPU over Thunderbolt) pinning the hot
    # components and letting the cold ones travel beats moving everything.
    # Only meaningful with offload "model" or "sequential".
    resident_components: list[str] = field(default_factory=list)
    quant: str = "none"  # "none" (full bf16) or a key of GGUF_QUANTS
    quant_repo: str = GGUF_REPO
    quant_file: str | None = None  # local .gguf path, or a file name inside quant_repo
    cuda_visible_devices: str | None = None
    hip_visible_devices: str | None = None  # ROCm's equivalent, for multi-GPU AMD boxes
    launcher: list[str] = field(default_factory=list)  # argv prefix, e.g. ["wsl.exe", "-d", "roc"]
    server_script: str | None = None  # path to server.py *as the launcher sees it*
    # Empty means aar's current working directory; a relative path is taken
    # relative to it, so "images" writes into ./images next to whatever the
    # user is working on.  Absolute and ``~``-paths are used as given.
    out_dir: str = ""
    width: int = 1024  # the model card's example uses 2048; 1024 is kinder to small GPUs
    height: int = 1024
    steps: int = 30  # model card default is 40
    true_cfg_scale: float = 4.0
    idle_timeout: float = 900.0
    startup_timeout: float = 600.0  # first run downloads ~20 GB of weights
    request_timeout: float = 900.0  # a 2048px image on an offloaded 6 GB GPU takes minutes
    # The model card's largest recommended aspect ratio is 4:3 at 2400x1792.
    # A 2048*2048 cap silently rejected every non-square documented size.
    max_pixels: int = 2400 * 1792
    log_file: str = "~/.aar/qwen-image/server.log"
    # Base URL of an Ollama instance that shares this GPU.  Empty disables the
    # whole mechanism.  When set, every loaded model there is unloaded before a
    # render, because a chat model pinned by ``OLLAMA_KEEP_ALIVE=-1`` does not
    # yield the card on its own and the renderer then has nowhere to put the
    # weights.  See ``evict_timeout``.
    evict_ollama: str = ""
    # Seconds to wait for the VRAM to actually come back after asking.  Ollama
    # answers the unload request before its runner process has exited.
    evict_timeout: float = 60.0
    tools: list[str] = field(default_factory=lambda: list(ALL_TOOLS))

    @classmethod
    def load(cls, path: Path | None = None) -> QwenImageConfig:
        path = path or config_path()
        raw: dict[str, Any] = {}
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown qwen-image config keys in {path}: {', '.join(unknown)}")
        cfg = cls(**raw)
        if url := os.environ.get("AAR_QWEN_IMAGE_URL"):
            cfg.url = url
        if cfg.autostart not in ("on_demand", "session", "off"):
            raise ValueError(f"autostart must be on_demand, session or off, got {cfg.autostart!r}")
        if cfg.offload not in ("none", "model", "sequential"):
            raise ValueError(f"offload must be none, model or sequential, got {cfg.offload!r}")
        if cfg.quant not in ("none", *GGUF_QUANTS):
            raise ValueError(
                f"quant must be none or one of {', '.join(GGUF_QUANTS)}, got {cfg.quant!r}"
            )
        return cfg

    @property
    def python_path(self) -> Path:
        return Path(self.python).expanduser()

    @property
    def log_path(self) -> Path:
        return Path(self.log_file).expanduser()

    @property
    def pid_path(self) -> Path:
        """Where the server records its process id while running.

        Derived from ``log_file`` rather than being its own setting, so a config
        that redirects the log (e.g. the SDNQ variant) automatically gets its own
        pidfile and the two servers do not mistake each other for orphans.
        """
        return self.log_path.with_name(self.log_path.stem + ".pid")

    @property
    def out_path(self) -> Path:
        """Absolute directory the rendered PNGs land in.

        Anchored on the *current* working directory rather than resolved once at
        load time: aar can be started anywhere, and an unconfigured ``out_dir``
        should follow the project the user is in.
        """
        raw = self.out_dir.strip()
        if not raw:
            return Path.cwd()
        path = Path(raw).expanduser()
        return path if path.is_absolute() else Path.cwd() / path


# ---------------------------------------------------------------------------
# Server management + HTTP client
# ---------------------------------------------------------------------------


class QwenImageUnavailable(RuntimeError):
    """The image server is not reachable and could not be started."""


class QwenImageError(RuntimeError):
    """The server was reached but refused or failed the render.

    Carries the server's own ``detail`` string, which ``httpx.HTTPStatusError``
    would otherwise drop.
    """


class QwenImageClient:
    """Talks to the Qwen-Image server and starts it on demand.

    *transport* / *async_transport* exist for tests (``httpx.MockTransport``).
    """

    def __init__(
        self,
        config: QwenImageConfig,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._async_transport = async_transport
        self._proc: subprocess.Popen[bytes] | None = None
        self._start_lock = asyncio.Lock()

    # -- sync (slash-commands) ------------------------------------------------

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(base_url=self.config.url, timeout=timeout, transport=self._transport)

    def health(self) -> dict[str, Any] | None:
        """Return the server's /health payload, or ``None`` when unreachable."""
        try:
            with self._client(2.0) as c:
                r = c.get("/health")
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError:
            return None

    def shutdown(self) -> dict[str, Any] | None:
        """Ask the server to exit. Returns its reply, or ``None`` if unreachable.

        The reply carries ``busy`` — whether a render is in flight — because a
        graceful shutdown cannot complete until that render does, and the caller
        deserves to know it is waiting on a kill rather than a clean exit.
        """
        try:
            with self._client(5.0) as c:
                r = c.post("/shutdown")
                r.raise_for_status()
                body = r.json()
        except (httpx.HTTPError, ValueError):
            return None
        return body if isinstance(body, dict) else {}

    def evict_ollama(self) -> list[str]:
        """Unload every model held by the configured Ollama, freeing its VRAM.

        Returns the names it asked to unload (empty when the feature is off,
        nothing is loaded, or Ollama is unreachable).

        This exists because the two halves of a local setup compete for one
        card: a chat model with ``OLLAMA_KEEP_ALIVE=-1`` holds its weights
        indefinitely, and a 16 GiB model plus a diffusion pipeline do not fit in
        24 GiB.  Ollama reloads the model by itself on the next prompt, so the
        only cost is that reload.

        Failures are deliberately soft: if Ollama is not running, or is on a
        different GPU entirely, rendering should still go ahead.
        """
        base = self.config.evict_ollama.rstrip("/")
        if not base:
            return []
        try:
            with httpx.Client(base_url=base, timeout=10.0, transport=self._transport) as c:
                loaded = [
                    m["name"] for m in c.get("/api/ps").json().get("models", []) if m.get("name")
                ]
                for name in loaded:
                    # keep_alive 0 is Ollama's "unload now"; an empty prompt
                    # means it never runs the model, it only drops it.
                    c.post("/api/generate", json={"model": name, "keep_alive": 0})
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            logger.debug("qwen-image: could not reach Ollama at %s to evict", base, exc_info=True)
            return []
        if not loaded:
            return []
        # The POST returns before the runner has exited, so wait for the VRAM.
        deadline = time.monotonic() + self.config.evict_timeout
        while time.monotonic() < deadline:
            try:
                with httpx.Client(base_url=base, timeout=10.0, transport=self._transport) as c:
                    if not c.get("/api/ps").json().get("models"):
                        break
            except (httpx.HTTPError, ValueError):
                break
            time.sleep(1.0)
        return loaded

    def render_sync(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Blocking POST to ``/generate`` or ``/edit`` — the slash-command path.

        Unlike :meth:`render` this never starts the server; the command checks
        ``/health`` first so it can tell the user to start it themselves.
        """
        self.evict_ollama()
        with self._client(self.config.request_timeout) as c:
            r = c.post(endpoint, json=payload)
            _raise_for_status(r)
            return r.json()

    def generate_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.render_sync("/generate", payload)

    # -- async (tools) --------------------------------------------------------

    def _async_client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.config.url, timeout=timeout, transport=self._async_transport
        )

    async def ahealth(self) -> dict[str, Any] | None:
        try:
            async with self._async_client(2.0) as c:
                r = await c.get("/health")
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError:
            return None

    async def ensure_ready(self, allow_start: bool = True) -> dict[str, Any]:
        """Wait until the server reports ``ready``, starting it if allowed."""
        async with self._start_lock:
            state = await self.ahealth()
            if state is None:
                if not allow_start or self.config.autostart == "off":
                    raise QwenImageUnavailable(
                        f"qwen-image server is not running at {self.config.url} "
                        "(autostart is off — run /qwenimage start)"
                    )
                await asyncio.to_thread(self.evict_ollama)
                self.start()
            deadline = time.monotonic() + self.config.startup_timeout
            while True:
                state = await self.ahealth()
                if state is not None and state.get("status") == "ready":
                    return state
                if state is not None and state.get("status") == "error":
                    raise QwenImageUnavailable(f"qwen-image failed to load: {state.get('error')}")
                if state is None and self._proc is not None and self._proc.poll() is not None:
                    raise QwenImageUnavailable(
                        f"qwen-image server exited with code {self._proc.returncode}; "
                        f"see {self.config.log_path}\n{_log_tail(self.config.log_path)}"
                    )
                if time.monotonic() > deadline:
                    raise QwenImageUnavailable(
                        f"qwen-image is still loading after {self.config.startup_timeout:.0f}s; "
                        f"try again shortly (log: {self.config.log_path})"
                    )
                await asyncio.sleep(2.0)

    async def render(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to ``/generate`` or ``/edit``, starting the server if needed."""
        await self.ensure_ready()
        # Blocking HTTP + a poll loop, so keep it off the event loop.
        await asyncio.to_thread(self.evict_ollama)
        async with self._async_client(self.config.request_timeout) as c:
            r = await c.post(endpoint, json=payload)
            _raise_for_status(r)
            return r.json()

    # -- process --------------------------------------------------------------

    def server_command(self) -> list[str]:
        cfg = self.config
        parts = urlsplit(cfg.url)
        cmd = [
            *cfg.launcher,
            cfg.python if cfg.launcher else str(cfg.python_path),
            cfg.server_script or str(SERVER_SCRIPT),
            "--host", parts.hostname or "127.0.0.1",
            "--port", str(parts.port or 80),
            "--model", cfg.model,
            "--device", cfg.device,
            "--dtype", cfg.dtype,
            "--offload", cfg.offload,
            "--max-pixels", str(cfg.max_pixels),
            "--max-images", str(MAX_REF_IMAGES),
            "--pid-file", str(cfg.pid_path),
            "--idle-timeout", str(cfg.idle_timeout),
            "--quant", cfg.quant,
            "--quant-repo", cfg.quant_repo,
        ]  # fmt: skip
        if cfg.quant_file:
            cmd += ["--quant-file", cfg.quant_file]
        if cfg.resident_components:
            cmd += ["--resident", ",".join(cfg.resident_components)]
        if cfg.vae_tiling:
            cmd.append("--vae-tiling")
        if cfg.vae_slicing:
            cmd.append("--vae-slicing")
        if cfg.attention_slicing:
            cmd.append("--attention-slicing")
        return cmd

    def start(self) -> None:
        """Launch the server detached, so it outlives this aar process.

        The command is fixed by the user's config — nothing model-controlled is
        executed.  The server frees its VRAM itself after ``idle_timeout``.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        cfg = self.config
        # With a launcher the interpreter lives on the other side of it (e.g. inside
        # WSL2), so only a plain local path can be checked here.
        if not cfg.launcher and not cfg.python_path.is_file():
            raise QwenImageUnavailable(
                f"qwen-image server interpreter not found: {cfg.python_path} — "
                "create the venv (see the aar-ext-qwen-image README) or set 'python' "
                "in ~/.aar/qwen-image.json"
            )
        # The port being free does not mean the GPU is.  uvicorn closes its
        # listening socket as soon as a graceful shutdown starts, but a wedged
        # render keeps the process — and the weights on the card — alive well
        # past that.  Starting a second server then quietly halves the VRAM both
        # have, and neither finishes.
        if (orphan := read_pid_file(cfg.pid_path)) is not None:
            raise QwenImageUnavailable(
                f"a qwen-image server (pid {orphan}) is still running and holding the GPU, "
                "even though it is not answering on "
                f"{cfg.url} — it is probably shutting down behind a stuck render. "
                "Wait for it to exit, or kill that pid, before starting another "
                f"(pid file: {cfg.pid_path})"
            )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if cfg.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
        if cfg.hip_visible_devices is not None:
            env["HIP_VISIBLE_DEVICES"] = cfg.hip_visible_devices
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.log_path.open("ab") as log:
            # stdin/stdout must never be inherited: `aar acp` speaks JSON-RPC over stdio
            kwargs: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": log,
                "stderr": subprocess.STDOUT,
                "env": env,
            }
            if os.name == "nt":
                flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                try:
                    self._proc = subprocess.Popen(
                        self.server_command(),
                        creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                        **kwargs,
                    )
                except OSError:  # parent job forbids breakaway
                    self._proc = subprocess.Popen(
                        self.server_command(), creationflags=flags, **kwargs
                    )
            else:
                self._proc = subprocess.Popen(
                    self.server_command(), start_new_session=True, **kwargs
                )


def _error_detail(r: httpx.Response) -> str:
    """Best-effort extraction of the server's error message.

    ``httpx.HTTPStatusError`` only carries the status line, so without this the
    ``detail`` FastAPI puts in the body — the only description of *why* a render
    failed — never reaches the model or the user.
    """
    try:
        payload = r.json()
    except ValueError:
        return r.text.strip()
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        return detail if isinstance(detail, str) else json.dumps(detail)
    return str(payload)


def _raise_for_status(r: httpx.Response) -> None:
    if r.status_code == 503:
        raise QwenImageUnavailable(f"qwen-image not ready: {_error_detail(r)}")
    if r.is_error:
        raise QwenImageError(f"server returned {r.status_code}: {_error_detail(r)}")


def _describe_failure(exc: BaseException, cfg: QwenImageConfig) -> str:
    """Describe a failed render in a way that is actually actionable.

    ``httpx.ReadError`` and friends stringify to ``""``, so a bare ``{exc}``
    produced the useless message ``qwen-image unavailable:``.  A transport error
    mid-render almost always means the server process died — on an offloaded GPU
    usually because a host allocation failed — and the only trace of that is in
    the server log, so point at it and quote the tail.
    """
    detail = str(exc).strip() or type(exc).__name__
    if not isinstance(exc, httpx.TransportError):
        return detail
    tail = _log_tail(cfg.log_path, lines=6)
    hint = (
        f"{detail} — the server closed the connection mid-render, so it most "
        f"likely died. Check {cfg.log_path}"
    )
    return f"{hint}:\n{tail}" if tail else hint


def _log_tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Files in / files out
#
# Both directions are model-controlled, so both are constrained here rather
# than trusted: outputs may only land on a plain file name inside ``out_dir``,
# and inputs must be existing image files of a bounded size.
# ---------------------------------------------------------------------------


def apply_rgba_prompt(prompt: str) -> str:
    """Wrap *prompt* in the model card's transparency phrasing.

    Qwen-Image-2.1 has no ``transparent`` pipeline argument — asking for RGBA is
    done in the prompt, and the model card's exact wording works markedly better
    than a paraphrase.  Applied only once: a prompt that already asks for RGBA is
    returned unchanged.
    """
    if "rgba image with transparency" in prompt.lower():
        return prompt
    return f"{RGBA_PROMPT_PREFIX}{prompt.strip()}{RGBA_PROMPT_SUFFIX}"


def resolve_out_path(out_dir: Path, name: str | None) -> Path:
    """Return a fresh PNG path inside *out_dir* for the model-supplied *name*.

    Only a bare file name is accepted — no separators, no ``..``, no absolute
    paths — and an existing file is never overwritten.
    """
    if name:
        candidate = name.strip()
        if not _SAFE_NAME.match(candidate) or "/" in candidate or "\\" in candidate:
            raise ValueError(
                f"invalid file name {name!r}: use a plain name like 'sunset.png' "
                "(no directories, it is always saved under the configured out_dir)"
            )
        stem = Path(candidate).stem
    else:
        stem = time.strftime("qwen-%Y%m%d-%H%M%S")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    n = 1
    while path.exists():
        path = out_dir / f"{stem}-{n}.png"
        n += 1
    return path


def pid_alive(pid: int) -> bool:
    """True if a process with *pid* currently exists.

    Deliberately does not use ``os.kill(pid, 0)``: on Windows that maps onto
    ``TerminateProcess``, so the "check" would kill the very server we are
    probing for.  ``OpenProcess`` + a zero-length wait is the safe equivalent.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        SYNCHRONIZE = 0x00100000
        WAIT_TIMEOUT = 0x00000102
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def read_pid_file(path: Path) -> int | None:
    """Return the pid recorded in *path*, or ``None`` if it is absent or stale."""
    try:
        pid = int(json.loads(path.read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return pid if pid_alive(pid) else None


def read_input_image(raw: str, out_dir: Path | None = None) -> str:
    """Read an image file from disk and return it base64-encoded.

    Accepts the two forms a model realistically produces besides a full path:

    * ``@path`` — aar's own attachment syntax.  When the user writes
      ``@~/pics/a.png`` the model tends to copy the ``@`` through verbatim.
    * a bare file name — ``image_generate`` saves to *out_dir* under a bare
      name, so the model refers back to its own output the same way.
    """
    candidate = raw.strip()
    if candidate.startswith("@"):
        candidate = candidate[1:].strip()
    if not candidate:
        raise ValueError(f"empty reference image path: {raw!r}")

    path = Path(candidate).expanduser()
    if not path.is_file() and out_dir is not None and Path(candidate).name == candidate:
        in_out_dir = out_dir / candidate
        if in_out_dir.is_file():
            path = in_out_dir
    if not path.is_file():
        raise ValueError(f"reference image not found: {path}")
    if path.suffix.lower() not in INPUT_SUFFIXES:
        raise ValueError(
            f"unsupported reference image {path.name!r}: "
            f"expected one of {', '.join(sorted(INPUT_SUFFIXES))}"
        )
    size = path.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise ValueError(
            f"reference image {path.name!r} is {size / 2**20:.1f} MB, "
            f"over the {MAX_INPUT_BYTES / 2**20:.0f} MB limit"
        )
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Slash-command flag parsing
# ---------------------------------------------------------------------------

_BOOL_FLAGS = {"transparent"}
_INT_FLAGS = {"width", "height", "steps", "seed"}
_STR_FLAGS = {"out", "negative"}
_LIST_FLAGS = {"image"}
_FLAG_ALIASES = {
    "w": "width",
    "h": "height",
    "neg": "negative",
    "negative-prompt": "negative",
    "out-name": "out",
    "ref": "image",
}
# ``--size 1600x960`` expands into width/height; ``x`` or ``*`` as the separator.
_SIZE_RE = re.compile(r"^(\d{1,5})\s*[x*]\s*(\d{1,5})$", re.IGNORECASE)
_FLAG_RE = re.compile(r"(?:(?<=\s)|^)--([A-Za-z][A-Za-z0-9-]*)")
# A value may be quoted, which is what ``--negative "a, b"`` needs.  The prose
# part of the line is never unquoted, so an apostrophe in a prompt is safe.
_VALUE_RE = re.compile(r"""(?:\s*=\s*|\s+)("[^"]*"|'[^']*'|\S+)""")

_RENDER_USAGE = (
    "usage: /qwenimage generate [flags] <prompt>\n"
    "       /qwenimage edit <image> [flags] <prompt>\n"
    "flags: --size WxH | --width N --height N, --steps N, --seed N,\n"
    '       --out name.png, --negative "...", --transparent, --image <path> (extra refs)'
)


def parse_render_flags(text: str) -> tuple[dict[str, Any], str]:
    """Split ``--flag value`` overrides out of a slash-command argument string.

    Flags may appear anywhere in *text*; everything left over is returned as the
    prose part, which is the prompt (and, for ``edit``, the leading file name).
    Returns ``(overrides, prose)`` and raises :class:`ValueError` on an unknown
    flag or a non-integer where an integer is required.
    """
    flags: dict[str, Any] = {}
    kept: list[str] = []
    pos = 0
    while match := _FLAG_RE.search(text, pos):
        name = _FLAG_ALIASES.get(match.group(1).lower(), match.group(1).lower())
        kept.append(text[pos : match.start()])
        end = match.end()
        if name in _BOOL_FLAGS:
            flags[name] = True
        elif name in _INT_FLAGS or name in _STR_FLAGS or name in _LIST_FLAGS or name == "size":
            value_match = _VALUE_RE.match(text, end)
            if value_match is None:
                raise ValueError(f"--{name} needs a value")
            end = value_match.end()
            raw = value_match.group(1)
            if raw[:1] in ("'", '"') and raw[-1:] == raw[:1]:
                raw = raw[1:-1]
            if name in _INT_FLAGS:
                try:
                    flags[name] = int(raw)
                except ValueError:
                    raise ValueError(f"--{name} must be a whole number, got {raw!r}") from None
            elif name == "size":
                size_match = _SIZE_RE.match(raw)
                if size_match is None:
                    raise ValueError(f"--size must look like 1600x960, got {raw!r}")
                flags["width"] = int(size_match.group(1))
                flags["height"] = int(size_match.group(2))
            elif name in _LIST_FLAGS:
                flags.setdefault(name, []).append(raw)
            else:
                flags[name] = raw
        else:
            known = sorted({*_BOOL_FLAGS, *_INT_FLAGS, *_STR_FLAGS, *_LIST_FLAGS, "size"})
            raise ValueError(f"unknown flag --{name}; known flags: {', '.join(known)}")
        pos = end
    kept.append(text[pos:])
    return flags, " ".join("".join(kept).split())


def save_result(result: dict[str, Any], path: Path) -> Path:
    """Write the server's base64 PNG to *path*."""
    images = result.get("images") or []
    if not images:
        raise ValueError("server returned no image")
    try:
        data = base64.b64decode(images[0], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"server returned a malformed image: {exc}") from exc
    path.write_bytes(data)
    return path


def format_result(path: Path, result: dict[str, Any]) -> str:
    """One-line summary of a finished render, plus any pipeline warnings."""
    size = result.get("size") or []
    dims = f"{size[0]}x{size[1]}" if len(size) == 2 else "?"
    bits = [
        f"{dims} {result.get('mode', 'RGB')}",
        f"seed {result.get('seed')}",
        f"{result.get('steps')} steps",
        f"{result.get('seconds', 0):.1f}s",
    ]
    lines = [f"saved {path} ({', '.join(bits)})"]
    if ignored := result.get("ignored"):
        lines.append(
            "note: the installed pipeline does not support "
            + ", ".join(ignored)
            + " — those options had no effect"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extension entry-point
# ---------------------------------------------------------------------------


def register(
    api: Any, config: QwenImageConfig | None = None, client: QwenImageClient | None = None
) -> None:
    """Register the qwen-image tools and the ``/qwenimage`` command."""

    cfg = config or QwenImageConfig.load()
    qwen = client or QwenImageClient(cfg)
    enabled = [t for t in cfg.tools if t in ALL_TOOLS]

    def _payload(
        prompt: str,
        negative_prompt: str | None,
        width: int | None,
        height: int | None,
        steps: int | None,
        seed: int | None,
        transparent: bool,
    ) -> dict[str, Any]:
        w = width or cfg.width
        h = height or cfg.height
        if w * h > cfg.max_pixels:
            raise ValueError(
                f"{w}x{h} exceeds the configured max_pixels ({cfg.max_pixels}); "
                "ask for a smaller image"
            )
        return {
            "prompt": apply_rgba_prompt(prompt) if transparent else prompt,
            "negative_prompt": negative_prompt or "",
            "width": w,
            "height": h,
            "steps": steps or cfg.steps,
            "seed": seed,
            "options": {"true_cfg_scale": cfg.true_cfg_scale},
        }

    async def _render(endpoint: str, payload: dict[str, Any], out: str | None) -> str:
        try:
            path = resolve_out_path(cfg.out_path, out)
        except (ValueError, OSError) as exc:
            return f"qwen-image: {exc}"
        try:
            result = await qwen.render(endpoint, payload)
        except (QwenImageUnavailable, QwenImageError, httpx.HTTPError) as exc:
            return f"qwen-image unavailable: {_describe_failure(exc, cfg)}"
        try:
            save_result(result, path)
        except (ValueError, OSError) as exc:
            return f"qwen-image: {exc}"
        return format_result(path, result)

    # -- tools ----------------------------------------------------------------

    _COMMON_PROPS: dict[str, Any] = {
        "negative_prompt": {
            "type": "string",
            "description": "What to avoid in the image (optional)",
        },
        "width": {"type": "integer", "description": f"Pixels, default {cfg.width}"},
        "height": {"type": "integer", "description": f"Pixels, default {cfg.height}"},
        "steps": {
            "type": "integer",
            "description": f"Denoising steps, default {cfg.steps}; more is slower and cleaner",
        },
        "seed": {"type": "integer", "description": "Fix for reproducible output (optional)"},
        "transparent": {
            "type": "boolean",
            "description": (
                "Render a true RGBA cutout with a transparent background — ideal for "
                "stickers, icons, game sprites and sprite sheets. Compose the image on "
                "the alpha channel; do not ask for a coloured backdrop as well."
            ),
        },
        "out": {
            "type": "string",
            "description": (
                "File name for the PNG, e.g. 'sunset.png'. No directories — it is always "
                "written under the configured output directory. Defaults to a timestamp."
            ),
        },
    }

    if "image_generate" in enabled:

        @api.tool(
            name="image_generate",
            description=(
                "Generate an image from a text prompt with a local Qwen-Image-2.1 model and "
                "save it as a PNG. Good at rendering legible text inside the image. Returns "
                "the file path — you cannot see the pixels unless the user attaches the file "
                "back to the conversation."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What to draw; be specific about subject, style and text",
                    },
                    **_COMMON_PROPS,
                },
                "required": ["prompt"],
            },
            side_effects=["network", "write"],
            # A render outlives the executor's shared command_timeout, which
            # would otherwise cancel it well before request_timeout expires.
            timeout_s=int(cfg.request_timeout) + 30,
        )
        async def image_generate(
            prompt: str,
            negative_prompt: str | None = None,
            width: int | None = None,
            height: int | None = None,
            steps: int | None = None,
            seed: int | None = None,
            transparent: bool = False,
            out: str | None = None,
        ) -> str:
            try:
                payload = _payload(prompt, negative_prompt, width, height, steps, seed, transparent)
            except ValueError as exc:
                return f"qwen-image: {exc}"
            return await _render("/generate", payload, out)

    if "image_edit" in enabled:

        @api.tool(
            name="image_edit",
            description=(
                "Edit, combine or restyle existing images with a local Qwen-Image-2.1 model "
                f"and save the result as a new PNG. Takes up to {MAX_REF_IMAGES} reference "
                "images and never modifies the originals. Use it to change parts of a "
                "picture, extract a subject, or merge references into one scene."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The edit to apply, e.g. 'replace the sky with a sunset'",
                    },
                    "images": {
                        "type": "array",
                        # ``format: path`` is what makes aar's safety policy apply
                        # ``allowed_paths`` / ``denied_paths`` to every reference —
                        # without it an array argument is not recognised as a path.
                        "items": {"type": "string", "format": "path"},
                        "minItems": 1,
                        "maxItems": MAX_REF_IMAGES,
                        "description": (
                            "Paths of the reference images to edit. If the user referred to "
                            "an image as '@some/path.png', pass 'some/path.png' here — that "
                            "attachment is what they want edited. To edit a picture you just "
                            "made, pass the file name you saved it as."
                        ),
                    },
                    **_COMMON_PROPS,
                },
                "required": ["prompt", "images"],
            },
            side_effects=["read", "network", "write"],
            timeout_s=int(cfg.request_timeout) + 30,
        )
        async def image_edit(
            prompt: str,
            images: list[str],
            negative_prompt: str | None = None,
            width: int | None = None,
            height: int | None = None,
            steps: int | None = None,
            seed: int | None = None,
            transparent: bool = False,
            out: str | None = None,
        ) -> str:
            if len(images) > MAX_REF_IMAGES:
                return f"qwen-image: at most {MAX_REF_IMAGES} reference images"
            try:
                payload = _payload(prompt, negative_prompt, width, height, steps, seed, transparent)
                payload["images"] = [read_input_image(p, cfg.out_path) for p in images]
            except (ValueError, OSError) as exc:
                return f"qwen-image: {exc}"
            return await _render("/edit", payload, out)

    if enabled:
        api.append_system_prompt(
            "A local Qwen-Image-2.1 renderer is available ("
            + ", ".join(enabled)
            + f"). Images are written to {cfg.out_path}. Rendering takes tens of seconds to "
            "minutes, so make one image at a time and reuse its seed when iterating. The "
            "tool result is a file path, not the picture — say where you saved it, and ask "
            "the user to attach it with @<path> if you need to look at it yourself. "
            "When the user asks you to change, restyle or fix an existing picture, call "
            "image_edit with that file's path in 'images' — do not call image_generate, "
            "which ignores the original and starts from scratch."
        )

    # -- lifecycle ------------------------------------------------------------

    @api.on("session_start")
    async def on_session_start(event: Any, ctx: Any) -> None:
        if cfg.autostart != "session" or not enabled:
            return
        if await qwen.ahealth() is None:
            try:
                qwen.start()
                ctx.logger.info("qwen-image: starting server (log: %s)", cfg.log_path)
            except QwenImageUnavailable as exc:
                ctx.logger.warning("qwen-image: %s", exc)

    # -- slash-command --------------------------------------------------------

    @api.command(
        "qwenimage",
        description=(
            "Qwen-Image server: status | devices | quant [<name>] | start | stop | "
            "generate <prompt> | edit <image> <prompt>"
        ),
    )
    def qwenimage_command(args: str, ctx: Any) -> str:
        sub, _, rest = (args or "").strip().partition(" ")
        sub = sub.lower() or "status"

        if sub == "status":
            state = qwen.health()
            # A running server keeps the quant it was started with, which is not
            # necessarily the one now in the config.
            lines = [
                f"url:     {cfg.url}",
                f"quant:   {describe_quant((state or {}).get('quant') or cfg.quant)}",
                f"tools:   {', '.join(enabled) or '(none)'}",
                f"out:     {cfg.out_path}",
            ]
            if state is None:
                lines.insert(0, "qwen-image: not running")
                lines.append(f"autostart: {cfg.autostart}  python: {cfg.python_path}")
            else:
                lines.insert(0, f"qwen-image: {state.get('status')}")
                lines[1:1] = [
                    f"model:   {state.get('model')} ({state.get('dtype')}, "
                    f"offload={state.get('offload')})",
                    f"device:  {state.get('device')}",
                    f"pid:     {state.get('pid')}  load: {state.get('load_seconds')}s",
                ]
                if state.get("error"):
                    lines.append(f"error:   {state['error']}")
            lines.append(f"log:     {cfg.log_path}")
            return "\n".join(lines)

        if sub == "devices":
            state = qwen.health()
            if state is None:
                # server_command() is [*launcher, python, server.py, ...flags]
                prefix = qwen.server_command()[: len(cfg.launcher) + 2]
                return (
                    "qwen-image: not running — start it, or list devices directly:\n"
                    f"  {' '.join(prefix)} --list-devices"
                )
            devices = state.get("devices") or []
            if not devices:
                return f"qwen-image: server is {state.get('status')}, no device list yet"
            rows = [
                f"{'*' if d['device'] == str(state.get('requested_device')) else ' '} "
                f"{d['device']:<10} {d['backend']:<9} "
                f"{(str(d['total_gb']) + ' GB') if d['total_gb'] else '?':<9} {d['name']}"
                for d in devices
            ]
            return f"in use: {state.get('device')}\n" + "\n".join(rows)

        if sub == "quant":
            choice = normalise_quant(rest) if rest.strip() else None
            if choice is None and rest.strip():
                return (
                    f"qwen-image: unknown quantization {rest.strip()!r} — "
                    f"one of: none, {', '.join(GGUF_QUANTS)}"
                )
            if choice is None:  # no argument: show the menu
                rows = [("none", BF16_TRANSFORMER_GIB, "full bf16 transformer")]
                rows += [(name, gb, note) for name, (gb, note) in GGUF_QUANTS.items()]
                lines = [
                    f"{'*' if name == cfg.quant else ' '} {name:<8} {gb:>5.1f} GiB  {note}".rstrip()
                    for name, gb, note in rows
                ]
                return "\n".join(
                    [f"repo:    {cfg.quant_repo}", *lines, "", "switch: /qwenimage quant <name>"]
                )
            if choice == cfg.quant:
                return f"qwen-image: already set to {describe_quant(choice)}"
            try:
                write_config_key(config_path(), "quant", choice)
            except (OSError, ValueError) as exc:
                return f"qwen-image: cannot write {config_path()}: {exc}"
            cfg.quant = choice
            tail = (
                " — restart to apply: /qwenimage stop, then /qwenimage start"
                if qwen.health() is not None
                else ""
            )
            return f"qwen-image: quant set to {describe_quant(choice)}{tail}"

        if sub == "start":
            if qwen.health() is not None:
                return "qwen-image: already running — /qwenimage status"
            try:
                qwen.start()
            except QwenImageUnavailable as exc:
                return f"qwen-image: {exc}"
            return (
                f"qwen-image: starting on {cfg.device} (offload={cfg.offload}); log: {cfg.log_path}"
            )

        if sub == "stop":
            reply = qwen.shutdown()
            if reply is None:
                return "qwen-image: not running"
            if reply.get("busy"):
                grace = reply.get("force_after")
                return (
                    "qwen-image: a render is still in flight — the server cannot exit until "
                    "it finishes"
                    + (f", so it will be killed in {grace:.0f}s" if grace else "")
                    + ". That render's image is lost either way; nothing else was queued "
                    "behind it."
                )
            return "qwen-image: stopping"

        if sub in ("generate", "edit"):
            if f"image_{sub}" not in enabled:
                return (
                    f"qwen-image: image_{sub} is not in the configured "
                    f"tools ({', '.join(enabled) or 'none'})"
                )
            try:
                flags, prose = parse_render_flags(rest)
            except ValueError as exc:
                return f"qwen-image: {exc}\n{_RENDER_USAGE}"

            # ``edit`` takes the picture to change as its first word, so that the
            # common single-reference case needs no flag at all.
            images: list[str] = []
            if sub == "edit":
                ref, _, prose = prose.partition(" ")
                if not ref:
                    return _RENDER_USAGE
                images = [ref, *flags.get("image", [])]
                if len(images) > MAX_REF_IMAGES:
                    return f"qwen-image: at most {MAX_REF_IMAGES} reference images"
            elif flags.get("image"):
                return "qwen-image: --image only applies to /qwenimage edit"

            prompt = prose.strip()
            if not prompt:
                return _RENDER_USAGE
            state = qwen.health()
            if state is None or state.get("status") != "ready":
                return "qwen-image: server not ready — /qwenimage start, then /qwenimage status"
            try:
                path = resolve_out_path(cfg.out_path, flags.get("out"))
                payload = _payload(
                    prompt,
                    flags.get("negative"),
                    flags.get("width"),
                    flags.get("height"),
                    flags.get("steps"),
                    flags.get("seed"),
                    bool(flags.get("transparent")),
                )
                if images:
                    payload["images"] = [read_input_image(p, cfg.out_path) for p in images]
                result = qwen.render_sync("/edit" if images else "/generate", payload)
                save_result(result, path)
            except (
                QwenImageUnavailable,
                QwenImageError,
                httpx.HTTPError,
                ValueError,
                OSError,
            ) as exc:
                return f"qwen-image: {exc}"
            return format_result(path, result)

        return (
            "usage: /qwenimage [status | devices | quant [<name>] | start | stop |\n"
            "                   generate <prompt> | edit <image> <prompt>]\n" + _RENDER_USAGE
        )
