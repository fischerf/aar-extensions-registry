"""Qwen-Image-2.1 render server — runs the model in its own Python environment.

This file is deliberately standalone (no package-relative imports): the aar
extension launches it by *file path* with the interpreter of a dedicated venv
that has ``torch`` + ``diffusers`` + ``pillow``, so aar itself never needs the
heavy ML stack.

    python server.py --port 8770 --device auto --offload model
    python server.py --port 8770 --quant Q4_K_M       # GGUF transformer
    python server.py --port 8770 --resident transformer,vae   # pin, don't offload
    python server.py --list-devices                   # what torch sees, then exit

``--quant`` loads the transformer from a GGUF quantization instead of the base
checkpoint's bf16 weights, keeping the text encoder and VAE unquantized.  The
tensor names in those files are already the ones diffusers uses, so nothing is
converted — see ``ensure_single_file_loadable``.

``--device auto`` picks the accelerator with the most memory.  ROCm builds of
torch address AMD cards as ``cuda:N`` too — including AMD's native Windows
wheels — so a Radeon card is selected exactly like an NVIDIA one; ``xpu``,
``mps``, DirectML (``dml:N``) and ``cpu`` are also accepted.

Endpoints (JSON):

    GET  /health    {"status": "loading" | "ready" | "error", ...}
    POST /generate  {"prompt": ..., "width": ..., "height": ..., "steps": ...,
                     "seed": ..., "options": {...}}
                    -> {"images": ["<base64 png>"], "seed": ..., "size": [w, h],
                        "mode": "RGB", "steps": ..., "seconds": ..., "ignored": [...]}
    POST /edit      same, plus {"images": ["<base64 png>", ...]} reference images
    POST /shutdown  stop the server (frees VRAM)

The socket is bound *before* the model loads, so a second instance on the same
port fails immediately instead of downloading 20 GB of weights first.  The
server exits on its own after ``--idle-timeout`` seconds without a request.

Renders are serialised behind one lock: a single diffusion pipeline cannot run
two prompts at once, and on an offloaded GPU a second one would thrash.

``--offload model`` keeps the weights in system RAM and moves each component onto
the card as the pipeline reaches it, so VRAM reads near-zero between renders.
``--resident`` pins named components instead, which matters when the bus is slow
(an eGPU over Thunderbolt) — but only on a card with headroom: whatever stays
resident is memory the activations no longer have. See ``_pin_resident``.

Unknown pipeline options (``true_cfg_scale``, ``output_resolution``, …) are filtered
against the installed pipeline's real signature and reported back in
``ignored`` rather than raising — diffusers' Qwen-Image support moves quickly,
and a missing keyword should degrade to a plain render, not an error.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import inspect
import json
import io
import logging
import os
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger("qwen_image.server")

# "auto" lets ``DiffusionPipeline`` read the checkpoint's model_index.json and pick
# the class the weights were published with — more durable than naming one here,
# since Qwen-Image-2.1 landed on diffusers main before any tagged release had it.
DEFAULT_PIPELINE = "auto"
# Qwen-Image-2.1 is a unified generate+edit model, but diffusers has shipped the
# editing half under several names.  Tried in order, first hit wins.
EDIT_PIPELINE_CANDIDATES = (
    "QwenImage21EditPipeline",
    "QwenImage21EditPlusPipeline",
    "QwenImageEditPlusPipeline",
    "QwenImageEditPipeline",
    "QwenImageImg2ImgPipeline",
)
DTYPES = ("bfloat16", "float16", "float32")
# A graceful uvicorn shutdown waits for in-flight requests.  A render holds the
# pipeline lock for minutes and can wedge outright, and until the process exits
# it keeps the weights on the GPU — so `/shutdown` escalates to a hard exit
# after this many seconds rather than leaving an orphan holding the card.
SHUTDOWN_GRACE_S = 20.0

MAX_SEED = 2**31 - 1

# Kept in step with GGUF_QUANTS in __init__.py by hand: this file is standalone
# on purpose (it runs under a different interpreter) and cannot import from the
# package.
GGUF_QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0")
GGUF_REPO = "abenzerps/Qwen-Image-2.1-GGUF"
GGUF_FILE_TEMPLATE = "qwen-image-2.1-{quant}.gguf"
# Diffusers' GGUF runtime dequantizes nn.Linear weights on demand. These Qwen
# modules are custom normalization layers, so their packed BF16 weights must be
# decoded eagerly instead of being installed as raw GGUFParameter byte tensors.
GGUF_EAGER_DEQUANT_MODULES = ("text_norm", "norm_q", "norm_k")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """What the HTTP layer needs from a model backend (a fake one in tests)."""

    def health(self) -> dict[str, Any]: ...

    def render(self, req: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class ModelSettings:
    model: str = "Qwen/Qwen-Image-2.1"
    pipeline: str = DEFAULT_PIPELINE
    device: str = "auto"
    dtype: str = "bfloat16"
    offload: str = "model"  # "none" | "model" | "sequential"
    quant: str = "none"  # "none" or one of GGUF_QUANTS
    quant_repo: str = GGUF_REPO
    quant_file: str | None = None  # local .gguf path, or a file name inside quant_repo
    # Memory savers that trade a little speed for a much lower activation peak.
    # They matter most with ``offload="none"``: the weights then occupy most of
    # the card, and it is the per-step activations that decide whether a larger
    # image fits or spills into host memory.
    vae_tiling: bool = False
    vae_slicing: bool = False
    attention_slicing: bool = False
    # Pipeline components pinned to the GPU while the rest stay CPU-offloaded.
    # The point is that components are not used equally: the transformer runs
    # once per denoising step, the text encoder once per *render*. On a slow bus
    # (an eGPU over Thunderbolt) pinning the hot components and letting the cold
    # ones travel costs a fraction of moving the whole pipeline every time.
    resident: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Device discovery
#
# torch names devices by *backend*, not by vendor: a ROCm build reports an AMD
# card as ``cuda:0`` and ``torch.version.hip`` is set.  So the selection logic
# below is vendor-agnostic, and only the reported ``backend`` differs.
# ---------------------------------------------------------------------------


def available_devices() -> list[dict[str, Any]]:
    """List the accelerators visible to torch, largest memory first, then cpu."""
    import torch

    found: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            found.append(
                {
                    "device": f"cuda:{i}",
                    "name": props.name,
                    "total_gb": round(props.total_memory / 2**30, 1),
                    "backend": backend,
                }
            )
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        for i in range(xpu.device_count()):
            props = xpu.get_device_properties(i)
            found.append(
                {
                    "device": f"xpu:{i}",
                    "name": getattr(props, "name", "Intel GPU"),
                    "total_gb": round(getattr(props, "total_memory", 0) / 2**30, 1),
                    "backend": "xpu",
                }
            )
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        found.append({"device": "mps", "name": "Apple GPU", "total_gb": 0.0, "backend": "mps"})
    try:  # DirectML — the only way to reach an AMD/Intel GPU on native Windows
        import torch_directml

        for i in range(torch_directml.device_count()):
            found.append(
                {
                    "device": f"dml:{i}",
                    "name": torch_directml.device_name(i),
                    "total_gb": 0.0,
                    "backend": "directml",
                }
            )
    except Exception:
        pass
    found.sort(key=lambda d: d["total_gb"], reverse=True)
    found.append({"device": "cpu", "name": "CPU", "total_gb": 0.0, "backend": "cpu"})
    return found


def resolve_device(requested: str) -> str:
    """Turn ``auto`` into a concrete device string and validate an explicit one."""
    devices = available_devices()
    if requested == "auto":
        chosen = devices[0]["device"]
        logger.info(
            "device auto -> %s (%s); candidates: %s",
            chosen,
            devices[0]["name"],
            ", ".join(f"{d['device']}={d['name']}" for d in devices),
        )
        return chosen
    known = {d["device"] for d in devices}
    if requested not in known and requested.split(":")[0] not in {"cuda", "xpu", "mps", "cpu"}:
        raise RuntimeError(
            f"unknown device {requested!r}; torch sees: "
            + ", ".join(f"{d['device']} ({d['name']})" for d in devices)
        )
    if requested.startswith("cuda") and not any(d["device"].startswith("cuda") for d in devices):
        raise RuntimeError(
            "no CUDA/ROCm device is visible to torch in this environment — "
            "run --list-devices to see what is. An AMD GPU needs a ROCm build of "
            "torch (AMD ships native Windows wheels on repo.radeon.com), not a "
            "stock CUDA build."
        )
    return requested


def torch_device(device: str) -> Any:
    """Map a device string onto something torch accepts (DirectML needs a lookup)."""
    if device.startswith("dml"):
        import torch_directml

        _, _, index = device.partition(":")
        return torch_directml.device(int(index or 0))
    return device


def _register_quant_backends() -> None:
    """Import optional quantization backends so diffusers can load their weights.

    Checkpoints quantized with SDNQ (SD.Next Quantization) are ordinary diffusers
    pipelines whose weights only deserialize once ``sdnq`` has registered itself,
    so the import has to happen *before* ``from_pretrained``.  It is a no-op for
    every other checkpoint, and simply absent on installs that do not need it.

    This is what makes an int4 pipeline — text encoder included — loadable:
    GGUF quantization only covers the transformer, leaving a ~15 GB bf16 text
    encoder that dominates the memory budget.
    """
    try:
        import sdnq  # noqa: F401
    except ImportError:
        logger.debug("sdnq not installed; SDNQ-quantized checkpoints will not load")
        return
    logger.info("sdnq %s registered", getattr(sdnq, "__version__", "?"))


def _resolve_attr(obj: Any, dotted: str) -> Any:
    """Look up a dotted attribute path, returning None if any hop is missing."""
    current = obj
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _is_module(component: Any) -> bool:
    """True for pipeline entries that are actual weights (not tokenizers/schedulers)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present in the venv
        return False
    return isinstance(component, torch.nn.Module)


# ---------------------------------------------------------------------------
# Quantized transformer (GGUF)
#
# Only the transformer is quantized.  The text encoder (Qwen3-VL-8B) and the VAE
# still load from the base checkpoint, and diffusers skips downloading the base
# transformer entirely once one is passed to ``from_pretrained``.
# ---------------------------------------------------------------------------


def gguf_filename(quant: str) -> str:
    """Repo file name for a quantization label."""
    if quant not in GGUF_QUANTS:
        raise ValueError(f"quant must be none or one of {', '.join(GGUF_QUANTS)}, got {quant!r}")
    return GGUF_FILE_TEMPLATE.format(quant=quant)


def resolve_gguf_path(settings: ModelSettings) -> str:
    """Local path of the GGUF transformer, fetching it into the HF cache if needed.

    ``quant_file`` may be a path to a file already on disk — useful for a
    quantization built locally — or a name to look up in ``quant_repo``.
    """
    explicit = settings.quant_file
    if explicit:
        local = Path(explicit).expanduser()
        if local.is_file():
            return str(local)
        if local.is_absolute() or local.parent != Path("."):
            raise RuntimeError(f"quant_file not found: {local}")
        filename = explicit
    else:
        filename = gguf_filename(settings.quant)

    from huggingface_hub import hf_hub_download

    logger.info("resolving %s from %s", filename, settings.quant_repo)
    return hf_hub_download(settings.quant_repo, filename)


def ensure_single_file_loadable(diffusers_mod: Any, cls: Any) -> bool:
    """Teach ``from_single_file`` about *cls* when diffusers has no entry for it.

    ``FromOriginalModelMixin`` dispatches through ``SINGLE_FILE_LOADABLE_CLASSES``
    by walking that table for a base class of *cls*.  Qwen-Image-2.1's
    transformer does not subclass the 1.0 one, so the lookup misses and loading
    raises — even though the checkpoint needs no key conversion at all: the 297
    tensors in the GGUF already carry diffusers' own parameter names, which is
    why the 1.0 entry's mapping function is the identity.  Registering the same
    entry is therefore exact rather than a guess, and turns into a no-op the day
    diffusers ships one.

    Returns True when an entry was added.
    """
    from diffusers.loaders import single_file_model

    table = single_file_model.SINGLE_FILE_LOADABLE_CLASSES
    for name in table:
        known = getattr(diffusers_mod, name, None)
        if isinstance(known, type) and issubclass(cls, known):
            return False
    table[cls.__name__] = {
        "checkpoint_mapping_fn": lambda checkpoint, **kwargs: checkpoint,
        "default_subfolder": "transformer",
    }
    logger.info("registered %s for single-file loading", cls.__name__)
    return True


def gguf_quantization_config(diffusers_mod: Any, dtype: Any) -> Any:
    """Build a GGUF config that eagerly decodes Qwen's non-linear norm weights."""
    config = diffusers_mod.GGUFQuantizationConfig(compute_dtype=dtype)
    config.modules_to_not_convert = list(GGUF_EAGER_DEQUANT_MODULES)
    return config


def split_kwargs(fn: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Split *kwargs* into those *fn* accepts and the names it does not.

    A ``**kwargs`` parameter means everything is accepted.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # C callable / no introspectable signature
        return dict(kwargs), []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), []
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return accepted, sorted(set(kwargs) - set(accepted))


def decode_image(data: str) -> Any:
    """Decode a base64 image into a PIL image."""
    from PIL import Image, UnidentifiedImageError

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"malformed base64 image: {exc}") from exc
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"unreadable image: {exc}") from exc


def encode_image(image: Any) -> str:
    """Encode a PIL image as a base64 PNG (keeping an alpha channel if present)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class QwenImageBackend:
    """Loads Qwen-Image-2.1 and renders one request at a time."""

    settings: ModelSettings
    status: str = "loading"
    error: str | None = None
    load_seconds: float | None = None
    pipeline_name: str | None = None  # the class actually instantiated
    edit_pipeline_name: str | None = None
    transformer_name: str | None = None  # quantized transformer class, when one is used
    quant_path: str | None = None  # resolved .gguf on disk
    device: str = ""  # resolved from settings.device at load time
    devices: list[dict[str, Any]] = field(default_factory=list)
    _pipe: Any = None
    _edit_pipe: Any = None
    _placed: Any = None  # pipeline the offload hooks are currently installed on
    resident_applied: tuple[str, ...] = ()  # of ``settings.resident``, what the pipeline had
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def load(self) -> None:
        t0 = time.monotonic()
        try:
            self._load()
        except Exception as exc:  # reported via /health, the process stays up
            logger.exception("model load failed")
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            return
        self.load_seconds = round(time.monotonic() - t0, 1)
        self.status = "ready"
        logger.info("model ready in %.1fs (%s)", self.load_seconds, self._vram())

    def _load(self) -> None:
        import diffusers
        import torch

        s = self.settings
        _register_quant_backends()
        if s.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {', '.join(DTYPES)}, got {s.dtype!r}")
        self.devices = available_devices()
        self.device = resolve_device(s.device)

        if s.pipeline == "auto":
            pipe_cls = diffusers.DiffusionPipeline
        else:
            pipe_cls = getattr(diffusers, s.pipeline, None)
            if pipe_cls is None:
                raise RuntimeError(
                    f"diffusers {diffusers.__version__} has no {s.pipeline}; it exposes "
                    + ", ".join(n for n in dir(diffusers) if n.endswith("Pipeline"))[:400]
                )

        dtype = getattr(torch, s.dtype)
        extra: dict[str, Any] = {}
        if s.quant != "none" or s.quant_file:
            extra["transformer"] = self._load_gguf_transformer(diffusers, dtype)

        logger.info(
            "loading %s via %s (%s, quant=%s) on %s",
            s.model, s.pipeline, s.dtype, s.quant, self.device,
        )  # fmt: skip
        try:
            pipe = pipe_cls.from_pretrained(s.model, torch_dtype=dtype, **extra)
        except ValueError as exc:
            # from_pretrained raises ValueError for many reasons; only an unknown class
            # name means "your diffusers is too old". Attaching that hint to every
            # ValueError sends people chasing the wrong fix — a missing
            # processor/chat_template.jinja surfaces here too, for instance.
            if "cannot be loaded" in str(exc) or "has no attribute" in str(exc):
                raise RuntimeError(
                    f"{exc}\n\ndiffusers {diffusers.__version__} does not know this "
                    "pipeline class. Qwen-Image-2.1 support may only exist on main: "
                    "pip install -U git+https://github.com/huggingface/diffusers"
                ) from exc
            raise
        self.pipeline_name = type(pipe).__name__
        self._pipe = pipe

        # Qwen-Image-2.1 is a *unified* model: its pipeline takes `image=` for editing,
        # so prefer it over any separate Edit class — those expect the older
        # Qwen-Image components and would be handed mismatched weights.
        _, missing = split_kwargs(type(pipe).__call__, {"image": None})
        if not missing:
            self._edit_pipe = pipe
            self.edit_pipeline_name = self.pipeline_name
        else:
            # Older checkpoints split the two. Reuse the loaded weights rather than
            # a second copy, and build *before* offload hooks are installed, because
            # those hooks are per-pipeline bookkeeping over shared modules.
            for name in EDIT_PIPELINE_CANDIDATES:
                edit_cls = getattr(diffusers, name, None)
                if edit_cls is None:
                    continue
                try:
                    self._edit_pipe = edit_cls(**pipe.components)
                except (TypeError, ValueError) as exc:
                    logger.warning("cannot build %s from loaded components: %s", name, exc)
                    continue
                self.edit_pipeline_name = name
                break
            else:
                self._edit_pipe = pipe  # no candidate: /edit will report the problem
                self.edit_pipeline_name = self.pipeline_name
        logger.info("edit pipeline: %s", self.edit_pipeline_name)
        self._place(pipe)
        self._placed = pipe

    def _load_gguf_transformer(self, diffusers: Any, dtype: Any) -> Any:
        """Build the transformer from a GGUF file instead of the base weights.

        The class to build is read from the checkpoint's ``model_index.json``
        rather than hardcoded, for the same reason ``pipeline: "auto"`` is the
        default: Qwen-Image's diffusers classes are still being renamed.
        """
        s = self.settings
        index = diffusers.DiffusionPipeline.load_config(s.model)
        entry = index.get("transformer")
        if not entry or len(entry) != 2:
            raise RuntimeError(f"{s.model} declares no transformer in model_index.json")
        cls = getattr(diffusers, entry[1], None)
        if cls is None:
            raise RuntimeError(
                f"diffusers {diffusers.__version__} has no {entry[1]}, which "
                f"{s.model} needs; pip install -U git+https://github.com/huggingface/diffusers"
            )
        ensure_single_file_loadable(diffusers, cls)

        path = resolve_gguf_path(s)
        self.quant_path = path
        self.transformer_name = cls.__name__
        logger.info("loading %s from %s (compute dtype %s)", cls.__name__, path, s.dtype)
        return cls.from_single_file(
            path,
            quantization_config=gguf_quantization_config(diffusers, dtype),
            config=s.model,
            subfolder="transformer",
            torch_dtype=dtype,
        )

    def _place(self, pipe: Any) -> None:
        """Apply the configured offload strategy, or move the pipeline to the GPU."""
        offload = self.settings.offload
        device = torch_device(self.device)
        if offload == "sequential":
            self._pin_resident(pipe)
            pipe.enable_sequential_cpu_offload(device=device)
        elif offload == "model":
            self._pin_resident(pipe)
            pipe.enable_model_cpu_offload(device=device)
        elif offload == "none":
            pipe.to(device)
        else:
            raise ValueError(f"offload must be none, model or sequential, got {offload!r}")

        self._apply_memory_savers(pipe)

    def _pin_resident(self, pipe: Any) -> None:
        """Exclude the configured components from CPU offloading.

        Two things are needed, and only doing the first is a silent no-op:

        * ``_exclude_from_cpu_offload`` names components that should be moved to
          the device once and left there.  But diffusers only consults it for
          components *outside* ``model_cpu_offload_seq`` — everything in the
          chain is hooked unconditionally.
        * So the pinned names must also be removed from ``model_cpu_offload_seq``.
          They then fall through to the branch that honours the exclusion list.

        Unknown names are dropped with a warning rather than raising — component
        names differ between pipeline classes, and a stale config entry should
        not stop the server from starting.
        """
        wanted = [c.strip() for c in self.settings.resident if c.strip()]
        if not wanted:
            return

        available = {
            name
            for name, component in getattr(pipe, "components", {}).items()
            if _is_module(component)
        }
        unknown = [name for name in wanted if name not in available]
        if unknown:
            logger.warning(
                "resident component(s) %s not on this pipeline; known: %s",
                ", ".join(unknown),
                ", ".join(sorted(available)) or "(none)",
            )
        keep = [name for name in wanted if name in available]
        self.resident_applied = tuple(keep)
        if not keep:
            return

        pipe._exclude_from_cpu_offload = list(keep)

        seq = getattr(pipe, "model_cpu_offload_seq", None)
        if seq:
            remaining = [name for name in seq.split("->") if name not in keep]
            pipe.model_cpu_offload_seq = "->".join(remaining)
            logger.debug("offload chain %r -> %r", seq, pipe.model_cpu_offload_seq)
        logger.info("pinned to %s: %s", self.device, ", ".join(keep))

    def _apply_memory_savers(self, pipe: Any) -> None:
        """Enable the configured activation-memory reducers.

        Each is best-effort: diffusers' Qwen-Image pipelines have gained and lost
        these helpers between releases, and a missing one should degrade to "not
        enabled" rather than refuse to start the server.
        """
        s = self.settings
        # Each saver has two spellings: the pipeline-level helper, and the one on
        # the VAE itself. Qwen-Image pipelines ship the second but not always the
        # first, so trying only ``pipe.enable_vae_tiling()`` reports "unavailable"
        # for something the model card actively recommends.
        wanted = (
            ("vae_tiling", s.vae_tiling, ("enable_vae_tiling", "vae.enable_tiling")),
            ("vae_slicing", s.vae_slicing, ("enable_vae_slicing", "vae.enable_slicing")),
            ("attention_slicing", s.attention_slicing, ("enable_attention_slicing",)),
        )
        for label, enabled, candidates in wanted:
            if not enabled:
                continue
            for path in candidates:
                fn = _resolve_attr(pipe, path)
                if fn is None:
                    continue
                try:
                    fn()
                    logger.info("%s enabled via %s()", label, path)
                    break
                except Exception:
                    logger.warning("%s: %s() failed", label, path, exc_info=True)
            else:
                logger.warning(
                    "%s requested but none of %s are available on this pipeline",
                    label,
                    ", ".join(f"{c}()" for c in candidates),
                )

    def _vram(self) -> str:
        if self.device.startswith("cuda"):
            try:
                import torch

                used = torch.cuda.memory_allocated(torch.device(self.device)) / 2**30
                name = torch.cuda.get_device_name(torch.device(self.device))
                return f"{self.device} ({name}), {used:.2f} GB allocated"
            except Exception:
                pass
        for entry in self.devices:
            if entry["device"] == self.device:
                return f"{self.device} ({entry['name']})"
        return self.device

    def health(self) -> dict[str, Any]:
        s = self.settings
        return {
            "status": self.status,
            "error": self.error,
            "model": s.model,
            "pipeline": self.pipeline_name or s.pipeline,
            "edit_pipeline": self.edit_pipeline_name,
            "transformer": self.transformer_name,
            "dtype": s.dtype,
            "offload": s.offload,
            "vae_tiling": s.vae_tiling,
            "vae_slicing": s.vae_slicing,
            "attention_slicing": s.attention_slicing,
            "resident": list(s.resident),
            "resident_applied": list(self.resident_applied),
            # quant_file alone (no --quant) still means a quantized transformer
            "quant": s.quant if s.quant != "none" else ("custom" if self.quant_path else "none"),
            "quant_file": self.quant_path,
            "requested_device": s.device,
            "device": self._vram() if self.status == "ready" else s.device,
            "devices": self.devices,
            "load_seconds": self.load_seconds,
            "pid": os.getpid(),
        }

    def _generator(self, seed: int) -> Any:
        import torch

        # With offloading the modules move between CPU and GPU, and DirectML has
        # no generator of its own, so a CPU generator is the portable choice.
        on_device = self.settings.offload == "none" and self.device.startswith("cuda")
        return torch.Generator(device=self.device if on_device else "cpu").manual_seed(seed)

    @property
    def busy(self) -> bool:
        """True while a render holds the pipeline lock.

        Used by ``/shutdown``: a graceful exit has to wait for the in-flight
        render, which on an offloaded GPU is minutes away at best.
        """
        return self._lock.locked()

    def render(self, req: dict[str, Any]) -> dict[str, Any]:
        """Run one generate (no ``images``) or edit (with ``images``) request."""
        if self.status != "ready":
            raise RuntimeError(f"model not ready (status={self.status})")

        refs = [decode_image(b) for b in req.get("images") or []]
        pipe = self._edit_pipe if refs else self._pipe
        if pipe is not self._placed:
            # Offload hooks belong to one pipeline at a time even when the two share
            # modules, so hand them over before running the other one.
            self._place(pipe)
            self._placed = pipe
        seed = req.get("seed")
        seed = random.randint(0, MAX_SEED) if seed is None else int(seed) % (MAX_SEED + 1)

        kwargs: dict[str, Any] = {
            "prompt": req["prompt"],
            "width": req["width"],
            "height": req["height"],
            "num_inference_steps": req["steps"],
            "generator": self._generator(seed),
        }
        if req.get("negative_prompt"):
            kwargs["negative_prompt"] = req["negative_prompt"]
        if refs:
            kwargs["image"] = refs if len(refs) > 1 else refs[0]

        options = {k: v for k, v in (req.get("options") or {}).items() if v not in (None, False)}
        accepted, ignored = split_kwargs(type(pipe).__call__, {**kwargs, **options})
        if "image" in ignored:
            raise RuntimeError(
                f"{self.edit_pipeline_name} does not accept reference images; "
                "upgrade diffusers or set 'tools' to ['image_generate'] in the config"
            )

        t0 = time.monotonic()
        with self._lock:
            image = pipe(**accepted).images[0]
        seconds = round(time.monotonic() - t0, 1)
        logger.info(
            "rendered %dx%d in %.1fs (%d refs, seed %d)",
            image.width, image.height, seconds, len(refs), seed,
        )  # fmt: skip
        return {
            "images": [encode_image(image)],
            "seed": seed,
            "size": [image.width, image.height],
            "mode": image.mode,
            "steps": req["steps"],
            "seconds": seconds,
            "ignored": ignored,
        }


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------


class RenderRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    steps: int = Field(default=30, ge=1, le=200)
    seed: int | None = None
    images: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class Activity:
    """Tracks the time of the last request for the idle watchdog."""

    def __init__(self) -> None:
        self.last = time.monotonic()

    def touch(self) -> None:
        self.last = time.monotonic()

    def idle_for(self) -> float:
        return time.monotonic() - self.last


def create_app(
    backend: Backend,
    activity: Activity | None = None,
    on_shutdown: Any = None,
    max_images: int = 10,
    max_pixels: int = 2048 * 2048,
) -> Any:
    """Build the FastAPI app around *backend* (kept separate for testing)."""
    from fastapi import FastAPI, HTTPException

    activity = activity or Activity()
    app = FastAPI(title="qwen-image", docs_url=None, redoc_url=None)

    def _render(req: RenderRequest, editing: bool) -> dict[str, Any]:
        if editing and not req.images:
            raise HTTPException(422, detail="/edit needs at least one reference image")
        if len(req.images) > max_images:
            raise HTTPException(422, detail=f"at most {max_images} reference images")
        if req.width * req.height > max_pixels:
            raise HTTPException(
                422, detail=f"{req.width}x{req.height} exceeds max_pixels ({max_pixels})"
            )
        activity.touch()
        state = backend.health()
        if state["status"] != "ready":
            raise HTTPException(503, detail=state.get("error") or state["status"])
        try:
            return backend.render(req.model_dump())
        except ValueError as exc:  # unreadable reference image
            raise HTTPException(422, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("render failed")
            raise HTTPException(500, detail=str(exc)) from exc
        finally:
            activity.touch()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return backend.health()

    @app.post("/generate")
    def generate(req: RenderRequest) -> dict[str, Any]:
        return _render(req.model_copy(update={"images": []}), editing=False)

    @app.post("/edit")
    def edit(req: RenderRequest) -> dict[str, Any]:
        return _render(req, editing=True)

    @app.post("/shutdown")
    def shutdown() -> dict[str, Any]:
        busy = bool(getattr(backend, "busy", False))
        logger.info("shutdown requested via /shutdown (busy=%s)", busy)
        if on_shutdown is not None:
            on_shutdown()
        # A render in flight means uvicorn's graceful shutdown cannot complete
        # until it does, so the caller is told the process will be killed.
        return {"status": "stopping", "busy": busy, "force_after": SHUTDOWN_GRACE_S}

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = ModelSettings()
    ap = argparse.ArgumentParser(description="Qwen-Image-2.1 render server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--pipeline", default=d.pipeline, help="diffusers text-to-image pipeline class")
    ap.add_argument(
        "--device",
        default=d.device,
        help="auto (most memory), cuda:N (NVIDIA, or AMD on a ROCm build), xpu, mps, dml:N, cpu",
    )
    ap.add_argument(
        "--list-devices", action="store_true", help="print the devices torch sees and exit"
    )
    ap.add_argument("--dtype", choices=DTYPES, default=d.dtype)
    ap.add_argument(
        "--offload",
        choices=["none", "model", "sequential"],
        default=d.offload,
        help="CPU offload strategy: none needs ~20 GB VRAM, sequential the least (and is slowest)",
    )
    ap.add_argument(
        "--resident",
        default=",".join(d.resident),
        help=(
            "Comma-separated pipeline components to pin to the GPU instead of offloading "
            "them, e.g. 'transformer,vae'. Only meaningful with --offload model/sequential"
        ),
    )
    ap.add_argument(
        "--vae-tiling",
        action="store_true",
        default=d.vae_tiling,
        help="Decode the VAE in tiles — much lower peak memory at high resolutions",
    )
    ap.add_argument(
        "--vae-slicing",
        action="store_true",
        default=d.vae_slicing,
        help="Decode one latent at a time — lowers peak memory when batching",
    )
    ap.add_argument(
        "--attention-slicing",
        action="store_true",
        default=d.attention_slicing,
        help="Compute attention in slices — lowers the per-step activation peak",
    )
    ap.add_argument(
        "--quant",
        choices=["none", *GGUF_QUANTS],
        default=d.quant,
        help="load the transformer from a GGUF quantization instead of the bf16 weights",
    )
    ap.add_argument(
        "--quant-repo", default=d.quant_repo, help="Hugging Face repo holding the GGUFs"
    )
    ap.add_argument(
        "--quant-file",
        default=d.quant_file,
        help="explicit .gguf path, or a file name inside --quant-repo; overrides --quant",
    )
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument(
        "--pid-file",
        default=None,
        help="Write this process id here while running, so a second start can "
        "detect an orphaned server that still holds the GPU",
    )
    ap.add_argument("--max-pixels", type=int, default=2048 * 2048)
    ap.add_argument(
        "--idle-timeout",
        type=float,
        default=900.0,
        help="exit after this many seconds without a request (0 = never)",
    )
    return ap.parse_args(argv)


def should_shut_down_for_idle(backend: Any, activity: Activity, idle_timeout: float) -> bool:
    """True when the server has genuinely been idle long enough to exit.

    A server that has not finished loading is **not** idle.  Loading the SDNQ
    checkpoint takes 60-90s on a 7900 XTX, so an ``idle_timeout`` shorter than
    that used to kill the process mid-load — it never served a single request,
    and the client saw the connection refused.
    """
    if idle_timeout <= 0:
        return False
    try:
        if backend.health().get("status") != "ready":
            return False
    except Exception:  # pragma: no cover - a backend that cannot self-report
        return False
    return activity.idle_for() > idle_timeout


def force_exit_after(
    grace: float,
    exited: threading.Event,
    pid_file: str | None = None,
    exit_fn: Any = None,
) -> threading.Thread:
    """Hard-exit the process if a graceful shutdown has not finished in *grace*.

    uvicorn waits for in-flight requests before returning from ``run()``.  A
    render holds the pipeline lock for minutes and can wedge indefinitely, and
    for all that time the process keeps its weights on the GPU while no longer
    answering on its port — so a second server can be started on top of it.
    Escalating to ``os._exit`` is what makes ``/shutdown`` mean it.

    Returns the watchdog thread; it ends as soon as *exited* is set.
    """
    exit_fn = exit_fn or os._exit

    def wait_then_kill() -> None:
        if exited.wait(grace):
            return
        logger.warning(
            "still running %.0fs after shutdown — forcing exit (pid %d)", grace, os.getpid()
        )
        _remove_pid_file(pid_file)
        exit_fn(3)

    t = threading.Thread(target=wait_then_kill, name="qwen-image-force-exit", daemon=True)
    t.start()
    return t


def _write_pid_file(path: str | None, port: int) -> None:
    """Record this process so a later ``start`` can see an orphan holding the GPU.

    The listening socket is *not* enough: uvicorn closes it as soon as a
    graceful shutdown begins, while the process — and its share of the card —
    lives on until the in-flight render returns.
    """
    if not path:
        return
    try:
        f = Path(path).expanduser()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"pid": os.getpid(), "port": port}), encoding="utf-8")
    except OSError:  # pragma: no cover - a pidfile is best-effort
        logger.warning("could not write pid file %s", path, exc_info=True)


def _remove_pid_file(path: str | None) -> None:
    """Delete the pidfile, tolerating a concurrent delete."""
    if not path:
        return
    try:
        Path(path).expanduser().unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.list_devices:
        for d in available_devices():
            size = f"{d['total_gb']:.1f} GB" if d["total_gb"] else "unknown size"
            print(f"{d['device']:<12} {d['backend']:<9} {size:<12} {d['name']}")
        return 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((args.host, args.port))
    except OSError as exc:
        logger.error(
            "cannot bind %s:%d (%s) — is another server running?", args.host, args.port, exc
        )
        return 2

    import uvicorn

    backend = QwenImageBackend(
        ModelSettings(
            model=args.model,
            pipeline=args.pipeline,
            device=args.device,
            dtype=args.dtype,
            offload=args.offload,
            vae_tiling=args.vae_tiling,
            vae_slicing=args.vae_slicing,
            attention_slicing=args.attention_slicing,
            resident=tuple(c.strip() for c in args.resident.split(",") if c.strip()),
            quant=args.quant,
            quant_repo=args.quant_repo,
            quant_file=args.quant_file,
        )
    )
    activity = Activity()
    server: uvicorn.Server | None = None
    exited = threading.Event()

    def stop() -> None:
        if server is None:
            return
        server.should_exit = True
        force_exit_after(SHUTDOWN_GRACE_S, exited, args.pid_file)

    app = create_app(
        backend,
        activity,
        on_shutdown=stop,
        max_images=args.max_images,
        max_pixels=args.max_pixels,
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))

    def watchdog() -> None:
        while not server.should_exit:
            time.sleep(10)
            if should_shut_down_for_idle(backend, activity, args.idle_timeout):
                logger.info("idle for %.0fs — shutting down", activity.idle_for())
                stop()

    def load_then_mark_ready() -> None:
        # The idle clock starts at process start, but loading 11-31 GiB is not
        # idleness: without this the watchdog can shut the server down before it
        # has served anything, and a short ``idle_timeout`` becomes unusable.
        try:
            backend.load()
        finally:
            activity.touch()

    threading.Thread(target=load_then_mark_ready, name="qwen-image-load", daemon=True).start()
    threading.Thread(target=watchdog, name="qwen-image-idle", daemon=True).start()
    logger.info("listening on http://%s:%d (pid %d)", args.host, args.port, os.getpid())
    _write_pid_file(args.pid_file, args.port)
    try:
        server.run(sockets=[sock])
    finally:
        exited.set()
        _remove_pid_file(args.pid_file)
    logger.info("server stopped (pid %d)", os.getpid())
    return 0


if __name__ == "__main__":
    sys.exit(main())
