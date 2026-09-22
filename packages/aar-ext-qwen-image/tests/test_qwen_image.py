"""Tests for the qwen-image extension (no GPU / diffusers needed — HTTP is mocked)."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from aar_ext_qwen_image import (
    ASPECT_RATIOS,
    GGUF_QUANTS,
    apply_rgba_prompt,
    describe_quant,
    MAX_REF_IMAGES,
    normalise_quant,
    QwenImageClient,
    QwenImageConfig,
    format_result,
    read_input_image,
    register,
    resolve_out_path,
    save_result,
)

# A 1x1 transparent PNG — enough to exercise encode/decode without pillow.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "DUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAPI:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.tool_meta: dict[str, dict[str, Any]] = {}
        self.handlers: dict[str, list[Any]] = {}
        self.commands: dict[str, Any] = {}
        self.prompts: list[str] = []

    def tool(self, name: str, description: str, input_schema: dict, **kw: Any):
        def deco(fn):
            self.tools[name] = fn
            self.tool_meta[name] = {"schema": input_schema, **kw}
            return fn

        return deco

    def on(self, event: str):
        def deco(fn):
            self.handlers.setdefault(event, []).append(fn)
            return fn

        return deco

    def command(self, name: str, *, description: str = ""):
        def deco(fn):
            self.commands[name] = fn
            return fn

        return deco

    def append_system_prompt(self, text: str) -> None:
        self.prompts.append(text)


class FakeServer:
    """Answers /health, /generate, /edit and /shutdown."""

    def __init__(self, status: str = "ready", mode: str = "RGB") -> None:
        self.status = status
        self.mode = mode
        self.quant = "none"
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.shutdowns = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": self.status,
                    "model": "Qwen/Qwen-Image-2.1",
                    "quant": self.quant,
                    "dtype": "bfloat16",
                    "offload": "model",
                    "requested_device": "cuda:0",
                    "device": "cuda:0 (Radeon RX 7900 XTX), 18.20 GB allocated",
                    "devices": [
                        {
                            "device": "cuda:0",
                            "name": "Radeon RX 7900 XTX",
                            "total_gb": 24.0,
                            "backend": "rocm",
                        },
                        {"device": "cpu", "name": "CPU", "total_gb": 0.0, "backend": "cpu"},
                    ],
                },
            )
        if path == "/shutdown":
            self.shutdowns += 1
            return httpx.Response(200, json={"status": "stopping"})
        body = json.loads(request.content)
        self.requests.append((path, body))
        if self.status != "ready":
            return httpx.Response(503, json={"detail": self.status})
        return httpx.Response(
            200,
            json={
                "images": [PNG_B64],
                "seed": 7,
                "size": [body["width"], body["height"]],
                "mode": self.mode,
                "steps": body["steps"],
                "seconds": 12.5,
                "ignored": [],
            },
        )


def make(server: FakeServer | None, **cfg: Any) -> tuple[FakeAPI, QwenImageClient]:
    config = QwenImageConfig(**cfg)
    if server is None:  # connection refused

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        handler: Any = refuse
    else:
        handler = server
    client = QwenImageClient(
        config,
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(handler),
    )
    api = FakeAPI()
    register(api, config=config, client=client)
    return api, client


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_when_file_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AAR_QWEN_IMAGE_URL", raising=False)
    cfg = QwenImageConfig.load(tmp_path / "missing.json")
    assert cfg.url == "http://127.0.0.1:8770"
    assert cfg.device == "auto"
    assert cfg.offload == "model"
    assert cfg.autostart == "on_demand"
    assert cfg.tools == ["image_generate", "image_edit"]


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "qwen-image.json"
    path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown qwen-image config keys"):
        QwenImageConfig.load(path)


def test_config_rejects_bad_offload(tmp_path: Path) -> None:
    path = tmp_path / "qwen-image.json"
    path.write_text(json.dumps({"offload": "half"}), encoding="utf-8")
    with pytest.raises(ValueError, match="offload must be"):
        QwenImageConfig.load(path)


def test_config_env_overrides_url(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AAR_QWEN_IMAGE_URL", "http://127.0.0.1:9999")
    assert QwenImageConfig.load(tmp_path / "missing.json").url == "http://127.0.0.1:9999"


# ---------------------------------------------------------------------------
# Server command — including the WSL launcher used to reach an AMD GPU
# ---------------------------------------------------------------------------


def test_server_command_uses_local_interpreter_by_default() -> None:
    _, client = make(None, python="/usr/bin/python3", device="cuda:1", offload="sequential")
    cmd = client.server_command()
    assert cmd[1].endswith("server.py")
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "cuda:1"
    assert cmd[cmd.index("--offload") + 1] == "sequential"


def test_server_command_prepends_launcher() -> None:
    _, client = make(
        None,
        launcher=["wsl.exe", "-d", "aar-ubuntu", "--"],
        python="/home/me/.aar/qwen-image/.venv/bin/python",
        server_script="/mnt/b/aar/server.py",
    )
    cmd = client.server_command()
    assert cmd[:4] == ["wsl.exe", "-d", "aar-ubuntu", "--"]
    assert cmd[4] == "/home/me/.aar/qwen-image/.venv/bin/python"
    assert cmd[5] == "/mnt/b/aar/server.py"


async def test_launcher_skips_local_interpreter_check(tmp_path: Path) -> None:
    """A WSL interpreter has no Windows path, so it must not be probed locally."""
    api, client = make(
        None,
        launcher=["wsl.exe", "-d", "aar-ubuntu", "--"],
        python="/home/me/.venv/bin/python",
        out_dir=str(tmp_path),
    )
    out = await api.tools["image_generate"](prompt="x")
    assert "interpreter not found" not in out
    assert out.startswith("qwen-image unavailable")


async def test_missing_local_interpreter_is_reported(tmp_path: Path) -> None:
    api, _ = make(None, python=str(tmp_path / "nope" / "python.exe"), out_dir=str(tmp_path))
    out = await api.tools["image_generate"](prompt="x")
    assert "interpreter not found" in out


# ---------------------------------------------------------------------------
# Output paths — the model chooses the name, so it must not choose the directory
# ---------------------------------------------------------------------------


def test_render_tools_outlive_the_shared_command_timeout(tmp_path: Path) -> None:
    """The executor caps every tool at tools.command_timeout (300s by default).

    A 2048px render on an offloaded card takes longer than that, so the tools
    must carry their own timeout_s or request_timeout can never take effect.
    """
    api, _ = make(None, out_dir=str(tmp_path), request_timeout=900.0)
    for name in ("image_generate", "image_edit"):
        assert api.tool_meta[name]["timeout_s"] > 900


def test_out_path_defaults_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert QwenImageConfig().out_path == Path.cwd()


def test_out_path_relative_is_a_cwd_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert QwenImageConfig(out_dir="assets/renders").out_path == Path.cwd() / "assets" / "renders"


def test_out_path_absolute_ignores_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixed = tmp_path.parent / "fixed"
    assert QwenImageConfig(out_dir=str(fixed)).out_path == fixed


def test_resolve_out_path_defaults_to_timestamp(tmp_path: Path) -> None:
    path = resolve_out_path(tmp_path, None)
    assert path.parent == tmp_path
    assert path.suffix == ".png"


def test_resolve_out_path_forces_png(tmp_path: Path) -> None:
    assert resolve_out_path(tmp_path, "sunset.jpg").name == "sunset.png"


def test_resolve_out_path_never_overwrites(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    assert resolve_out_path(tmp_path, "a.png").name == "a-1.png"


@pytest.mark.parametrize(
    "name",
    ["../escape.png", "sub/dir.png", "sub\\dir.png", "/abs/path.png", "..", ".hidden.png"],
)
def test_resolve_out_path_rejects_traversal(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="invalid file name"):
        resolve_out_path(tmp_path, name)


async def test_tool_rejects_traversal_before_rendering(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    out = await api.tools["image_generate"](prompt="x", out="../escape.png")
    assert "invalid file name" in out
    assert server.requests == []  # never reached the GPU


# ---------------------------------------------------------------------------
# Reference images
# ---------------------------------------------------------------------------


def test_read_input_image_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "ref.png"
    src.write_bytes(base64.b64decode(PNG_B64))
    assert base64.b64decode(read_input_image(str(src))) == src.read_bytes()


def test_read_input_image_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        read_input_image(str(tmp_path / "nope.png"))


def test_read_input_image_rejects_non_image(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported reference image"):
        read_input_image(str(src))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def test_image_generate_saves_file(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path), steps=25)
    out = await api.tools["image_generate"](prompt="a red bicycle", out="bike.png")

    saved = tmp_path / "bike.png"
    assert saved.is_file()
    assert saved.read_bytes() == base64.b64decode(PNG_B64)
    assert out.startswith(f"saved {saved}")
    assert "seed 7" in out and "25 steps" in out

    endpoint, body = server.requests[0]
    assert endpoint == "/generate"
    assert body["prompt"] == "a red bicycle"
    assert body["steps"] == 25
    assert body["options"]["true_cfg_scale"] == 4.0


async def test_image_generate_honours_explicit_size_and_seed(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    await api.tools["image_generate"](prompt="x", width=768, height=512, seed=42)
    _, body = server.requests[0]
    assert (body["width"], body["height"], body["seed"]) == (768, 512, 42)


async def test_image_generate_rejects_oversized_request(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path), max_pixels=1024 * 1024)
    out = await api.tools["image_generate"](prompt="x", width=4096, height=4096)
    assert "exceeds the configured max_pixels" in out
    assert server.requests == []


async def test_image_edit_sends_encoded_references(tmp_path: Path) -> None:
    ref = tmp_path / "ref.png"
    ref.write_bytes(base64.b64decode(PNG_B64))
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))

    out = await api.tools["image_edit"](prompt="make it night", images=[str(ref)], out="night.png")

    endpoint, body = server.requests[0]
    assert endpoint == "/edit"
    assert body["images"] == [PNG_B64]
    assert (tmp_path / "night.png").is_file()
    assert out.startswith("saved")


async def test_image_edit_rejects_too_many_references(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    out = await api.tools["image_edit"](prompt="x", images=["a.png"] * (MAX_REF_IMAGES + 1))
    assert f"at most {MAX_REF_IMAGES} reference images" in out
    assert server.requests == []


async def test_image_edit_reports_missing_reference(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    out = await api.tools["image_edit"](prompt="x", images=[str(tmp_path / "nope.png")])
    assert "not found" in out
    assert server.requests == []


async def test_tool_reports_unavailable_when_autostart_off(tmp_path: Path) -> None:
    api, _ = make(None, autostart="off", out_dir=str(tmp_path))
    out = await api.tools["image_generate"](prompt="x")
    assert out.startswith("qwen-image unavailable")
    assert "/qwenimage start" in out


async def test_tool_reports_load_error(tmp_path: Path) -> None:
    api, _ = make(FakeServer(status="error"), out_dir=str(tmp_path))
    out = await api.tools["image_generate"](prompt="x")
    assert "failed to load" in out


def test_tools_declare_side_effects(tmp_path: Path) -> None:
    api, _ = make(FakeServer(), out_dir=str(tmp_path))
    assert api.tool_meta["image_generate"]["side_effects"] == ["network", "write"]
    assert api.tool_meta["image_edit"]["side_effects"] == ["read", "network", "write"]


def test_tools_subset_is_respected(tmp_path: Path) -> None:
    api, _ = make(FakeServer(), out_dir=str(tmp_path), tools=["image_generate"])
    assert set(api.tools) == {"image_generate"}


def test_format_result_reports_ignored_options(tmp_path: Path) -> None:
    result = {"size": [8, 8], "mode": "RGBA", "seed": 1, "steps": 2, "seconds": 3.0}
    out = format_result(tmp_path / "x.png", {**result, "ignored": ["transparent"]})
    assert "does not support transparent" in out


def test_save_result_rejects_malformed_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="malformed image"):
        save_result({"images": ["not base64 !!"]}, tmp_path / "x.png")


# ---------------------------------------------------------------------------
# Slash-command
# ---------------------------------------------------------------------------


def test_command_status_not_running(tmp_path: Path) -> None:
    api, _ = make(None, out_dir=str(tmp_path))
    out = api.commands["qwenimage"]("", MagicMock())
    assert out.startswith("qwen-image: not running")


def test_command_status_running(tmp_path: Path) -> None:
    api, _ = make(FakeServer(), out_dir=str(tmp_path))
    out = api.commands["qwenimage"]("status", MagicMock())
    assert out.startswith("qwen-image: ready")
    assert "offload=model" in out


def test_command_devices_lists_what_torch_sees(tmp_path: Path) -> None:
    api, _ = make(FakeServer(), out_dir=str(tmp_path))
    out = api.commands["qwenimage"]("devices", MagicMock())
    assert "Radeon RX 7900 XTX" in out
    assert "rocm" in out
    assert "24.0 GB" in out


def test_command_devices_when_not_running(tmp_path: Path) -> None:
    api, _ = make(None, out_dir=str(tmp_path))
    assert "--list-devices" in api.commands["qwenimage"]("devices", MagicMock())


def test_command_generate_happy_path(tmp_path: Path) -> None:
    api, _ = make(FakeServer(), out_dir=str(tmp_path))
    out = api.commands["qwenimage"]("generate a cat on a bike", MagicMock())
    assert out.startswith("saved")
    assert list(tmp_path.glob("*.png"))


def test_command_generate_usage(tmp_path: Path) -> None:
    api, _ = make(FakeServer(), out_dir=str(tmp_path))
    assert api.commands["qwenimage"]("generate", MagicMock()).startswith("usage:")


def test_command_stop(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    assert api.commands["qwenimage"]("stop", MagicMock()) == "qwen-image: stopping"
    assert server.shutdowns == 1


def test_command_start_when_running(tmp_path: Path) -> None:
    api, _ = make(FakeServer(), out_dir=str(tmp_path))
    assert "already running" in api.commands["qwenimage"]("start", MagicMock())


# ---------------------------------------------------------------------------
# Server: kwarg filtering + HTTP layer (fake backend, no torch)
# ---------------------------------------------------------------------------


def test_split_kwargs_reports_unknown_names() -> None:
    from aar_ext_qwen_image.server import split_kwargs

    def pipe(prompt, width=None, height=None):  # noqa: ANN001, ANN202
        return None

    accepted, ignored = split_kwargs(pipe, {"prompt": "x", "width": 8, "transparent": True})
    assert accepted == {"prompt": "x", "width": 8}
    assert ignored == ["transparent"]


def test_split_kwargs_accepts_everything_with_var_keyword() -> None:
    from aar_ext_qwen_image.server import split_kwargs

    def pipe(prompt, **kw):  # noqa: ANN001, ANN202
        return None

    accepted, ignored = split_kwargs(pipe, {"prompt": "x", "whatever": 1})
    assert ignored == [] and accepted == {"prompt": "x", "whatever": 1}


# ---------------------------------------------------------------------------
# Server: resolving the GGUF transformer
#
# The abenzerps GGUFs carry diffusers' own parameter names, so no key conversion
# is needed — but QwenImage21Transformer2DModel is not a subclass of the 1.0
# transformer, so diffusers' single-file lookup misses it and has to be taught.
# ---------------------------------------------------------------------------


def test_gguf_filename_matches_the_repo_layout() -> None:
    from aar_ext_qwen_image.server import gguf_filename

    assert gguf_filename("Q4_K_M") == "qwen-image-2.1-Q4_K_M.gguf"
    assert gguf_filename("Q8_0") == "qwen-image-2.1-Q8_0.gguf"
    with pytest.raises(ValueError, match="quant must be none or one of"):
        gguf_filename("Q3_K")


def test_resolve_gguf_path_prefers_a_file_already_on_disk(tmp_path: Path) -> None:
    from aar_ext_qwen_image.server import ModelSettings, resolve_gguf_path

    local = tmp_path / "custom.gguf"
    local.write_bytes(b"GGUF")
    # No network: an existing path short-circuits the hub download.
    assert resolve_gguf_path(ModelSettings(quant_file=str(local))) == str(local)


def test_resolve_gguf_path_reports_a_missing_local_file(tmp_path: Path) -> None:
    from aar_ext_qwen_image.server import ModelSettings, resolve_gguf_path

    missing = tmp_path / "nope.gguf"
    with pytest.raises(RuntimeError, match="quant_file not found"):
        resolve_gguf_path(ModelSettings(quant_file=str(missing)))


def test_ensure_single_file_loadable_registers_only_unknown_classes(monkeypatch) -> None:
    """Fakes diffusers: this pins our shim, not the library's table."""
    from aar_ext_qwen_image.server import ensure_single_file_loadable

    class Known:
        pass

    class Subclass(Known):
        pass

    class Unknown:
        pass

    table: dict[str, Any] = {"Known": {"checkpoint_mapping_fn": lambda ckpt, **kw: ckpt}}
    single_file_model = ModuleType("diffusers.loaders.single_file_model")
    single_file_model.SINGLE_FILE_LOADABLE_CLASSES = table
    loaders = ModuleType("diffusers.loaders")
    loaders.single_file_model = single_file_model
    diffusers = ModuleType("diffusers")
    diffusers.loaders = loaders
    diffusers.Known = Known
    for name, mod in [
        ("diffusers", diffusers),
        ("diffusers.loaders", loaders),
        ("diffusers.loaders.single_file_model", single_file_model),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    # a subclass of a registered class already dispatches — leave the table alone
    assert ensure_single_file_loadable(diffusers, Subclass) is False
    assert "Subclass" not in table

    assert ensure_single_file_loadable(diffusers, Unknown) is True
    entry = table["Unknown"]
    assert entry["default_subfolder"] == "transformer"
    # the mapping is the identity: the GGUF's keys are already diffusers' own
    state_dict = {"transformer_blocks.0.attn.to_q.weight": 1}
    assert entry["checkpoint_mapping_fn"](state_dict) is state_dict


def test_gguf_config_eagerly_dequantizes_custom_norms() -> None:
    from aar_ext_qwen_image.server import gguf_quantization_config

    class FakeConfig:
        def __init__(self, compute_dtype: Any) -> None:
            self.compute_dtype = compute_dtype
            self.modules_to_not_convert: list[str] | None = None

    diffusers = MagicMock(GGUFQuantizationConfig=FakeConfig)
    config = gguf_quantization_config(diffusers, "bfloat16")

    assert config.compute_dtype == "bfloat16"
    assert config.modules_to_not_convert == ["text_norm", "norm_q", "norm_k"]


class FakeBackend:
    def __init__(self, status: str = "ready") -> None:
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {"status": self.status, "error": None}

    def render(self, req: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(req)
        return {
            "images": [PNG_B64],
            "seed": 1,
            "size": [req["width"], req["height"]],
            "mode": "RGB",
            "steps": req["steps"],
            "seconds": 0.1,
            "ignored": [],
        }


def _server_app(backend: FakeBackend, **kw: Any):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import aar_ext_qwen_image.server as server

    return TestClient(server.create_app(backend, **kw))


def test_server_generate_and_health() -> None:
    client = _server_app(FakeBackend())
    assert client.get("/health").json()["status"] == "ready"
    r = client.post("/generate", json={"prompt": "x", "width": 64, "height": 64, "steps": 2})
    assert r.status_code == 200
    assert r.json()["size"] == [64, 64]


def test_server_generate_drops_smuggled_reference_images() -> None:
    backend = FakeBackend()
    client = _server_app(backend)
    client.post("/generate", json={"prompt": "x", "images": [PNG_B64]})
    assert backend.calls[0]["images"] == []


def test_server_edit_requires_an_image() -> None:
    client = _server_app(FakeBackend())
    assert client.post("/edit", json={"prompt": "x"}).status_code == 422


def test_server_edit_rejects_too_many_images() -> None:
    client = _server_app(FakeBackend(), max_images=2)
    r = client.post("/edit", json={"prompt": "x", "images": [PNG_B64] * 3})
    assert r.status_code == 422


def test_server_rejects_oversized_request() -> None:
    client = _server_app(FakeBackend(), max_pixels=64 * 64)
    r = client.post("/generate", json={"prompt": "x", "width": 1024, "height": 1024})
    assert r.status_code == 422


def test_server_503_while_loading() -> None:
    client = _server_app(FakeBackend(status="loading"))
    assert client.post("/generate", json={"prompt": "x"}).status_code == 503


def test_server_shutdown_calls_hook() -> None:
    called: list[bool] = []
    client = _server_app(FakeBackend(), on_shutdown=lambda: called.append(True))
    assert client.post("/shutdown").json() == {"status": "stopping"}
    assert called == [True]


# ---------------------------------------------------------------------------
# GGUF quantization switch
#
# The GGUF files hold only the transformer (4.1-7.6 GB instead of 13.3 GB); the
# text encoder and VAE still come from the base checkpoint.
# ---------------------------------------------------------------------------


def test_config_defaults_to_unquantized_weights(tmp_path: Path) -> None:
    cfg = QwenImageConfig.load(tmp_path / "missing.json")
    assert cfg.quant == "none"
    assert cfg.quant_file is None
    assert cfg.quant_repo == "abenzerps/Qwen-Image-2.1-GGUF"


def test_config_rejects_bad_quant(tmp_path: Path) -> None:
    path = tmp_path / "qwen-image.json"
    path.write_text(json.dumps({"quant": "Q3"}), encoding="utf-8")
    with pytest.raises(ValueError, match="quant must be none or one of"):
        QwenImageConfig.load(path)


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("Q4_K_M", "Q4_K_M"),
        ("q4_k_m", "Q4_K_M"),
        ("q4-k-m", "Q4_K_M"),
        (" Q8_0 ", "Q8_0"),
        ("none", "none"),
        ("bf16", "none"),
        ("off", "none"),
        ("Q3_K_S", None),
    ],
)
def test_normalise_quant(typed: str, expected: str | None) -> None:
    assert normalise_quant(typed) == expected


def test_resident_components_are_empty_by_default() -> None:
    _, client = make(None)
    assert "--resident" not in client.server_command()


def test_resident_components_reach_the_server_command() -> None:
    _, client = make(None, resident_components=["transformer", "vae"])
    cmd = client.server_command()
    assert cmd[cmd.index("--resident") + 1] == "transformer,vae"


def test_memory_savers_are_off_by_default() -> None:
    _, client = make(None)
    cmd = client.server_command()
    for flag in ("--vae-tiling", "--vae-slicing", "--attention-slicing"):
        assert flag not in cmd


def test_memory_savers_reach_the_server_command() -> None:
    """They matter with offload="none", where activations decide what fits."""
    _, client = make(None, offload="none", vae_tiling=True, attention_slicing=True)
    cmd = client.server_command()
    assert "--vae-tiling" in cmd
    assert "--attention-slicing" in cmd
    assert "--vae-slicing" not in cmd
    assert cmd[cmd.index("--offload") + 1] == "none"


def test_server_command_passes_the_quant_through() -> None:
    _, client = make(None, quant="Q4_K_M")
    cmd = client.server_command()
    assert cmd[cmd.index("--quant") + 1] == "Q4_K_M"
    assert cmd[cmd.index("--quant-repo") + 1] == "abenzerps/Qwen-Image-2.1-GGUF"
    assert "--quant-file" not in cmd


def test_server_command_includes_an_explicit_quant_file() -> None:
    _, client = make(None, quant_file="/models/custom.gguf")
    cmd = client.server_command()
    assert cmd[cmd.index("--quant-file") + 1] == "/models/custom.gguf"


def test_command_quant_lists_every_option(tmp_path: Path) -> None:
    api, _ = make(None, out_dir=str(tmp_path))
    out = api.commands["qwenimage"]("quant", MagicMock())
    for name in GGUF_QUANTS:
        assert name in out
    assert "* none" in out  # the default is marked as active
    assert "13.3 GiB" in out


def test_command_quant_persists_the_choice(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "qwen-image.json"
    path.write_text(json.dumps({"steps": 25}), encoding="utf-8")
    monkeypatch.setenv("AAR_QWEN_IMAGE_CONFIG", str(path))

    api, client = make(None, out_dir=str(tmp_path), steps=25)
    out = api.commands["qwenimage"]("quant q4_k_m", MagicMock())

    assert "Q4_K_M" in out
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"steps": 25, "quant": "Q4_K_M"}  # other keys survive
    # the live config follows, so the next start uses it without a reload
    assert client.config.quant == "Q4_K_M"
    assert client.server_command()[client.server_command().index("--quant") + 1] == "Q4_K_M"


def test_command_quant_asks_for_a_restart_when_running(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AAR_QWEN_IMAGE_CONFIG", str(tmp_path / "qwen-image.json"))
    api, _ = make(FakeServer(), out_dir=str(tmp_path))
    out = api.commands["qwenimage"]("quant Q8_0", MagicMock())
    assert "restart to apply" in out


def test_command_quant_rejects_an_unknown_label(tmp_path: Path) -> None:
    api, _ = make(None, out_dir=str(tmp_path))
    out = api.commands["qwenimage"]("quant Q2_K", MagicMock())
    assert out.startswith("qwen-image: unknown quantization")
    assert "Q4_K_M" in out


def test_status_reports_the_quant_the_server_actually_loaded(tmp_path: Path) -> None:
    """A running server keeps the quant it started with, not the one in the config."""
    server = FakeServer()
    server.quant = "Q4_K_M"
    api, _ = make(server, out_dir=str(tmp_path), quant="Q8_0")
    assert "Q4_K_M" in api.commands["qwenimage"]("status", MagicMock())


def test_describe_quant() -> None:
    assert describe_quant("none").startswith("none (full bf16")
    assert describe_quant("Q4_K_M") == "Q4_K_M (GGUF, 4.3 GiB, recommended)"
    assert describe_quant("custom") == "custom"


# ---------------------------------------------------------------------------
# Live (real server) — opt-in: AAR_QWEN_IMAGE_LIVE=1
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("AAR_QWEN_IMAGE_LIVE"), reason="set AAR_QWEN_IMAGE_LIVE=1")
async def test_live_generate(tmp_path: Path) -> None:
    cfg = QwenImageConfig.load()
    client = QwenImageClient(cfg)
    result = await client.render(
        "/generate",
        {
            "prompt": "a single red apple on a white table",
            "negative_prompt": "",
            "width": 512,
            "height": 512,
            "steps": 8,
            "seed": 1,
            "options": {},
        },
    )
    path = save_result(result, tmp_path / "live.png")
    assert path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Server-side failures must reach the caller
#
# httpx.HTTPStatusError carries only the status line, so a bare
# raise_for_status() drops FastAPI's ``detail`` — the only description of *why*
# a render failed.  These pin the message to the tool result.
# ---------------------------------------------------------------------------


async def test_render_error_detail_reaches_the_tool_result(tmp_path: Path) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return FakeServer()(request)
        return httpx.Response(500, json={"detail": "CUDA out of memory, tried to allocate 13 GB"})

    client = QwenImageClient(
        QwenImageConfig(out_dir=str(tmp_path)),
        transport=httpx.MockTransport(failing),
        async_transport=httpx.MockTransport(failing),
    )
    api = FakeAPI()
    register(api, config=QwenImageConfig(out_dir=str(tmp_path)), client=client)

    out = await api.tools["image_generate"](prompt="a red bicycle")
    assert "CUDA out of memory, tried to allocate 13 GB" in out
    assert "500" in out
    assert not list(tmp_path.glob("*.png"))  # nothing saved on failure


async def test_render_error_without_json_body_still_reports_text(tmp_path: Path) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return FakeServer()(request)
        return httpx.Response(502, text="upstream died")

    client = QwenImageClient(
        QwenImageConfig(out_dir=str(tmp_path)),
        transport=httpx.MockTransport(failing),
        async_transport=httpx.MockTransport(failing),
    )
    api = FakeAPI()
    register(api, config=QwenImageConfig(out_dir=str(tmp_path)), client=client)

    out = await api.tools["image_generate"](prompt="a red bicycle")
    assert "upstream died" in out
    assert "502" in out


async def test_transport_error_names_the_error_and_points_at_the_log(tmp_path: Path) -> None:
    """A crashed server must not report an empty 'qwen-image unavailable:'."""
    log = tmp_path / "server.log"
    log.write_text("Exception Code: 0xC0000005\nalloc_cpu failed\n", encoding="utf-8")

    def die(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return FakeServer()(request)
        raise httpx.ReadError("", request=request)  # stringifies to ""

    cfg = QwenImageConfig(out_dir=str(tmp_path), log_file=str(log))
    client = QwenImageClient(
        cfg, transport=httpx.MockTransport(die), async_transport=httpx.MockTransport(die)
    )
    api = FakeAPI()
    register(api, config=cfg, client=client)

    out = await api.tools["image_generate"](prompt="a red bicycle")
    assert "unavailable:" in out
    assert out.rstrip() != "qwen-image unavailable:"
    assert "ReadError" in out  # the empty message is replaced by the type name
    assert str(log) in out  # and the user is told where to look
    assert "0xC0000005" in out  # with the smoking gun quoted


# ---------------------------------------------------------------------------
# Native RGBA transparency
#
# Qwen-Image-2.1 has no `transparent` pipeline argument: RGBA is requested in the
# prompt text. Sending it as a pipeline option silently did nothing.
# ---------------------------------------------------------------------------


async def test_transparent_wraps_the_prompt_in_the_rgba_template(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    await api.tools["image_generate"](prompt="a cartoon dragon sticker", transparent=True)

    _, body = server.requests[0]
    assert body["prompt"].startswith("This is an RGBA image with transparency.")
    assert body["prompt"].endswith("the background is transparent.")
    assert "a cartoon dragon sticker" in body["prompt"]
    # the dead pipeline kwarg must not be sent any more
    assert "transparent" not in body["options"]


async def test_transparent_false_leaves_the_prompt_alone(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    await api.tools["image_generate"](prompt="a red bicycle")

    _, body = server.requests[0]
    assert body["prompt"] == "a red bicycle"
    assert "transparent" not in body["options"]


async def test_transparent_is_not_applied_twice(tmp_path: Path) -> None:
    """A prompt that already asks for RGBA must not be wrapped again."""
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    already = (
        "This is an RGBA image with transparency. A dragon. "
        "The image has alpha channel and the background is transparent."
    )
    await api.tools["image_generate"](prompt=already, transparent=True)

    _, body = server.requests[0]
    assert body["prompt"] == already
    assert body["prompt"].lower().count("rgba image with transparency") == 1


def test_apply_rgba_prompt_is_idempotent() -> None:
    once = apply_rgba_prompt("a dragon")
    assert apply_rgba_prompt(once) == once


@pytest.mark.parametrize("ratio", sorted(ASPECT_RATIOS))
async def test_every_documented_aspect_ratio_is_renderable(ratio: str, tmp_path: Path) -> None:
    """The model card's aspect ratios must not trip the max_pixels guard.

    A 2048*2048 cap rejected all six non-square sizes.
    """
    w, h = ASPECT_RATIOS[ratio]
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    out = await api.tools["image_generate"](prompt="x", width=w, height=h)

    assert "exceeds" not in out, out
    _, body = server.requests[0]
    assert (body["width"], body["height"]) == (w, h)


async def test_oversized_request_is_still_rejected(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    out = await api.tools["image_generate"](prompt="x", width=4096, height=4096)
    assert "exceeds" in out
    assert not server.requests  # never reached the server


# ---------------------------------------------------------------------------
# Reference paths a model realistically produces
# ---------------------------------------------------------------------------


async def test_image_edit_accepts_at_prefixed_path(tmp_path: Path) -> None:
    """`@path` is aar's attachment syntax; models copy it through verbatim."""
    ref = tmp_path / "neon.png"
    ref.write_bytes(base64.b64decode(PNG_B64))
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))

    out = await api.tools["image_edit"](prompt="make it snowy", images=[f"@{ref}"])
    assert "not found" not in out
    _, body = server.requests[0]
    assert body["images"] == [PNG_B64]


async def test_image_edit_resolves_bare_name_against_out_dir(tmp_path: Path) -> None:
    """image_generate saves bare names into out_dir, so edits must find them there."""
    ref = tmp_path / "neon.png"
    ref.write_bytes(base64.b64decode(PNG_B64))
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))

    out = await api.tools["image_edit"](prompt="make it snowy", images=["neon.png"])
    assert "not found" not in out
    _, body = server.requests[0]
    assert body["images"] == [PNG_B64]


async def test_image_edit_still_reports_a_genuinely_missing_file(tmp_path: Path) -> None:
    server = FakeServer()
    api, _ = make(server, out_dir=str(tmp_path))
    out = await api.tools["image_edit"](prompt="x", images=["@nope.png"])
    assert "not found" in out
    assert not server.requests


def test_read_input_image_rejects_empty_after_stripping_at(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty reference image path"):
        read_input_image("@", tmp_path)


# ---------------------------------------------------------------------------
# Per-component placement (server side)
#
# ``_exclude_from_cpu_offload`` is diffusers' own hook: names listed there are
# moved onto the device once and left, while everything else gets offload hooks.
# ---------------------------------------------------------------------------


class _FakePipe:
    def __init__(self, components):
        self.components = components
        self._exclude_from_cpu_offload = []
        self.model_cpu_offload_seq = "text_encoder->transformer->vae"


def _backend(**settings):
    from aar_ext_qwen_image.server import ModelSettings, QwenImageBackend

    return QwenImageBackend(ModelSettings(**settings))


def _pipe_with_modules():
    """A pipeline whose weight-bearing entries are real (empty) nn.Modules."""
    import torch

    mods = {name: torch.nn.Linear(1, 1) for name in ("transformer", "vae", "text_encoder")}
    mods["tokenizer"] = object()  # not a module — must never be pinned
    return _FakePipe(mods)


def test_pin_resident_excludes_named_components() -> None:
    pytest.importorskip("torch")
    backend = _backend(resident=("transformer", "vae"))
    pipe = _pipe_with_modules()
    backend._pin_resident(pipe)
    assert pipe._exclude_from_cpu_offload == ["transformer", "vae"]
    assert backend.resident_applied == ("transformer", "vae")


def test_pin_resident_removes_them_from_the_offload_chain() -> None:
    """Without this the exclusion list is a silent no-op: diffusers only honours
    it for components outside ``model_cpu_offload_seq``."""
    pytest.importorskip("torch")
    backend = _backend(resident=("transformer", "vae"))
    pipe = _pipe_with_modules()
    backend._pin_resident(pipe)
    assert pipe.model_cpu_offload_seq == "text_encoder"


def test_pin_resident_leaves_the_chain_alone_when_unset() -> None:
    pytest.importorskip("torch")
    backend = _backend()
    pipe = _pipe_with_modules()
    backend._pin_resident(pipe)
    assert pipe.model_cpu_offload_seq == "text_encoder->transformer->vae"


def test_pin_resident_drops_unknown_names_instead_of_raising() -> None:
    """Component names differ between pipeline classes; a stale entry must not
    stop the server from starting."""
    pytest.importorskip("torch")
    backend = _backend(resident=("transformer", "unet"))
    pipe = _pipe_with_modules()
    backend._pin_resident(pipe)
    assert pipe._exclude_from_cpu_offload == ["transformer"]
    assert backend.resident_applied == ("transformer",)


def test_pin_resident_ignores_non_modules() -> None:
    pytest.importorskip("torch")
    backend = _backend(resident=("tokenizer",))
    pipe = _pipe_with_modules()
    backend._pin_resident(pipe)
    assert pipe._exclude_from_cpu_offload == []


def test_pin_resident_is_a_noop_when_unset() -> None:
    pytest.importorskip("torch")
    backend = _backend()
    pipe = _pipe_with_modules()
    backend._pin_resident(pipe)
    assert pipe._exclude_from_cpu_offload == []
