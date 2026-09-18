"""Tests for the openjev extension (no GPU / model needed — HTTP is mocked)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from aar_ext_openjev import (
    RERANK_HYPOTHESIS,
    SERVER_SCRIPT,
    OpenJevClient,
    OpenJevConfig,
    format_check,
    register,
)

LABELS = ["contradiction", "entailment", "neutral"]


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
    """Answers /health and /predict; ``scores`` maps hypothesis -> probs row."""

    def __init__(self, status: str = "ready", scores: dict[str, list[float]] | None = None):
        self.status = status
        self.scores = scores or {}
        self.requests: list[dict[str, Any]] = []
        self.shutdowns = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": self.status, "model": "m", "bits": 4})
        if request.url.path == "/shutdown":
            self.shutdowns += 1
            return httpx.Response(200, json={"status": "stopping"})
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != "ready":
            return httpx.Response(503, json={"detail": self.status})
        probs = [self.scores.get(p["hypothesis"], [0.1, 0.8, 0.1]) for p in body["pairs"]]
        return httpx.Response(
            200, json={"labels": LABELS, "probs": probs, "truncated": [False] * len(probs)}
        )


def make(server: FakeServer | None, **cfg: Any) -> tuple[FakeAPI, OpenJevClient]:
    config = OpenJevConfig(**cfg)
    if server is None:  # connection refused

        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        handler: Any = refuse
    else:
        handler = server
    client = OpenJevClient(
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
    monkeypatch.delenv("AAR_OPENJEV_URL", raising=False)
    cfg = OpenJevConfig.load(tmp_path / "missing.json")
    assert cfg.url == "http://127.0.0.1:8765"
    assert cfg.bits == 4
    assert cfg.autostart == "on_demand"
    assert cfg.tools == ["nli_check", "nli_rerank", "nli_grade"]


def test_config_file_and_env_override(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "openjev.json"
    path.write_text(json.dumps({"bits": 8, "tools": ["nli_check"]}), encoding="utf-8")
    monkeypatch.setenv("AAR_OPENJEV_URL", "http://127.0.0.1:9999")
    cfg = OpenJevConfig.load(path)
    assert cfg.bits == 8
    assert cfg.tools == ["nli_check"]
    assert cfg.url == "http://127.0.0.1:9999"


def test_example_config_is_valid(monkeypatch) -> None:
    monkeypatch.delenv("AAR_OPENJEV_URL", raising=False)
    example = Path(__file__).parent.parent / "openjev.example.json"
    cfg = OpenJevConfig.load(example)
    assert cfg.autostart == "off"
    defaults = OpenJevConfig()
    for key in ("url", "bits", "device", "max_length", "batch_size", "tools"):
        assert getattr(cfg, key) == getattr(defaults, key)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "openjev.json"
    path.write_text(json.dumps({"bitz": 4}), encoding="utf-8")
    with pytest.raises(ValueError, match="bitz"):
        OpenJevConfig.load(path)


def test_config_rejects_bad_autostart(tmp_path: Path) -> None:
    path = tmp_path / "openjev.json"
    path.write_text(json.dumps({"autostart": "always"}), encoding="utf-8")
    with pytest.raises(ValueError, match="autostart"):
        OpenJevConfig.load(path)


def test_server_command_targets_script_and_port() -> None:
    client = OpenJevClient(OpenJevConfig(url="http://127.0.0.1:9001", bits=4, python="py"))
    cmd = client.server_command()
    assert cmd[:2] == ["py", str(SERVER_SCRIPT)]
    assert cmd[cmd.index("--port") + 1] == "9001"
    assert cmd[cmd.index("--bits") + 1] == "4"
    assert SERVER_SCRIPT.is_file()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_all_tools_and_prompt() -> None:
    api, _ = make(FakeServer())
    assert set(api.tools) == {"nli_check", "nli_rerank", "nli_grade"}
    assert "openjev" in api.commands
    assert api.prompts and "nli_check" in api.prompts[0]
    assert all(m["side_effects"] == ["network"] for m in api.tool_meta.values())


def test_register_tool_subset() -> None:
    api, _ = make(FakeServer(), tools=["nli_grade"])
    assert set(api.tools) == {"nli_grade"}


def test_register_no_tools_no_prompt() -> None:
    api, _ = make(FakeServer(), tools=[])
    assert api.tools == {}
    assert api.prompts == []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def test_nli_check_formats_verdicts() -> None:
    server = FakeServer(
        scores={
            "The bird is below the gap.": [0.02, 0.95, 0.03],
            "The bird is above the gap.": [0.90, 0.05, 0.05],
            "The bird is blue.": [0.10, 0.10, 0.80],
        }
    )
    api, _ = make(server)
    claims = ["The bird is below the gap.", "The bird is above the gap.", "The bird is blue."]
    out = await api.tools["nli_check"](premise="The bird is 0.05 below the gap.", claims=claims)

    assert out.startswith("1 supported, 1 contradicted, 1 not established")
    assert "1. SUPPORTED" in out
    assert "2. CONTRADICTED" in out
    assert "3. NOT ESTABLISHED" in out
    assert server.requests[0]["pairs"][0]["premise"] == "The bird is 0.05 below the gap."


async def test_nli_rerank_orders_by_entailment() -> None:
    options = ["oxygen", "carbon dioxide", "nitrogen"]
    server = FakeServer(
        scores={
            RERANK_HYPOTHESIS.format("oxygen"): [0.6, 0.3, 0.1],
            RERANK_HYPOTHESIS.format("carbon dioxide"): [0.05, 0.9, 0.05],
            RERANK_HYPOTHESIS.format("nitrogen"): [0.7, 0.1, 0.2],
        }
    )
    api, _ = make(server)
    out = await api.tools["nli_rerank"](question="Which gas do plants absorb?", options=options)

    assert out.splitlines()[0] == "best option: [1] carbon dioxide"
    ranked = out.splitlines()[1:]
    assert ranked[0].startswith("1. [1]") and ranked[1].startswith("2. [0]")
    hyps = [p["hypothesis"] for p in server.requests[0]["pairs"]]
    assert hyps == [RERANK_HYPOTHESIS.format(o) for o in options]


async def test_nli_grade_uses_reference_premise() -> None:
    server = FakeServer(scores={"Answer: Paris": [0.01, 0.97, 0.02]})
    api, _ = make(server)
    out = await api.tools["nli_grade"](
        question="Capital of France?", reference="Paris", answer="Paris"
    )
    assert out.startswith("CORRECT")
    pair = server.requests[0]["pairs"][0]
    assert pair["premise"] == "Capital of France?\nReference answer: Paris"


async def test_tool_reports_unavailable_when_autostart_off() -> None:
    api, _ = make(None, autostart="off")
    out = await api.tools["nli_check"](premise="p", claims=["h"])
    assert out.startswith("openjev unavailable")
    assert "/openjev start" in out


async def test_tool_reports_missing_interpreter(tmp_path: Path) -> None:
    api, _ = make(None, python=str(tmp_path / "nope" / "python.exe"))
    out = await api.tools["nli_check"](premise="p", claims=["h"])
    assert "interpreter not found" in out


async def test_tool_reports_load_error() -> None:
    api, _ = make(FakeServer(status="error"))
    out = await api.tools["nli_check"](premise="p", claims=["h"])
    assert "failed to load" in out


def test_format_check_flags_truncation() -> None:
    result = {"labels": LABELS, "probs": [[0.1, 0.8, 0.1]], "truncated": [True]}
    assert "truncated" in format_check(["h"], result)


# ---------------------------------------------------------------------------
# Slash-command
# ---------------------------------------------------------------------------


def test_command_status_not_running() -> None:
    api, _ = make(None)
    out = api.commands["openjev"]("", MagicMock())
    assert out.startswith("openjev: not running")


def test_command_status_running() -> None:
    api, _ = make(FakeServer())
    out = api.commands["openjev"]("status", MagicMock())
    assert out.startswith("openjev: ready")
    assert "4-bit" in out


def test_command_check_happy_path() -> None:
    api, _ = make(FakeServer(scores={"music": [0.01, 0.98, 0.01]}))
    out = api.commands["openjev"]("check A man plays guitar => music", MagicMock())
    assert "1. SUPPORTED" in out


def test_command_check_usage() -> None:
    api, _ = make(FakeServer())
    assert api.commands["openjev"]("check no arrow", MagicMock()).startswith("usage:")


def test_command_stop() -> None:
    server = FakeServer()
    api, _ = make(server)
    assert api.commands["openjev"]("stop", MagicMock()) == "openjev: stopping"
    assert server.shutdowns == 1


def test_command_start_when_running() -> None:
    api, _ = make(FakeServer())
    assert "already running" in api.commands["openjev"]("start", MagicMock())


# ---------------------------------------------------------------------------
# Server HTTP layer (fake backend, no torch)
# ---------------------------------------------------------------------------


class FakeBackend:
    def __init__(self, status: str = "ready") -> None:
        self.status = status

    def health(self) -> dict[str, Any]:
        return {"status": self.status, "error": None}

    def predict(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        return {"labels": LABELS, "probs": [[0.1, 0.8, 0.1]] * len(pairs), "truncated": []}


def _server_app(backend: FakeBackend, **kw: Any):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import aar_ext_openjev.server as server

    return TestClient(server.create_app(backend, **kw))


def test_server_predict_and_health() -> None:
    client = _server_app(FakeBackend())
    assert client.get("/health").json()["status"] == "ready"
    r = client.post("/predict", json={"pairs": [{"premise": "p", "hypothesis": "h"}]})
    assert r.status_code == 200
    assert r.json()["probs"] == [[0.1, 0.8, 0.1]]


def test_server_predict_503_while_loading() -> None:
    client = _server_app(FakeBackend(status="loading"))
    r = client.post("/predict", json={"pairs": [{"premise": "p", "hypothesis": "h"}]})
    assert r.status_code == 503


def test_server_rejects_too_many_pairs() -> None:
    client = _server_app(FakeBackend(), max_pairs=2)
    pairs = [{"premise": "p", "hypothesis": "h"}] * 3
    assert client.post("/predict", json={"pairs": pairs}).status_code == 422


def test_server_shutdown_calls_hook() -> None:
    called: list[bool] = []
    client = _server_app(FakeBackend(), on_shutdown=lambda: called.append(True))
    assert client.post("/shutdown").json() == {"status": "stopping"}
    assert called == [True]


# ---------------------------------------------------------------------------
# Live (real server) — opt-in: AAR_OPENJEV_LIVE=1
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("AAR_OPENJEV_LIVE"), reason="set AAR_OPENJEV_LIVE=1")
async def test_live_entailment_and_contradiction() -> None:
    client = OpenJevClient(OpenJevConfig.load())
    result = await client.predict(
        [
            ("A man is playing a guitar on stage.", "Someone is making music."),
            ("A man is playing a guitar on stage.", "The man is asleep in bed."),
        ]
    )
    rows = [dict(zip(result["labels"], p)) for p in result["probs"]]
    assert max(rows[0], key=rows[0].get) == "entailment"
    assert max(rows[1], key=rows[1].get) == "contradiction"
