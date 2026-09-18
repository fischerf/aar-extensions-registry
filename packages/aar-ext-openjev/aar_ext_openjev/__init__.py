"""Aar extension: openjev — local NLI verification tools backed by AlexWortega/openjev.

openjev is a Qwen3.5-4B cross-encoder that classifies a (premise, hypothesis)
pair as entailment / contradiction / neutral.  It is *not* a chat model, so it
runs next to the agent's normal provider as a small local HTTP server
(``server.py``) in a dedicated Python environment, typically on a spare GPU.

Tools exposed to the model:

- ``nli_check``  — which claims does a source text support / contradict?
- ``nli_rerank`` — rank candidate answers to a question by P(entailment)
- ``nli_grade``  — grade an answer against a reference answer

Slash-command ``/openjev [status|start|stop|check <premise> => <hypothesis>]``.

Configuration is read from ``~/.aar/openjev.json`` (all keys optional)::

    {
      "url": "http://127.0.0.1:8765",
      "autostart": "on_demand",          // "on_demand" | "session" | "off"
      "python": "~/.aar/openjev/.venv/Scripts/python.exe",
      "bits": 4,
      "device": "cuda:0",
      "cuda_visible_devices": null,
      "idle_timeout": 1800,
      "tools": ["nli_check", "nli_rerank", "nli_grade"]
    }

``AAR_OPENJEV_CONFIG`` points at a different config file and ``AAR_OPENJEV_URL``
overrides ``url``.

Standalone usage — copy the ``aar_ext_openjev`` directory into
``~/.aar/extensions/`` and it will be loaded automatically.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

SERVER_SCRIPT = Path(__file__).with_name("server.py")
ALL_TOOLS = ("nli_check", "nli_rerank", "nli_grade")
MAX_ITEMS = 32
RERANK_HYPOTHESIS = "The correct answer is: {}"  # same phrasing as OpenJevCrossEncoder.rerank


def _default_python() -> str:
    venv = Path.home() / ".aar" / "openjev" / ".venv"
    return str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OpenJevConfig:
    """Extension settings, loaded from ``~/.aar/openjev.json``."""

    url: str = "http://127.0.0.1:8765"
    autostart: str = "on_demand"  # "on_demand" | "session" | "off"
    python: str = field(default_factory=_default_python)
    model: str = "AlexWortega/openjev"
    subfolder: str = "qwen3.5-4b-nli"
    bits: int = 4
    device: str = "cuda:0"
    cuda_visible_devices: str | None = None
    max_length: int = 2048
    batch_size: int = 4
    idle_timeout: float = 1800.0
    startup_timeout: float = 240.0
    request_timeout: float = 120.0
    log_file: str = "~/.aar/openjev/server.log"
    tools: list[str] = field(default_factory=lambda: list(ALL_TOOLS))

    @classmethod
    def load(cls, path: Path | None = None) -> OpenJevConfig:
        env_path = os.environ.get("AAR_OPENJEV_CONFIG")
        path = path or (Path(env_path) if env_path else Path.home() / ".aar" / "openjev.json")
        raw: dict[str, Any] = {}
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown openjev config keys in {path}: {', '.join(unknown)}")
        cfg = cls(**raw)
        if url := os.environ.get("AAR_OPENJEV_URL"):
            cfg.url = url
        if cfg.autostart not in ("on_demand", "session", "off"):
            raise ValueError(f"autostart must be on_demand, session or off, got {cfg.autostart!r}")
        return cfg

    @property
    def python_path(self) -> Path:
        return Path(self.python).expanduser()

    @property
    def log_path(self) -> Path:
        return Path(self.log_file).expanduser()


# ---------------------------------------------------------------------------
# Server management + HTTP client
# ---------------------------------------------------------------------------


class OpenJevUnavailable(RuntimeError):
    """The NLI server is not reachable and could not be started."""


class OpenJevClient:
    """Talks to the openjev server and starts it on demand.

    *transport* / *async_transport* exist for tests (``httpx.MockTransport``).
    """

    def __init__(
        self,
        config: OpenJevConfig,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._async_transport = async_transport
        self._proc: subprocess.Popen[bytes] | None = None
        self._start_lock = asyncio.Lock()

    # -- sync (slash-commands) ----------------------------------------------

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

    def shutdown(self) -> bool:
        try:
            with self._client(5.0) as c:
                c.post("/shutdown").raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def predict_sync(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        with self._client(self.config.request_timeout) as c:
            r = c.post("/predict", json=_pairs_payload(pairs))
            _raise_for_status(r)
            return r.json()

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
                    raise OpenJevUnavailable(
                        f"openjev server is not running at {self.config.url} "
                        "(autostart is off — run /openjev start)"
                    )
                self.start()
            deadline = time.monotonic() + self.config.startup_timeout
            while True:
                state = await self.ahealth()
                if state is not None and state.get("status") == "ready":
                    return state
                if state is not None and state.get("status") == "error":
                    raise OpenJevUnavailable(f"openjev failed to load: {state.get('error')}")
                if state is None and self._proc is not None and self._proc.poll() is not None:
                    raise OpenJevUnavailable(
                        f"openjev server exited with code {self._proc.returncode}; "
                        f"see {self.config.log_path}\n{_log_tail(self.config.log_path)}"
                    )
                if time.monotonic() > deadline:
                    raise OpenJevUnavailable(
                        f"openjev is still loading after {self.config.startup_timeout:.0f}s; "
                        f"try again shortly (log: {self.config.log_path})"
                    )
                await asyncio.sleep(1.0)

    async def predict(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        await self.ensure_ready()
        async with self._async_client(self.config.request_timeout) as c:
            r = await c.post("/predict", json=_pairs_payload(pairs))
            _raise_for_status(r)
            return r.json()

    # -- process --------------------------------------------------------------

    def server_command(self) -> list[str]:
        cfg = self.config
        parts = urlsplit(cfg.url)
        return [
            str(cfg.python_path),
            str(SERVER_SCRIPT),
            "--host", parts.hostname or "127.0.0.1",
            "--port", str(parts.port or 80),
            "--model", cfg.model,
            "--subfolder", cfg.subfolder,
            "--bits", str(cfg.bits),
            "--device", cfg.device,
            "--max-length", str(cfg.max_length),
            "--batch-size", str(cfg.batch_size),
            "--idle-timeout", str(cfg.idle_timeout),
        ]  # fmt: skip

    def start(self) -> None:
        """Launch the server detached, so it outlives this aar process.

        The command is fixed by the user's config — nothing model-controlled is
        executed.  The server frees its VRAM itself after ``idle_timeout``.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        cfg = self.config
        if not cfg.python_path.is_file():
            raise OpenJevUnavailable(
                f"openjev server interpreter not found: {cfg.python_path} — "
                "create the venv (see the aar-ext-openjev README) or set 'python' "
                "in ~/.aar/openjev.json"
            )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if cfg.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
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


def _pairs_payload(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    return {"pairs": [{"premise": p, "hypothesis": h} for p, h in pairs]}


def _raise_for_status(r: httpx.Response) -> None:
    if r.status_code == 503:
        raise OpenJevUnavailable(f"openjev not ready: {r.text}")
    r.raise_for_status()


def _log_tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

VERDICT = {"entailment": "SUPPORTED", "contradiction": "CONTRADICTED", "neutral": "NOT ESTABLISHED"}


def _as_dicts(result: dict[str, Any]) -> list[dict[str, float]]:
    labels = result["labels"]
    return [dict(zip(labels, row)) for row in result["probs"]]


def _probs_str(p: dict[str, float]) -> str:
    return " · ".join(f"{k} {p[k]:.2f}" for k in ("entailment", "contradiction", "neutral"))


def _clip(text: str, n: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def format_check(claims: list[str], result: dict[str, Any]) -> str:
    rows = _as_dicts(result)
    counts = {"entailment": 0, "contradiction": 0, "neutral": 0}
    lines = []
    for i, (claim, p) in enumerate(zip(claims, rows), 1):
        label = max(p, key=p.__getitem__)
        counts[label] += 1
        lines.append(f'{i}. {VERDICT[label]} ({_probs_str(p)}) — "{_clip(claim)}"')
    summary = (
        f"{counts['entailment']} supported, {counts['contradiction']} contradicted, "
        f"{counts['neutral']} not established by the premise"
    )
    if any(result.get("truncated", [])):
        summary += " (premise was truncated to fit the model's context)"
    return summary + "\n" + "\n".join(lines)


def format_rerank(options: list[str], result: dict[str, Any]) -> str:
    rows = _as_dicts(result)
    order = sorted(range(len(options)), key=lambda i: rows[i]["entailment"], reverse=True)
    lines = [
        f"{rank}. [{i}] P(entailment)={rows[i]['entailment']:.2f} "
        f"contradiction={rows[i]['contradiction']:.2f} — {_clip(options[i])}"
        for rank, i in enumerate(order, 1)
    ]
    return f"best option: [{order[0]}] {_clip(options[order[0]])}\n" + "\n".join(lines)


GRADE = {
    "entailment": "CORRECT — consistent with the reference",
    "contradiction": "INCORRECT — contradicts the reference",
    "neutral": "UNVERIFIED — the reference neither confirms nor refutes it",
}


def format_grade(result: dict[str, Any]) -> str:
    p = _as_dicts(result)[0]
    label = max(p, key=p.__getitem__)
    return f"{GRADE[label]} ({_probs_str(p)})"


# ---------------------------------------------------------------------------
# Extension entry-point
# ---------------------------------------------------------------------------


def register(
    api: Any, config: OpenJevConfig | None = None, client: OpenJevClient | None = None
) -> None:
    """Register the openjev tools and the ``/openjev`` command."""

    cfg = config or OpenJevConfig.load()
    jev = client or OpenJevClient(cfg)
    enabled = [t for t in cfg.tools if t in ALL_TOOLS]

    async def _run(pairs: list[tuple[str, str]]) -> dict[str, Any] | str:
        try:
            return await jev.predict(pairs)
        except (OpenJevUnavailable, httpx.HTTPError) as exc:
            return f"openjev unavailable: {exc}"

    # -- tools ----------------------------------------------------------------

    if "nli_check" in enabled:

        @api.tool(
            name="nli_check",
            description=(
                "Verify claims against a source text with a local NLI model. For each claim "
                "returns SUPPORTED (entailed by the premise), CONTRADICTED, or NOT ESTABLISHED, "
                "with probabilities. Use it to fact-check statements against file contents, "
                "docs or tool output before asserting them."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "premise": {
                        "type": "string",
                        "description": "Source text treated as ground truth",
                    },
                    "claims": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_ITEMS,
                        "description": "Short, self-contained statements to verify",
                    },
                },
                "required": ["premise", "claims"],
            },
            side_effects=["network"],
        )
        async def nli_check(premise: str, claims: list[str]) -> str:
            result = await _run([(premise, c) for c in claims])
            return result if isinstance(result, str) else format_check(claims, result)

    if "nli_rerank" in enabled:

        @api.tool(
            name="nli_rerank",
            description=(
                "Rank candidate answers to a question (optionally with context in the question "
                "text) by how strongly a local NLI model judges each to be the correct answer."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question, plus any context"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": MAX_ITEMS,
                    },
                },
                "required": ["question", "options"],
            },
            side_effects=["network"],
        )
        async def nli_rerank(question: str, options: list[str]) -> str:
            result = await _run([(question, RERANK_HYPOTHESIS.format(o)) for o in options])
            return result if isinstance(result, str) else format_rerank(options, result)

    if "nli_grade" in enabled:

        @api.tool(
            name="nli_grade",
            description=(
                "Grade a candidate answer against a reference answer with a local NLI model: "
                "CORRECT, INCORRECT or UNVERIFIED."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "reference": {"type": "string", "description": "Known-good answer"},
                    "answer": {"type": "string", "description": "Answer to grade"},
                },
                "required": ["question", "reference", "answer"],
            },
            side_effects=["network"],
        )
        async def nli_grade(question: str, reference: str, answer: str) -> str:
            premise = f"{question}\nReference answer: {reference}"  # OpenJevCrossEncoder.grade
            result = await _run([(premise, f"Answer: {answer}")])
            return result if isinstance(result, str) else format_grade(result)

    if enabled:
        api.append_system_prompt(
            "A local NLI verifier is available ("
            + ", ".join(enabled)
            + "). When you state facts derived from files, docs or tool output, you can "
            "check them against the source text with nli_check. The verifier judges only "
            "what the premise says, not world knowledge; keep claims short and specific."
        )

    # -- lifecycle ------------------------------------------------------------

    @api.on("session_start")
    async def on_session_start(event: Any, ctx: Any) -> None:
        if cfg.autostart != "session" or not enabled:
            return
        if await jev.ahealth() is None:
            try:
                jev.start()
                ctx.logger.info("openjev: starting server (log: %s)", cfg.log_path)
            except OpenJevUnavailable as exc:
                ctx.logger.warning("openjev: %s", exc)

    # -- slash-command --------------------------------------------------------

    @api.command("openjev", description="openjev NLI server: status | start | stop | check P => H")
    def openjev_command(args: str, ctx: Any) -> str:
        sub, _, rest = (args or "").strip().partition(" ")
        sub = sub.lower() or "status"

        if sub == "status":
            state = jev.health()
            lines = [f"url:     {cfg.url}", f"tools:   {', '.join(enabled) or '(none)'}"]
            if state is None:
                lines.insert(0, "openjev: not running")
                lines.append(f"autostart: {cfg.autostart}  python: {cfg.python_path}")
            else:
                lines.insert(0, f"openjev: {state.get('status')}")
                lines += [
                    f"model:   {state.get('model')} ({state.get('bits')}-bit)",
                    f"device:  {state.get('device')}",
                    f"pid:     {state.get('pid')}  load: {state.get('load_seconds')}s",
                ]
                if state.get("error"):
                    lines.append(f"error:   {state['error']}")
            lines.append(f"log:     {cfg.log_path}")
            return "\n".join(lines)

        if sub == "start":
            if jev.health() is not None:
                return "openjev: already running — /openjev status"
            try:
                jev.start()
            except OpenJevUnavailable as exc:
                return f"openjev: {exc}"
            return f"openjev: starting on {cfg.device} ({cfg.bits}-bit); log: {cfg.log_path}"

        if sub == "stop":
            return "openjev: stopping" if jev.shutdown() else "openjev: not running"

        if sub == "check":
            premise, sep, hypothesis = rest.partition("=>")
            if not sep or not premise.strip() or not hypothesis.strip():
                return "usage: /openjev check <premise> => <hypothesis>"
            state = jev.health()
            if state is None or state.get("status") != "ready":
                return "openjev: server not ready — /openjev start, then /openjev status"
            try:
                result = jev.predict_sync([(premise.strip(), hypothesis.strip())])
            except (OpenJevUnavailable, httpx.HTTPError) as exc:
                return f"openjev: {exc}"
            return format_check([hypothesis.strip()], result)

        return "usage: /openjev [status | start | stop | check <premise> => <hypothesis>]"
