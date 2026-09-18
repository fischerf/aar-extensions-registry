"""Flappy Bird played zero-shot by the openjev NLI cross-encoder, live in the terminal.

Every frame the game state is written out as text (the *premise*) and the model scores two
*hypotheses*, one per action. The action whose hypothesis has the higher P(entailment) wins.
No training, no game-specific head: just the NLI model the aar extension already serves.

The trick (from the openjev authors): hypotheses must be **statements about the state**, not
action names. "The correct action is: flap" barely works; "The offset relative to the gap centre
is negative" (→ flap) plays the game.

Game physics, state rendering and the oracle are taken from openjev's ``code/flappy.py``
(https://huggingface.co/AlexWortega/openjev, MIT). Runs in aar's environment (httpx + rich) against
the running openjev server — it never loads the model itself.

    python flappy_nli.py                         # live game, "sign" hypotheses
    python flappy_nli.py --hyp action            # the phrasing that fails, for comparison
    python flappy_nli.py --policy oracle         # the hand-written heuristic (instant)
    python flappy_nli.py --baselines --episodes 5 --no-render
    python flappy_nli.py --replay flappy.json    # replay for openjev's flappy_video.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ----------------------------------------------------------------------------- game (openjev)
GRAVITY, FLAP_V, SPEED = 0.003, 0.015, 0.015
PIPE_EVERY, GAP, PIPE_HW, BIRD_X, BIRD_HW = 0.45, 0.28, 0.04, 0.2, 0.03


class Flappy:
    """Deterministic Flappy Bird on a unit square; score = pipes passed."""

    def __init__(self, seed: int = 0, max_steps: int = 2000) -> None:
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> dict[str, float]:
        self.y, self.vy, self.t, self.score, self.done = 0.5, 0.0, 0, 0, False
        self.crashed = False
        self.pipes = [[1.2, self._gap()]]  # [x, gap_lo]
        return self.state()

    def _gap(self) -> float:
        return self.rng.uniform(0.15, 0.85 - GAP)

    def next_pipe(self) -> tuple[float, float]:
        for x, lo in self.pipes:
            if x + PIPE_HW >= BIRD_X - BIRD_HW:
                return x, lo
        return self.pipes[-1][0], self.pipes[-1][1]

    def step(self, flap: bool) -> dict[str, float]:
        if self.done:
            return self.state()
        self.vy = FLAP_V if flap else self.vy - GRAVITY
        self.y += self.vy
        self.t += 1
        for p in self.pipes:
            p[0] -= SPEED
        if self.pipes[-1][0] < 1.2 - PIPE_EVERY:
            self.pipes.append([self.pipes[-1][0] + PIPE_EVERY, self._gap()])
        if self.pipes[0][0] + PIPE_HW < BIRD_X - BIRD_HW:
            self.pipes.pop(0)
            self.score += 1
        x, lo = self.next_pipe()
        hit = self.y <= 0 or self.y >= 1
        if abs(x - BIRD_X) < PIPE_HW + BIRD_HW and not (lo < self.y < lo + GAP):
            hit = True
        self.crashed = hit
        if hit or self.t >= self.max_steps:
            self.done = True
        return self.state()

    def state(self) -> dict[str, float]:
        x, lo = self.next_pipe()
        return {
            "y": self.y,
            "vy": self.vy,
            "dx": x - BIRD_X,
            "lo": lo,
            "hi": lo + GAP,
            "score": self.score,
            "t": self.t,
        }


def oracle(s: dict[str, float], margin: float = 0.03, lookahead: int = 3) -> bool:
    """Flap if the height a few frames ahead falls below the gap centre (minus a margin)."""
    target = (s["lo"] + s["hi"]) / 2 - margin
    y_pred = s["y"] + s["vy"] * lookahead - GRAVITY * lookahead * (lookahead - 1) / 2
    return y_pred < target


def render_text(s: dict[str, float]) -> str:
    """The premise the model reads (openjev's ``base`` prompt)."""
    if s["lo"] < s["y"] < s["hi"]:
        pos = "inside the gap"
    else:
        pos = "above the gap" if s["y"] >= s["hi"] else "below the gap"
    move = "rising" if s["vy"] > 0 else "falling"
    centre = (s["lo"] + s["hi"]) / 2
    return (
        f"Flappy Bird. The bird is at height {s['y']:.2f} (0 = ground, 1 = ceiling) and is {move} "
        f"with vertical velocity {s['vy']:+.3f} per frame; gravity pulls it down every frame and "
        f"flapping pushes it up. The next pipe is {s['dx']:.2f} ahead; its gap spans heights "
        f"{s['lo']:.2f} to {s['hi']:.2f} (centre {centre:.2f}). The bird is currently {pos}, "
        f"{s['y'] - centre:+.2f} relative to the gap centre. The bird must fly through the gap "
        f"without touching the pipe, the ground or the ceiling."
    )


# (hypothesis for "flap", hypothesis for "do nothing")
HYPOTHESES: dict[str, tuple[str, str]] = {
    "sign": (
        "The offset relative to the gap centre is negative.",
        "The offset relative to the gap centre is positive.",
    ),
    "position": (
        "The bird is below the centre of the gap.",
        "The bird is above the centre of the gap.",
    ),
    "action": ("The correct action is: flap", "The correct action is: do nothing"),
}

# ----------------------------------------------------------------------------- policies


@dataclass
class Decision:
    flap: bool
    probs: tuple[float, float] | None = None  # P(entailment) for (flap, do nothing)
    latency_ms: float = 0.0


Policy = Callable[[dict[str, float]], Decision]


class NliPolicy:
    """Zero-shot: argmax P(entailment) over the two action hypotheses, via the openjev server."""

    def __init__(self, url: str, hyp: str = "sign", client: httpx.Client | None = None) -> None:
        self.hyp_flap, self.hyp_wait = HYPOTHESES[hyp]
        self.client = client or httpx.Client(base_url=url, timeout=120)

    def __call__(self, s: dict[str, float]) -> Decision:
        premise = render_text(s)
        t0 = time.perf_counter()
        r = self.client.post(
            "/predict",
            json={
                "pairs": [
                    {"premise": premise, "hypothesis": self.hyp_flap},
                    {"premise": premise, "hypothesis": self.hyp_wait},
                ]
            },
        )
        r.raise_for_status()
        body = r.json()
        ent = body["labels"].index("entailment")
        p_flap, p_wait = (row[ent] for row in body["probs"])
        return Decision(p_flap > p_wait, (p_flap, p_wait), (time.perf_counter() - t0) * 1000)


def make_baseline(name: str, seed: int = 0) -> Policy:
    rng = random.Random(seed)
    if name == "oracle":
        return lambda s: Decision(oracle(s))
    if name == "never":
        return lambda s: Decision(False)
    if name == "random":
        return lambda s: Decision(rng.random() < 0.1)
    raise ValueError(f"unknown policy {name!r}")


# ----------------------------------------------------------------------------- episode


@dataclass
class Episode:
    score: int = 0
    steps: int = 0
    crashed: bool = False
    agree: int = 0  # decisions matching the oracle
    latencies: list[float] = field(default_factory=list)
    replay: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        return self.agree / max(self.steps, 1)

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


def play(
    policy: Policy,
    seed: int = 1000,
    max_steps: int = 300,
    on_frame: Callable[[Flappy, dict[str, float], Decision, Episode], None] | None = None,
) -> Episode:
    """Turn-based: the game waits for every decision (the model is slower than real time)."""
    env = Flappy(seed=seed, max_steps=max_steps)
    s = env.reset()
    ep = Episode()
    while not env.done:
        d = policy(s)
        ep.agree += d.flap == oracle(s)
        if d.latency_ms:
            ep.latencies.append(d.latency_ms)
        ep.replay.append(
            {  # same fields as openjev's flappy.py --record-only (for flappy_video.py)
                "t": env.t,
                "y": round(env.y, 4),
                "pipes": [[round(x, 3), round(lo, 3)] for x, lo in env.pipes[:3]],
                "a": int(d.flap),
                "lat_ms": round(d.latency_ms, 1),
                "skipped": 0,
                "score": env.score,
                "probs": [round(p, 4) for p in d.probs] if d.probs else None,
            }
        )
        s = env.step(d.flap)
        ep.steps = env.t
        ep.score = env.score
        if on_frame is not None:
            on_frame(env, s, d, ep)
    ep.crashed = env.crashed
    return ep


# ----------------------------------------------------------------------------- terminal view


def draw_board(env: Flappy, rows: int = 18, cols: int = 54) -> list[str]:
    """ASCII frame: '>' bird, '█' pipes, x range [0, 1.2]."""
    grid = [[" "] * cols for _ in range(rows)]
    to_col = lambda x: int(round(x / 1.2 * (cols - 1)))  # noqa: E731
    for x, lo in env.pipes:
        for c in range(to_col(x - PIPE_HW), to_col(x + PIPE_HW) + 1):
            if not 0 <= c < cols:
                continue
            for r in range(rows):
                yy = 1 - (r + 0.5) / rows
                if not (lo < yy < lo + GAP):
                    grid[r][c] = "█"
    br = min(rows - 1, max(0, int((1 - env.y) * rows)))
    grid[br][to_col(BIRD_X)] = "✖" if env.crashed else ">"
    return ["".join(r) for r in grid]


def make_renderer(hyp: str | None, title: str) -> tuple[Any, Callable[..., None]]:
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    live = Live(refresh_per_second=30, transient=False)

    def bar(p: float, width: int = 30) -> str:
        n = int(round(p * width))
        return "█" * n + "·" * (width - n)

    def on_frame(env: Flappy, s: dict[str, float], d: Decision, ep: Episode) -> None:
        board = Text("\n".join(draw_board(env)), style="green")
        stats = Table.grid(padding=(0, 2))
        stats.add_row("score", f"[bold yellow]{ep.score}[/]", "frame", str(ep.steps))
        stats.add_row(
            "action",
            "[bold yellow]FLAP[/]" if d.flap else "[dim]do nothing[/]",
            "agrees with oracle",
            f"{ep.agreement:.0%}",
        )
        if ep.latencies:
            stats.add_row(
                "decision", f"{d.latency_ms:.0f} ms", "mean", f"{ep.mean_latency_ms:.0f} ms"
            )
        parts: list[Any] = [board, stats]
        if hyp and d.probs:
            h_flap, h_wait = HYPOTHESES[hyp]
            probs = Table.grid(padding=(0, 1))
            probs.add_row(
                "[yellow]flap[/]", bar(d.probs[0]), f"{d.probs[0]:.3f}", f"[dim]{h_flap}[/]"
            )
            probs.add_row(
                "[green]wait[/]", bar(d.probs[1]), f"{d.probs[1]:.3f}", f"[dim]{h_wait}[/]"
            )
            parts += [Text("P(entailment)", style="bold"), probs]
            parts.append(Text(render_text(s), style="dim", overflow="fold"))
        if env.done:
            end = "CRASHED" if env.crashed else "SURVIVED"
            parts.append(Text(f"{end} — score {ep.score} in {ep.steps} frames", style="bold red"))
        live.update(Panel(Group(*parts), title=title, border_style="cyan"))

    return live, on_frame


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8765", help="openjev server")
    ap.add_argument("--policy", default="nli", choices=["nli", "oracle", "random", "never"])
    ap.add_argument("--hyp", default="sign", choices=list(HYPOTHESES), help="hypothesis phrasing")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000, help="first episode seed")
    ap.add_argument("--max-steps", type=int, default=300, help="frames per episode (~2 pipes/60)")
    ap.add_argument("--baselines", action="store_true", help="also run oracle/random/never")
    ap.add_argument("--no-render", action="store_true", help="print results only")
    ap.add_argument("--replay", type=Path, help="write a replay JSON (openjev flappy_video.py)")
    args = ap.parse_args(argv)

    if args.policy == "nli":
        try:
            health = httpx.get(f"{args.url}/health", timeout=3).json()
        except httpx.HTTPError:
            print(f"openjev server not reachable at {args.url} — start it first", file=sys.stderr)
            return 1
        if health.get("status") != "ready":
            print(
                f"openjev server is {health.get('status')}: {health.get('error')}", file=sys.stderr
            )
            return 1

    names = ([args.policy] if args.policy != "nli" else []) + (
        ["oracle", "random", "never"] if args.baselines else []
    )
    runs: list[tuple[str, Policy, str | None]] = [
        (n, make_baseline(n, args.seed), None) for n in dict.fromkeys(names)
    ]
    if args.policy == "nli":
        runs.insert(0, (f"nli:{args.hyp}", NliPolicy(args.url, args.hyp), args.hyp))

    summary: dict[str, dict[str, Any]] = {}
    for name, policy, hyp in runs:
        episodes = []
        for i in range(args.episodes):
            seed = args.seed + i
            if args.no_render:
                ep = play(policy, seed, args.max_steps)
            else:
                title = f"openjev plays Flappy Bird · {name} · episode {i + 1}/{args.episodes}"
                live, on_frame = make_renderer(hyp, title)
                with live:
                    ep = play(policy, seed, args.max_steps, on_frame)
                    time.sleep(0.8)
            episodes.append(ep)
            lat = f"  {ep.mean_latency_ms:5.0f} ms/decision" if ep.latencies else ""
            print(
                f"{name:14s} seed {seed}  score {ep.score:3d}  frames {ep.steps:4d}  "
                f"{'crashed ' if ep.crashed else 'survived'}  oracle agreement {ep.agreement:4.0%}{lat}"
            )
        best = max(episodes, key=lambda e: e.score)
        key = "nli" if name.startswith("nli") else name
        summary[key] = {
            "mean_score": sum(e.score for e in episodes) / len(episodes),
            "max_score": best.score,
            "replay": best.replay,
            "replay_score": best.score,
        }
        if len(episodes) > 1:
            print(f"{'':14s} mean score {summary[key]['mean_score']:.1f}  max {best.score}")

    if args.replay:
        args.replay.write_text(
            json.dumps({"args": vars(args) | {"replay": str(args.replay)}, "results": summary}),
            encoding="utf-8",
        )
        print(f"replay written to {args.replay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
