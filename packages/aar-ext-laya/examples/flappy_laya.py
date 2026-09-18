"""Flappy Bird played zero-shot by the Laya decision model, live in the terminal.

Every frame the game state is written out as text and Laya answers one typed question
about it; the answer picks the action. No training, no game-specific head: just the
decision model the aar extension already serves.

This is a port of ``aar-ext-openjev/examples/flappy_nli.py``, and the interesting part
is what changes. openjev is an **NLI cross-encoder**, so it can only score
(premise, hypothesis) pairs — asking it directly for an action barely works
("The correct action is: flap" agreed with the oracle 3/24), and the openjev authors'
trick is to ask about the *state* instead ("The offset relative to the gap centre is
negative" → flap), which gets 21/24.

Laya has a native ``choice`` type, so unlike openjev it *can* be asked the action
question directly. It still does not work — but the *descriptive* framing does, and
plays as well as the hand-written oracle. Two things have to be right.

**1. The question must be descriptive, not prescriptive.** ``--q`` switches framing:

    action   choice{flap, wait} with neutral option descriptions   (the direct ask)
    hint     choice{flap, wait} whose descriptions state the rule  (criteria matter?)
    sign     choice{negative, positive} about the offset           (openjev's framing)
    noul     one calibrated P(should the bird flap now?)

**2. The state text must not contain words that leak into the answer.** ``--state``
switches wording: ``full`` is openjev's original prompt, ``terse`` drops velocity.

Measured on CPU, seed 1001, 400 frames (``--compare``)::

    framing  state    result                                     agreement   latency
    sign     terse    survived 400 frames, score 11              86%         279 ms
    sign     full     CRASHED into pipe at frame 63, score 0     86%         419 ms
    action   terse    CRASHED into ceiling at frame 34, score 0   0%         238 ms
    hint     terse    CRASHED into ceiling at frame 34, score 0   0%         301 ms
    noul     terse    CRASHED into ceiling at frame 34, score 0   0%         259 ms
    oracle   -        survived 400 frames, score 11             100%           -

``sign``/``terse`` matches the oracle's score exactly.

Why ``action``, ``hint`` and ``noul`` fail: they are state-blind. ``action`` answers
"flap" whether the bird is below the gap or above it, and grows *more* certain when it
is above (0.536 -> 0.640); ``noul`` is flat at 0.946 vs 0.945. Always flapping puts the
bird into the ceiling on frame 34, which is what all three do. Writing the decision rule
into the option descriptions (``hint``) does not rescue it.

Why ``sign`` needs ``terse``: ``render_text`` says the bird is "rising" or "falling", and
the model reads "falling" as "negative". On a fixed offset of +0.51 (bird far *above* the
gap) p(negative) is 0.079 while rising but 0.483 while falling — a swing larger than the
offset itself produces. So every time the bird began to descend it flapped, pinning it
above the gap until a pipe arrived. Dropping five words of velocity wording makes the
answer velocity-invariant, and shortens the state, which is also the main latency driver.

Note that ``sign``/``terse`` still disagrees with the oracle on 14% of frames and scores
11 anyway: the disagreements are near-centre oscillations the game forgives, not the
systematic bias the ``full`` state induced. Oracle agreement alone is a poor metric here.

Also note it takes **71 frames to pass the first pipe** (the bird sits at x=0.2 and pipes
spawn at x=1.2, closing at 0.015/frame), then 30 frames each. A run shorter than that
scores 0 no matter how well it flies, so ``--max-steps`` below ~100 measures nothing.

Two lessons, both of which generalise past this toy — see "Writing good questions" in the
README: **ask what the state is, not what to do about it**, and **put nothing in the state
you do not want answered**.

Game physics, state rendering and the oracle are taken from openjev's ``code/flappy.py``
(https://huggingface.co/AlexWortega/openjev, MIT). Runs in aar's environment (httpx +
rich) against the running laya server — it never loads the model itself.

    python flappy_laya.py --q sign                 # the framing that plays (score 11)
    python flappy_laya.py --q sign --state full    # the confound, crashes at frame 63
    python flappy_laya.py                          # the direct ask, ceiling on frame 34
    python flappy_laya.py --policy oracle          # the hand-written heuristic (instant)
    python flappy_laya.py --compare --episodes 3 --no-render   # all framings + baselines
    python flappy_laya.py --replay flappy.json     # replay for openjev's flappy_video.py
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
    """The state text Laya reads (openjev's ``base`` prompt)."""
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


def render_terse(s: dict[str, float]) -> str:
    """Height, gap and offset only — no velocity, no direction word.

    ``render_text`` says the bird is "rising" or "falling", and Laya reads "falling" as
    "negative": on an offset of +0.51 it answers p(negative) 0.079 while rising and 0.483
    while falling, a swing larger than the offset itself produces. Dropping the word makes
    the answer velocity-invariant (0.047 vs 0.982 on the same two offsets) and shortens
    the state, which is also the main latency driver.
    """
    if s["lo"] < s["y"] < s["hi"]:
        pos = "inside the gap"
    else:
        pos = "above the gap" if s["y"] >= s["hi"] else "below the gap"
    centre = (s["lo"] + s["hi"]) / 2
    return (
        f"Flappy Bird. The bird is at height {s['y']:.2f} (0 = ground, 1 = ceiling). "
        f"The next pipe's gap spans heights {s['lo']:.2f} to {s['hi']:.2f} "
        f"(centre {centre:.2f}). The bird is currently {pos}, "
        f"{s['y'] - centre:+.2f} relative to the gap centre."
    )


STATES: dict[str, Callable[[dict[str, float]], str]] = {
    "full": render_text,  # openjev's original prompt
    "terse": render_terse,  # height + offset only
}


# ----------------------------------------------------------------------------- questions
#
# Each framing is one Laya question plus a rule for reading "flap" out of the answer.
# ``labels`` names the two sides for the UI; for ``noul`` the second side is 1 - P.

QUESTIONS: dict[str, dict[str, Any]] = {
    "action": {
        "question": {
            "type": "choice",
            "instructions": "What should the bird do on this frame?",
            "criteria": {
                "flap": "flap the wings now",
                "wait": "do nothing this frame",
            },
        },
        "labels": ("flap", "wait"),
    },
    "hint": {
        "question": {
            "type": "choice",
            "instructions": "What should the bird do on this frame?",
            "criteria": {
                "flap": "the bird is at or below the centre of the gap and needs to rise",
                "wait": "the bird is above the centre of the gap and should keep falling",
            },
        },
        "labels": ("flap", "wait"),
    },
    "sign": {
        "question": {
            "type": "choice",
            "instructions": "Is the bird's offset relative to the gap centre negative or positive?",
            "criteria": {
                "negative": "the offset relative to the gap centre is negative",
                "positive": "the offset relative to the gap centre is positive",
            },
        },
        "labels": ("negative", "positive"),
    },
    "noul": {
        "question": {"type": "noul", "instructions": "Should the bird flap on this frame?"},
        "labels": ("flap", "wait"),
    },
}

QKEY = "decision"


# ----------------------------------------------------------------------------- policies


@dataclass
class Decision:
    flap: bool
    probs: tuple[float, float] | None = None  # (P flap-side, P wait-side)
    confidence: float | None = None
    latency_ms: float = 0.0


Policy = Callable[[dict[str, float]], Decision]


class LayaPolicy:
    """Zero-shot: one typed question per frame, answered by the laya server."""

    def __init__(
        self,
        url: str,
        q: str = "action",
        state: str = "terse",
        client: httpx.Client | None = None,
    ) -> None:
        spec = QUESTIONS[q]
        self.q = q
        self.question = spec["question"]
        self.labels: tuple[str, str] = spec["labels"]
        self.render = STATES[state]
        self.client = client or httpx.Client(base_url=url, timeout=120)

    def __call__(self, s: dict[str, float]) -> Decision:
        t0 = time.perf_counter()
        r = self.client.post(
            "/predict",
            json={"state": self.render(s), "questions": {QKEY: self.question}},
        )
        r.raise_for_status()
        answer = r.json()["answers"][QKEY]
        latency = (time.perf_counter() - t0) * 1000

        if answer.get("type") == "noul" or "noul" in answer:
            p = float(answer["noul"])
            return Decision(p >= 0.5, (p, 1.0 - p), answer.get("confidence"), latency)

        probs = answer.get("probabilities") or {}
        a, b = self.labels
        p_a, p_b = float(probs.get(a, 0.0)), float(probs.get(b, 0.0))
        # Fall back to the reported choice when probabilities are missing.
        flap = p_a > p_b if (p_a or p_b) else answer.get("choice") == a
        return Decision(flap, (p_a, p_b), answer.get("confidence"), latency)


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


def make_renderer(
    q: str | None, title: str, state: str = "terse"
) -> tuple[Any, Callable[..., None]]:
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
        if d.confidence is not None:
            stats.add_row("confidence", f"{d.confidence:.2f}", "", "")
        parts: list[Any] = [board, stats]
        if q and d.probs:
            a, b = QUESTIONS[q]["labels"]
            probs = Table.grid(padding=(0, 1))
            probs.add_row(f"[yellow]{a}[/]", bar(d.probs[0]), f"{d.probs[0]:.3f}")
            probs.add_row(f"[green]{b}[/]", bar(d.probs[1]), f"{d.probs[1]:.3f}")
            head = "P(true)" if q == "noul" else f"laya · {q} · P(option)"
            parts += [Text(head, style="bold"), probs]
            parts.append(Text(STATES[state](s), style="dim", overflow="fold"))
        if env.done:
            end = "CRASHED" if env.crashed else "SURVIVED"
            parts.append(Text(f"{end} — score {ep.score} in {ep.steps} frames", style="bold red"))
        live.update(Panel(Group(*parts), title=title, border_style="cyan"))

    return live, on_frame


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8770", help="laya server")
    ap.add_argument("--policy", default="laya", choices=["laya", "oracle", "random", "never"])
    ap.add_argument("--q", default="action", choices=list(QUESTIONS), help="question framing")
    ap.add_argument("--state", default="terse", choices=list(STATES), help="state wording")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000, help="first episode seed")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="frames per episode (first pipe passes at frame 71, then one per 30)",
    )
    ap.add_argument("--baselines", action="store_true", help="also run oracle/random/never")
    ap.add_argument("--compare", action="store_true", help="run every framing plus the baselines")
    ap.add_argument("--no-render", action="store_true", help="print results only")
    ap.add_argument("--replay", type=Path, help="write a replay JSON (openjev flappy_video.py)")
    args = ap.parse_args(argv)

    uses_model = args.policy == "laya" or args.compare
    if uses_model:
        try:
            health = httpx.get(f"{args.url}/health", timeout=3).json()
        except httpx.HTTPError:
            print(f"laya server not reachable at {args.url} — start it first", file=sys.stderr)
            return 1
        if health.get("status") != "ready":
            print(f"laya server is {health.get('status')}: {health.get('error')}", file=sys.stderr)
            return 1

    runs: list[tuple[str, Policy, str | None]] = []
    if args.compare:
        runs += [(f"laya:{q}", LayaPolicy(args.url, q, args.state), q) for q in QUESTIONS]
    elif args.policy == "laya":
        runs.append((f"laya:{args.q}", LayaPolicy(args.url, args.q, args.state), args.q))
    baselines = ([args.policy] if args.policy != "laya" else []) + (
        ["oracle", "random", "never"] if (args.baselines or args.compare) else []
    )
    runs += [(n, make_baseline(n, args.seed), None) for n in dict.fromkeys(baselines)]

    summary: dict[str, dict[str, Any]] = {}
    for name, policy, q in runs:
        episodes = []
        for i in range(args.episodes):
            seed = args.seed + i
            if args.no_render:
                ep = play(policy, seed, args.max_steps)
            else:
                title = (
                    f"laya plays Flappy Bird · {name} · {args.state} state · "
                    f"episode {i + 1}/{args.episodes}"
                )
                live, on_frame = make_renderer(q, title, args.state)
                with live:
                    ep = play(policy, seed, args.max_steps, on_frame)
                    time.sleep(0.8)
            episodes.append(ep)
            lat = f"  {ep.mean_latency_ms:5.0f} ms/decision" if ep.latencies else ""
            print(
                f"{name:14s} seed {seed}  score {ep.score:3d}  frames {ep.steps:4d}  "
                f"{'crashed ' if ep.crashed else 'survived'}  "
                f"oracle agreement {ep.agreement:4.0%}{lat}"
            )
        best = max(episodes, key=lambda e: e.score)
        summary[name] = {
            "mean_score": sum(e.score for e in episodes) / len(episodes),
            "mean_agreement": sum(e.agreement for e in episodes) / len(episodes),
            "max_score": best.score,
            "replay": best.replay,
            "replay_score": best.score,
        }
        if len(episodes) > 1:
            print(
                f"{'':14s} mean score {summary[name]['mean_score']:.1f}  max {best.score}  "
                f"mean agreement {summary[name]['mean_agreement']:.0%}"
            )

    if args.replay:
        args.replay.write_text(
            json.dumps({"args": vars(args) | {"replay": str(args.replay)}, "results": summary}),
            encoding="utf-8",
        )
        print(f"replay written to {args.replay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
