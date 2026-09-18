"""Tests for examples/flappy_nli.py (game logic + NLI policy over a mocked server)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

_PATH = Path(__file__).parent.parent / "examples" / "flappy_nli.py"


@pytest.fixture(scope="module")
def flappy():
    spec = importlib.util.spec_from_file_location("flappy_nli", _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["flappy_nli"] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def test_oracle_survives_and_baselines_crash(flappy) -> None:
    oracle = flappy.play(flappy.make_baseline("oracle"), seed=1001, max_steps=900)
    never = flappy.play(flappy.make_baseline("never"), seed=1001, max_steps=900)
    assert not oracle.crashed and oracle.score > 20
    assert oracle.agreement == 1.0
    assert never.crashed and never.score == 0


def test_render_text_reports_offset(flappy) -> None:
    s = {"y": 0.40, "vy": -0.003, "dx": 0.5, "lo": 0.36, "hi": 0.64, "score": 0, "t": 1}
    text = flappy.render_text(s)
    assert "-0.10 relative to the gap centre" in text
    assert "falling" in text and "inside the gap" in text


def test_nli_policy_flaps_on_negative_offset(flappy) -> None:
    """A fake 'perfect NLI' server: entails the sign hypothesis matching the premise."""
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        probs = []
        for pair in body["pairs"]:
            negative = " -0." in pair["premise"].split("relative to the gap centre")[0][-8:]
            says_negative = "negative" in pair["hypothesis"]
            ent = 0.9 if negative == says_negative else 0.05
            probs.append([1 - ent - 0.02, ent, 0.02])
        labels = ["contradiction", "entailment", "neutral"]
        return httpx.Response(200, json={"labels": labels, "probs": probs})

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    policy = flappy.NliPolicy("http://test", "sign", client=client)

    below = {"y": 0.40, "vy": 0.0, "dx": 0.5, "lo": 0.36, "hi": 0.64, "score": 0, "t": 0}
    above = dict(below, y=0.60)
    d_below, d_above = policy(below), policy(above)

    assert d_below.flap and d_below.probs[0] > d_below.probs[1]
    assert not d_above.flap
    hyps = [p["hypothesis"] for p in sent[0]["pairs"]]
    assert hyps == list(flappy.HYPOTHESES["sign"])

    ep = flappy.play(policy, seed=1001, max_steps=300)
    assert ep.score > 0
    assert ep.replay[0].keys() >= {"t", "y", "pipes", "a", "probs", "score"}


def test_draw_board_marks_bird_and_pipes(flappy) -> None:
    env = flappy.Flappy(seed=1)
    for _ in range(40):
        env.step(env.t % 12 == 0)
    board = flappy.draw_board(env)
    assert len(board) == 18
    assert any(">" in row for row in board)
    assert any("█" in row for row in board)
