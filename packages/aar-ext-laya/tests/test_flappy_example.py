"""Tests for examples/flappy_laya.py (game logic + laya policy over a mocked server)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

_PATH = Path(__file__).parent.parent / "examples" / "flappy_laya.py"


@pytest.fixture(scope="module")
def flappy():
    spec = importlib.util.spec_from_file_location("flappy_laya", _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["flappy_laya"] = module  # dataclasses resolve annotations via sys.modules
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


def _perfect_server(flappy, q: str):
    """A fake laya that answers each framing correctly from the rendered state."""
    sent: list[dict] = []
    a, b = flappy.QUESTIONS[q]["labels"]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        offset = body["state"].split("relative to the gap centre")[0][-8:]
        below = " -0." in offset
        spec = body["questions"][flappy.QKEY]
        if spec["type"] == "noul":
            answer = {"type": "noul", "noul": 0.9 if below else 0.1, "confidence": 0.9}
        else:
            hot = a if below else b
            probs = {name: (0.9 if name == hot else 0.1) for name in spec["criteria"]}
            answer = {
                "type": "choice",
                "choice": hot,
                "probabilities": probs,
                "confidence": 0.9,
            }
        return httpx.Response(200, json={"model": "laya", "answers": {flappy.QKEY: answer}})

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return flappy.LayaPolicy("http://test", q, client=client), sent


@pytest.mark.parametrize("q", ["action", "hint", "sign", "noul"])
def test_policy_flaps_when_below_the_centre(flappy, q: str) -> None:
    policy, sent = _perfect_server(flappy, q)

    below = {"y": 0.40, "vy": 0.0, "dx": 0.5, "lo": 0.36, "hi": 0.64, "score": 0, "t": 0}
    above = dict(below, y=0.60)
    d_below, d_above = policy(below), policy(above)

    assert d_below.flap and d_below.probs[0] > d_below.probs[1]
    assert not d_above.flap
    assert sent[0]["questions"][flappy.QKEY] == flappy.QUESTIONS[q]["question"]


@pytest.mark.parametrize("q", ["action", "sign", "noul"])
def test_policy_plays_a_full_episode(flappy, q: str) -> None:
    policy, _ = _perfect_server(flappy, q)
    ep = flappy.play(policy, seed=1001, max_steps=300)
    assert ep.score > 0
    assert ep.replay[0].keys() >= {"t", "y", "pipes", "a", "probs", "score"}


def test_policy_falls_back_to_choice_without_probabilities(flappy) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"answers": {flappy.QKEY: {"type": "choice", "choice": "flap"}}},
        )

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    policy = flappy.LayaPolicy("http://test", "action", client=client)
    s = {"y": 0.5, "vy": 0.0, "dx": 0.5, "lo": 0.36, "hi": 0.64, "score": 0, "t": 0}
    assert policy(s).flap


def test_every_framing_is_well_formed(flappy) -> None:
    for name, spec in flappy.QUESTIONS.items():
        q = spec["question"]
        assert q["type"] in ("choice", "noul"), name
        assert q["instructions"].strip()
        a, b = spec["labels"]
        if q["type"] == "choice":
            assert set(q["criteria"]) == {a, b}, name


def test_draw_board_marks_bird_and_pipes(flappy) -> None:
    env = flappy.Flappy(seed=1)
    for _ in range(40):
        env.step(env.t % 12 == 0)
    board = flappy.draw_board(env)
    assert len(board) == 18
    assert any(">" in row for row in board)
    assert any("█" in row for row in board)


def test_terse_state_drops_the_direction_word(flappy) -> None:
    """The word 'falling' swings p(negative) more than the offset does — see render_terse."""
    falling = {"y": 0.827, "vy": -0.003, "dx": 0.3, "lo": 0.175, "hi": 0.455, "score": 0, "t": 1}
    rising = dict(falling, vy=+0.003)

    full_f, full_r = flappy.render_text(falling), flappy.render_text(rising)
    assert "falling" in full_f and "rising" in full_r
    assert full_f != full_r  # the confound: same offset, different text

    terse_f, terse_r = flappy.render_terse(falling), flappy.render_terse(rising)
    assert terse_f == terse_r  # velocity-invariant
    for word in ("falling", "rising", "velocity"):
        assert word not in terse_f
    assert "+0.51 relative to the gap centre" in terse_f
    assert "above the gap" in terse_f
    assert len(terse_f) < len(full_f)


def test_states_mapping_and_policy_uses_it(flappy) -> None:
    assert set(flappy.STATES) == {"full", "terse"}

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {flappy.QKEY: {"type": "noul", "noul": 0.9}}})

    s = {"y": 0.4, "vy": -0.003, "dx": 0.5, "lo": 0.36, "hi": 0.64, "score": 0, "t": 1}
    for name in ("full", "terse"):
        client = httpx.Client(base_url="http://t", transport=httpx.MockTransport(handler))
        flappy.LayaPolicy("http://t", "noul", name, client=client)(s)
        assert sent[-1]["state"] == flappy.STATES[name](s)
    assert sent[0]["state"] != sent[1]["state"]
