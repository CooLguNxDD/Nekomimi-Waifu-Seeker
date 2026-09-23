"""Per-turn timing: spans add up, payloads carry them, one log line per turn."""

from __future__ import annotations

import logging

import pytest

from waifu_engine import timing
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod

POOL = [
    {"id": "mario", "name": "Mario", "series": "Super Mario", "medium": "game",
     "tags": ["game", "male"]},
    {"id": "miku", "name": "Hatsune Miku", "series": "Vocaloid", "medium": "game",
     "tags": ["game", "female", "blue"]},
    {"id": "makima", "name": "Makima", "series": "Chainsaw Man", "medium": "anime",
     "tags": ["anime", "female"]},
]


class FakeAgent:
    backend = "torch"

    def predict(self, state, questions):
        key = next(iter(questions))
        q = questions[key]
        if q.get("type") == "choice":
            keys = list(q["criteria"])
            return {"answers": {key: {"probabilities": {k: 1 / len(keys) for k in keys}}}}
        return {"answers": {key: {"noul": 0.6, "action": {"act_probability": 0.1}}}}


@pytest.fixture
def lines():
    got = []

    class Grab(logging.Handler):
        def emit(self, record):
            got.append(record.getMessage())

    handler = Grab()
    timing.log.addHandler(handler)
    yield got
    timing.log.removeHandler(handler)


def test_span_is_a_no_op_outside_a_trace():
    with timing.span("anything"):
        pass
    assert timing.current() is None


def test_spans_add_up_and_count_calls(lines):
    @timing.traced("demo")
    def handler():
        for _ in range(3):
            with timing.span("laya.match"):
                pass
        timing.note(turn=2)
        return {"ok": True}

    out = handler()
    spans = out["timing"]["spans"]
    assert spans["laya.match"]["calls"] == 3
    assert out["timing"]["step"] == "demo" and out["timing"]["turn"] == 2
    assert len(lines) == 1 and lines[0].startswith("demo turn=2 total=")
    assert "laya.match" in lines[0] and "/3x" in lines[0]
    assert timing.current() is None


def test_log_line_can_be_silenced(monkeypatch, lines):
    monkeypatch.setenv("WAIFU_TIMING_LOG", "0")
    out = timing.traced("quiet")(lambda: {})()
    assert "timing" in out and lines == []


def test_a_real_turn_reports_its_bottleneck(monkeypatch, lines):
    monkeypatch.setattr(laya_client, "get_agent", lambda: FakeAgent())
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: list(POOL))
    monkeypatch.setattr(engine, "ONLINE", True)
    state = engine.start("")
    assert state["timing"]["step"] == "start"
    s = sess_mod.get_session(state["session_id"])
    state = engine.submit_answer(s, "yes")  # medium: settled by tags, no model calls
    q = state["question"]
    state = engine.submit_answer(s, "other" if q.get("kind") == "choice" else "yes")
    t = state["timing"]
    spans = t["spans"]
    for name in ("score", "search", "search.fetch", "pick", "laya.ready_to_guess"):
        assert name in spans, name
    # Both game candidates got their own match call; the anime one was
    # removed by the medium answer before inference.
    assert spans["laya.match"]["calls"] == 2
    assert t["total_ms"] >= spans["score"]["ms"]
    assert list(spans) == sorted(spans, key=lambda n: -spans[n]["ms"])  # slowest first
    assert lines[-1].startswith("answer ")
