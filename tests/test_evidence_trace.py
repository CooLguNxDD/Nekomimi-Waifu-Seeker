"""Miss evidence: per-answer attribution, round trace, LLM-names funnel, EIG notes."""

from __future__ import annotations

import logging
import math

import pytest

from waifu_engine import sources, timing, web_search
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod
from waifu_engine.nekomimi.harness import SessionMemory
from waifu_engine.nekomimi.traits import QUESTIONS_BY_ID
from waifu_engine.sources import anilist, llm_names, wikipedia

# The autouse fixture stubs ``sources.find_candidates`` for the engine; the
# duplicate count is read from the real one.
_FIND_CANDIDATES = sources.find_candidates


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Heuristic scoring only, and no search."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    monkeypatch.setattr(engine, "ONLINE", False)


def _round() -> sess_mod.GuessSession:
    """Three candidates: Lyra (elf), Mara and Nell (humans), Mara the most famous."""
    state = engine.start("")
    sess = sess_mod.get_session(state["session_id"])
    sess.candidates = []
    sess.add_candidates([
        {"id": "lyra", "name": "Lyra", "series": "Grove", "medium": "game",
         "tags": ["elf"], "blurb": "An elf archer.", "popularity": 20},
        {"id": "mara", "name": "Mara", "series": "Grove", "medium": "game",
         "tags": ["human"], "blurb": "A human knight.", "popularity": 5000},
        {"id": "nell", "name": "Nell", "series": "Fen", "medium": "game",
         "tags": ["human"], "blurb": "A human thief.", "popularity": 10},
    ])
    return sess


def _answer(sess: sess_mod.GuessSession, qid: str, answer: str) -> dict:
    """Make ``qid`` the pending question and answer it through the engine."""
    engine._emit_asking(sess, QUESTIONS_BY_ID[qid])
    sess.stage = "asking"
    return engine.submit_answer(sess, answer)


def test_attribution_rows_sum_to_each_candidates_logodds():
    sess = _round()
    _answer(sess, "species_elf", "yes")
    _answer(sess, "gender_female", "yes")
    for cand in sess.alive_candidates():
        rows = engine.evidence_breakdown(sess, cand.id)
        assert {"__prior__", "__adjust__", "species_elf"} <= {r["qid"] for r in rows}
        assert math.isclose(sum(r["logodds"] for r in rows), cand.logodds, abs_tol=1e-3)
        assert rows == sorted(rows, key=lambda r: r["logodds"])
    worst = engine.evidence_breakdown(sess, "mara")[0]
    assert worst == {"qid": "species_elf", "answer": "yes", "logodds": worst["logodds"]}
    assert worst["logodds"] < 0


def test_trace_follows_the_target_and_explains_a_wrong_guess():
    sess = _round()
    _answer(sess, "species_human", "no")
    trace = engine.trace_payload(sess, "Nell")
    assert [t["qid"] for t in trace["turns"]] == ["species_human"]
    assert trace["turns"][0]["target"]["name"] == "Nell"
    assert trace["target"]["in_pool"] and trace["target"]["alive"]

    engine._guess_payload(sess)
    guessed = sess.guess_log[-1]["id"]
    engine.submit_guess_result(sess, False)
    trace = engine.trace_payload(sess, "Nell")
    guess = trace["guesses"][0]
    # Heuristic likelihoods are capped to [0.4, 0.6], so Mara's fame wins
    # over "human: no". The gap has to name the prior, not the answer.
    assert guessed == "mara"
    assert guess["evidence"], "a rejected guess keeps its snapshot"
    gap = guess["gap"]
    assert gap[0]["qid"] == "__prior__" and gap[0]["delta"] < 0
    assert gap == sorted(gap, key=lambda r: r["delta"])
    assert trace["wrong_guesses"] == 1
    assert trace["turn"] <= trace["turn_cap"]


def test_trace_without_target_and_with_an_absent_target():
    sess = _round()
    _answer(sess, "species_elf", "yes")
    assert "target" not in engine.trace_payload(sess)
    absent = engine.trace_payload(sess, "Makima")
    assert absent["target"]["in_pool"] is False
    assert absent["turns"][0]["target"] is None


def test_harness_miss_carries_the_trace_and_flags_turns_past_the_cap():
    sess = _round()
    _answer(sess, "species_elf", "no")
    memory = SessionMemory(target="Lyra")
    memory.note(sess, answer="no")
    assert memory.cap_violations == []
    sess.turn = sess.turn_cap() + 1
    memory.note(sess, answer="no")
    memory.record_miss(sess)
    out = memory.export()
    assert out["cap_violations"] == [sess.turn_cap() + 1]
    assert out["miss"]["trace"]["target"]["name"] == "Lyra"


def _hit(name: str, series: str) -> dict:
    return {"id": f"x_{name}", "name": name, "series": series, "medium": "anime",
            "blurb": f"{name} from {series}.", "tags": [], "source": "anilist"}


def test_llm_names_funnel_counts_where_the_names_went(monkeypatch):
    monkeypatch.setattr(llm_names.query_llm, "propose_characters", lambda facts, medium, n: [
        {"name": "Makima", "series": "Chainsaw Man"},
        {"name": "Invented Girl", "series": "Nowhere"},
    ])
    pages = {"Makima": [_hit("Makima", "Chainsaw Man")]}
    monkeypatch.setattr(anilist, "search_characters", lambda q, limit=10: pages.get(q, []))
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit=8: [])
    stats: dict = {}
    out = llm_names.search(["female", "demon"], "anime", stats=stats)
    assert [c["name"] for c in out] == ["Makima"]
    assert stats == {"proposed": ["Makima", "Invented Girl"], "resolved": ["Makima"],
                     "unresolved": ["Invented Girl"]}
    note = sources._funnel_note(stats)
    assert note.startswith("proposed=2 resolved=1 unresolved=1 names=[Makima; Invented Girl]")


def test_background_log_line_carries_the_funnel(monkeypatch, caplog):
    def fake_search(facts, medium_hint=None, errors=None, stats=None):
        stats.update(proposed=["Makima"], resolved=[], unresolved=["Makima"])
        return []

    monkeypatch.setattr(llm_names, "search", fake_search)
    monkeypatch.setattr(timing.log, "propagate", True)
    with caplog.at_level(logging.INFO, logger="waifu.timing"):
        sources._LLM_BG.pending.add("funnel-key")
        sources._llm_names_run("funnel-key", ["female"], None)
    assert "found=0 proposed=1 resolved=0 unresolved=1 names=[Makima]" in caplog.text


def test_search_counts_background_hits_already_in_play(monkeypatch):
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    monkeypatch.setattr(web_search, "_want_ddg_fill", lambda *a, **k: False)
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit=8: [])
    monkeypatch.setattr(anilist, "search_characters", lambda q, limit=10: [])
    monkeypatch.setattr(sources, "popular_characters", lambda medium, n: [])
    monkeypatch.setattr(sources.gemini, "enabled", lambda: False)
    with sources._LLM_BG.lock:
        sources._LLM_BG.ready["dup-key"] = [_hit("Makima", "Chainsaw Man"),
                                            _hit("Power", "Chainsaw Man")]
    _FIND_CANDIDATES(["female"], background_key="dup-key",
                     exclude_names={"Makima"}, specific=False)
    assert web_search.last_search_meta()["dups"] == {"llm_bg": 1}


def test_lookahead_notes_passes_and_packed_questions(monkeypatch):
    sess = _round()
    engine._rescore_candidates(sess)
    questions = [QUESTIONS_BY_ID["species_elf"], QUESTIONS_BY_ID["gender_female"]]

    def ask(state, qs):
        return {key: {"noul": 0.7} for key in qs}

    monkeypatch.setattr(laya_client, "ask", ask)
    notes: dict = {}
    monkeypatch.setattr(timing, "note", lambda **kw: notes.update(kw))
    assert engine._lookahead_eig(sess, questions) is not None
    assert notes == {"eig_calls": 3, "eig_qs": 6}


def test_trace_endpoint_reads_a_live_session():
    from fastapi.testclient import TestClient

    from waifu_engine import web

    sess = _round()
    _answer(sess, "species_elf", "yes")
    client = TestClient(web.app)
    got = client.get(f"/api/nekomimi/trace/{sess.id}", params={"target": "Lyra"}).json()
    assert got["turns"][0]["qid"] == "species_elf"
    assert got["target"]["alive"] is True
    assert "error" in client.get("/api/nekomimi/trace/nope").json()
