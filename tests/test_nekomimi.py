"""Offline tests for the Nekomimi loop.

Laya and DuckDuckGo are both stubbed, so these run with no weights and no
network.
"""

from __future__ import annotations

import time

import pytest

from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.nekomimi.session import Candidate, GuessSession, logit

POOL = [
    {
        "id": "c_makima", "name": "Makima", "series": "Chainsaw Man", "medium": "anime",
        "blurb": "A mysterious woman and public safety devil hunter.",
        "tags": ["anime", "female", "antagonist", "mysterious", "adult"],
    },
    {
        "id": "c_mario", "name": "Mario", "series": "Super Mario", "medium": "game",
        "blurb": "An Italian plumber and the playable hero of Nintendo games.",
        "tags": ["game", "male", "protagonist", "playable", "famous", "mascot"],
    },
    {
        "id": "c_spiderman", "name": "Spider-Man", "series": "Marvel Comics", "medium": "comic",
        "blurb": "A teenage superhero from Marvel comics who shoots webs.",
        "tags": ["comic", "male", "protagonist", "teen", "superhuman"],
    },
    {
        "id": "c_miku", "name": "Hatsune Miku", "series": "Vocaloid", "medium": "game",
        "blurb": "A virtual singer with long blue twintails.",
        "tags": ["game", "female", "idol", "blue", "twintails"],
    },
]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No model, no network."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    monkeypatch.setattr(laya_client, "available", lambda: False)
    monkeypatch.setattr(engine.web_search, "search_by_constraints", lambda *a, **k: list(POOL))
    monkeypatch.setattr(
        engine.web_search, "search_characters_multiround", lambda *a, **k: (list(POOL), [])
    )
    # The engine now sources through AniList/Wikipedia; without this the suite
    # would quietly go to the network and pull in real characters.
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: list(POOL))
    monkeypatch.setattr(engine, "load_catalog", lambda: [])
    yield


def _fresh_session() -> GuessSession:
    s = sess_mod.new_session("test")
    s.add_candidates(POOL)
    return s


# --- trait bank ------------------------------------------------------


def test_bank_ids_unique_and_well_formed():
    ids = [q["id"] for q in traits.QUESTION_BANK]
    assert len(ids) == len(set(ids))
    for q in traits.QUESTION_BANK:
        assert q["text"].endswith("?")
        assert "`candidate`" in q["instructions"]
        assert 0.0 < q["prior"] < 1.0
        assert q["category"]


def test_answer_weights():
    assert traits.ANSWER_WEIGHT["yes"] == 1.0
    assert traits.ANSWER_WEIGHT["no"] == -1.0
    assert traits.ANSWER_WEIGHT["detail"] == 0.0


# --- session maths ---------------------------------------------------


def test_posterior_sums_to_one():
    s = _fresh_session()
    s.candidates[0].logodds = 2.0
    total = sum(p for _, p in s.posterior())
    assert total == pytest.approx(1.0)
    assert s.posterior()[0][0].id == "c_makima"


def test_prune_eliminates_far_behind_candidates():
    s = _fresh_session()
    s.candidates[0].logodds = 10.0
    dropped = s.prune()
    assert dropped == 3
    assert [c.id for c in s.alive_candidates()] == ["c_makima"]


def test_prune_keeps_at_least_two():
    s = GuessSession(id="x", created=0, updated=0)
    s.add_candidates(POOL[:2])
    s.candidates[0].logodds = 99.0
    assert s.prune() == 0


def test_new_candidates_start_at_pool_median():
    s = _fresh_session()
    for c, v in zip(s.candidates, [4.0, 2.0, 0.0, -2.0]):
        c.logodds = v
    s.add_candidates([{"id": "c_new", "name": "New", "tags": []}])
    assert s.by_id("c_new").logodds == 2.0


def test_ttl_purge(monkeypatch):
    s = sess_mod.new_session()
    s.updated = time.time() - sess_mod.SESSION_TTL_SECONDS - 1
    assert sess_mod.get_session(s.id) is None


def test_logit_is_clamped():
    assert logit(0.0) == pytest.approx(logit(0.02))
    assert logit(1.0) == pytest.approx(logit(0.98))


# --- question selection ----------------------------------------------


def test_settled_category_is_not_asked_again():
    s = _fresh_session()
    s.asked.append({"qid": "medium_game", "text": "?", "category": "medium", "answer": "yes"})
    picked = engine.candidate_questions(s)
    assert all(q["category"] != "medium" for q in picked)


def test_question_ranking_prefers_even_splits():
    s = _fresh_session()
    picked = engine.candidate_questions(s)
    qualities = [engine._split_quality(q, s.alive_candidates()) for q in picked]
    assert qualities == sorted(qualities, reverse=True)


# --- evidence --------------------------------------------------------


def test_yes_answer_promotes_matching_candidates():
    s = _fresh_session()
    q = traits.QUESTIONS_BY_ID["medium_game"]
    engine.score_candidates(s, q, "yes")
    assert s.by_id("c_mario").logodds > s.by_id("c_makima").logodds


def test_no_answer_demotes_matching_candidates():
    s = _fresh_session()
    q = traits.QUESTIONS_BY_ID["medium_game"]
    engine.score_candidates(s, q, "no")
    assert s.by_id("c_mario").logodds < s.by_id("c_makima").logodds


def test_detail_answer_carries_no_weight():
    s = _fresh_session()
    before = [c.logodds for c in s.candidates]
    engine.score_candidates(s, traits.QUESTIONS_BY_ID["medium_game"], "detail")
    assert [c.logodds for c in s.candidates] == before


# --- full loop -------------------------------------------------------


def _answer_as(target_tags: set[str], question_id: str) -> str:
    q = traits.QUESTIONS_BY_ID.get(question_id)
    if q is None:
        return "no"
    return "yes" if set(q["tags_true"]) & target_tags else "no"


def test_loop_converges_on_the_target():
    target = set(POOL[1]["tags"])  # Mario
    state = engine.start("nintendo")
    assert state["stage"] == "asking"
    s = sess_mod.get_session(state["session_id"])
    for _ in range(engine.MAX_TURNS + 2):
        if state["stage"] != "asking":
            break
        state = engine.submit_answer(s, _answer_as(target, state["question"]["qid"]))
    assert state["stage"] == "guessing"
    assert state["guess"]["name"] == "Mario"


def test_rejected_guess_never_returns():
    state = engine.start("")
    s = sess_mod.get_session(state["session_id"])
    while state["stage"] == "asking":
        state = engine.submit_answer(s, "yes")
    rejected = state["guess"]["id"]
    state = engine.submit_guess_result(s, correct=False)
    assert rejected in s.rejected
    assert all(c.id != rejected for c in s.alive_candidates())
    while state.get("stage") == "asking":
        state = engine.submit_answer(s, "no")
    if state.get("stage") == "guessing":
        assert state["guess"]["id"] != rejected


def test_correct_guess_finishes_the_round():
    state = engine.start("")
    s = sess_mod.get_session(state["session_id"])
    while state["stage"] == "asking":
        state = engine.submit_answer(s, "yes")
    done = engine.submit_guess_result(s, correct=True)
    assert done["stage"] == "done"
    assert done["correct"] is True
    assert done["winner"]["name"]


def test_loop_stops_at_max_turns():
    state = engine.start("")
    s = sess_mod.get_session(state["session_id"])
    turns = 0
    while state["stage"] == "asking" and turns < engine.MAX_TURNS + 5:
        state = engine.submit_answer(s, "detail", detail="still vague")
        turns += 1
    assert s.turn <= engine.MAX_TURNS
    assert state["stage"] in {"guessing", "done"}


def test_bad_answer_is_rejected():
    state = engine.start("")
    s = sess_mod.get_session(state["session_id"])
    out = engine.submit_answer(s, "maybe")
    assert "error" in out


def test_answer_without_pending_question_errors():
    s = _fresh_session()
    assert "error" in engine.submit_answer(s, "yes")
    assert "error" in engine.submit_guess_result(s, True)


# --- Laya wiring -----------------------------------------------------


def test_question_choice_is_ours_not_layas(monkeypatch):
    """Information gain picks the question; Laya only reports readiness.

    Laya used to choose from a list of question ids and came back near-uniform
    (confidence 0.0003), so the choice call was dropped.
    """
    s = _fresh_session()
    options = engine.candidate_questions(s)
    asked = []

    def fake_ask(state, questions):
        asked.append(set(questions))
        return {"ready_to_guess": {"noul": 0.1, "action": {"act_probability": 0.1}}}

    monkeypatch.setattr(laya_client, "ask", fake_ask)
    chosen, answers = engine._pick_question(s)

    assert chosen["id"] == options[0]["id"]
    assert asked == [{"ready_to_guess"}]
    assert answers is not None
    assert s.laya_used


def test_question_ranking_is_information_gain():
    s = _fresh_session()
    # Every candidate is tagged with its medium, so "is it a video game
    # character" splits the pool and beats a trait nothing is tagged with.
    splitter = engine._split_quality(traits.QUESTIONS_BY_ID["medium_game"], s.posterior())
    dud = engine._split_quality(traits.QUESTIONS_BY_ID["eyes_heterochromia"], s.posterior())
    assert splitter > dud
    assert 0.0 <= dud <= splitter <= 1.0


def test_medium_answer_eliminates_contradicting_candidates():
    s = _fresh_session()
    engine.score_candidates(s, traits.QUESTIONS_BY_ID["medium_game"], "yes")
    alive = {c.name for c in s.alive_candidates()}
    assert "Mario" in alive
    assert "Makima" not in alive  # anime
    assert "Spider-Man" not in alive  # comic


def test_laya_noul_drives_scoring(monkeypatch):
    s = _fresh_session()

    def fake_ask(state, questions):
        # score_candidates scores one candidate per call, state["candidate"]
        profile = state.get("candidate", "")
        return {"match": {"noul": 0.95 if "Hatsune Miku" in profile else 0.05}}

    monkeypatch.setattr(laya_client, "ask", fake_ask)
    engine.score_candidates(s, traits.QUESTIONS_BY_ID["gender_female"], "yes")
    assert s.posterior()[0][0].id == "c_miku"


def test_laya_client_returns_none_without_model(monkeypatch):
    monkeypatch.setenv("WAIFU_FORCE_FALLBACK", "1")
    laya_client.reset()
    assert laya_client.get_agent() is None
    assert laya_client.available() is False


def test_nekomimi_page_and_api_paths():
    from fastapi.testclient import TestClient

    from waifu_engine.web import app

    client = TestClient(app, follow_redirects=False)
    page = client.get("/nekomimi")
    assert page.status_code == 200
    assert b"Nekomimi" in page.content
    assert b"/api/nekomimi/" in page.content
    assert client.get("/akinator").status_code == 404
    assert client.post("/api/akinator/start", json={}).status_code == 404
