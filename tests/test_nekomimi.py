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
    monkeypatch.setattr(engine, "ONLINE", True)
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


def test_catalog_popularity_prior_does_not_compound_with_insertion_order():
    s = sess_mod.new_session()
    s.add_candidates([{"id": str(i), "name": f"Character {i}", "popularity": 10000}
                      for i in range(900)])
    assert {c.logodds for c in s.candidates} == {sess_mod.popularity_prior(10000)}


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


def test_model_scores_target_beyond_top_ten_in_900_candidates(monkeypatch):
    s = sess_mod.new_session()
    s.add_candidates([
        {"id": f"d{i}", "name": f"Decoy {i}", "medium": "game",
         "tags": ["female"], "popularity": 100000}
        for i in range(899)
    ] + [POOL[1]])
    mario = s.by_id("c_mario")
    mario.logodds = -100
    assert mario not in s.scoring_pool()
    calls = []

    def fake_ask(state, questions):
        calls.append(state["candidate"])
        return {"match": {"noul": 0.05 if "Mario (" in state["candidate"] else 0.95}}

    monkeypatch.setattr(laya_client, "ask", fake_ask)
    q = traits.QUESTIONS_BY_ID["gender_female"]
    engine.score_candidates(s, q, "no")
    assert len(calls) == 900
    assert s.posterior()[0][0] is mario
    assert mario.alive
    before = [c.logodds for c in s.candidates]
    engine.score_candidates(s, q, "no")
    assert len(calls) == 900  # replay uses cache, not another model pass
    assert [c.logodds for c in s.candidates] == before


def test_refresh_replays_history_and_filters_medium_before_model(monkeypatch):
    s = _fresh_session()
    calls = []

    def fake_ask(state, questions):
        calls.append(state["candidate"])
        return {"match": {"noul": 0.8}}

    monkeypatch.setattr(laya_client, "ask", fake_ask)
    engine.score_candidates(s, traits.QUESTIONS_BY_ID["medium_game"], "yes")
    engine.score_candidates(s, traits.QUESTIONS_BY_ID["gender_female"], "yes")
    assert len(calls) == 2
    monkeypatch.setattr(engine, "ONLINE", True)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [
        {"id": "new_game", "name": "New Game", "medium": "game"},
        {"id": "new_anime", "name": "New Anime", "medium": "anime"},
    ])
    engine.refresh_candidates(s)
    assert not s.by_id("new_anime").alive
    assert len(calls) == 3
    assert s.by_id("new_game").logodds == pytest.approx(s.by_id("c_miku").logodds)


def test_weak_model_evidence_is_recoverable_and_does_not_rewrite_tags(monkeypatch):
    s = _fresh_session()
    tags = {c.id: list(c.tags) for c in s.candidates}
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: {
        "match": {"noul": 0.02 if "Mario (" in state["candidate"] else 0.98}
    })
    engine.score_candidates(s, traits.QUESTIONS_BY_ID["gender_female"], "yes")
    assert s.by_id("c_mario").alive
    assert {c.id: c.tags for c in s.candidates} == tags


def test_unknown_traits_do_not_match_substrings():
    q = {"tags_true": ["male"], "tags_false": []}
    assert engine._tag_match(q, Candidate(id="x", name="A female character")) == 0.5


def test_model_profile_does_not_promote_mined_tags_to_facts():
    mario = Candidate(id="m", name="Mario", tags=["royalty", "antagonist"],
                      blurb="A plumber who rescues Princess Peach from Bowser.")
    assert "royalty" not in mario.profile()
    assert "antagonist" not in mario.profile()
    assert "plumber" in mario.profile()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.2, 1.2, None])
def test_invalid_model_probabilities_fall_back_without_poisoning_posterior(monkeypatch, invalid):
    s = _fresh_session()
    monkeypatch.setattr(laya_client, "ask", lambda *a: {"match": {"noul": invalid}})
    engine.score_candidates(s, traits.QUESTIONS_BY_ID["gender_female"], "yes")
    assert not s.match_cache
    assert sum(p for _, p in s.posterior()) == pytest.approx(1.0)


def test_unknown_question_has_zero_information_gain():
    s = _fresh_session()
    q = dict(traits.QUESTIONS_BY_ID["gender_female"], tags_true=["absent"], tags_false=[])
    assert engine._split_quality(q, s.posterior()) == pytest.approx(0.0)


def test_full_round_with_900_candidates_and_model_judgments(monkeypatch):
    catalog = [
        {"id": f"d{i}", "name": f"Decoy {i}", "medium": "game",
         "tags": ["game", "female"], "popularity": 100000}
        for i in range(899)
    ] + [POOL[1]]
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: catalog)
    target_tags = set(POOL[1]["tags"])
    by_instruction = {q["instructions"]: q for q in traits.QUESTION_BANK}
    evaluated = set()

    def fake_ask(state, questions):
        if "match" not in questions:
            return None
        q = by_instruction[questions["match"]["instructions"]]
        is_mario = state["candidate"].startswith("Mario (")
        if is_mario:
            evaluated.add(q["id"])
        candidate_tags = target_tags if is_mario else {"game", "female"}
        p = 0.95 if set(q["tags_true"]) & candidate_tags else 0.05
        return {"match": {"noul": p}}

    monkeypatch.setattr(laya_client, "ask", fake_ask)
    state = engine.start()
    s = sess_mod.get_session(state["session_id"])
    while state["stage"] == "asking":
        state = engine.submit_answer(s, _answer_as(target_tags, state["question"]["qid"]))
    assert evaluated
    assert state["stage"] == "guessing"
    assert state["guess"]["name"] == "Mario"
    assert state["guess_number"] == 1
    assert s.turn <= engine.MAX_TURNS


def test_empty_search_continues_and_target_can_arrive_after_answer(monkeypatch):
    calls = []

    def search(terms, **kwargs):
        calls.append((terms, kwargs.get("medium_hint")))
        return [] if len(calls) == 1 else [POOL[1]]

    monkeypatch.setattr(engine.sources, "find_candidates", search)
    state = engine.start()
    assert state["stage"] == "asking"
    assert state["question"]["qid"] == "medium_game"
    s = sess_mod.get_session(state["session_id"])
    assert not s.candidates
    state = engine.submit_answer(s, "yes")
    assert len(calls) == 2
    assert calls[-1][1] == "game"
    assert s.by_id("c_mario") is not None
    assert state["stage"] == "asking"


def test_search_terms_preserve_seed_and_details_but_not_negative_keywords():
    s = sess_mod.new_session("Nintendo")
    s.asked = [
        {"text": "Is your character female?", "answer": "no", "detail": None},
        {"text": "Is your character male?", "answer": "yes", "detail": "Italian plumber"},
    ]
    assert engine._search_terms(s) == ["Nintendo", "Italian plumber", "male"]


def test_a_single_unverified_search_result_is_not_enough_to_guess():
    s = sess_mod.new_session()
    s.add_candidates([POOL[1]])
    s.turn = 6
    assert s.posterior()[0][1] == 1.0
    assert not engine._should_guess(s, {"ready_to_guess": {
        "noul": 0.99, "action": {"act_probability": 0.99}}})


def test_free_text_clues_are_evaluated_by_laya(monkeypatch):
    seen = []
    monkeypatch.setattr(laya_client, "ask", lambda state, questions:
                        seen.append(state) or {"match": {"noul": 0.8}})
    state = engine.start("Italian plumber")
    s = sess_mod.get_session(state["session_id"])
    assert "clue_seed" in s.evidence
    assert any(x.get("clues") == "Italian plumber" for x in seen)


def test_runtime_never_reads_local_catalog(monkeypatch):
    from waifu_engine import catalog, decide

    def forbidden():
        pytest.fail("runtime must not read catalog.json")

    monkeypatch.setattr(catalog, "load_catalog", forbidden)
    assert engine.start()["stage"] == "asking"
    assert decide.determine("Mario", online=False)["mode"] == "empty"


def test_search_driven_round_discovers_target_from_later_detail(monkeypatch):
    target_tags = set(POOL[1]["tags"])
    queries = []

    def search(terms, **kwargs):
        queries.append(terms)
        return [POOL[1]] if "Italian plumber" in terms else [POOL[0]]

    def judge(state, questions):
        if "match" not in questions:
            return None
        if state.get("clues"):
            return {"match": {"noul": 0.95 if state["candidate"].startswith("Mario (") else 0.05}}
        q = next(q for q in traits.QUESTION_BANK
                 if q["instructions"] == questions["match"]["instructions"])
        return {"match": {"noul": 0.95 if set(q["tags_true"]) & target_tags else 0.05}}

    monkeypatch.setattr(engine.sources, "find_candidates", search)
    monkeypatch.setattr(laya_client, "ask", judge)
    state = engine.start()
    s = sess_mod.get_session(state["session_id"])
    assert s.by_id("c_mario") is None
    state = engine.submit_answer(s, "yes")  # game; the first anime hit is removed
    assert state["stage"] == "asking"
    state = engine.submit_answer(s, "detail", "Italian plumber")
    assert s.by_id("c_mario") is not None
    while state["stage"] == "asking":
        state = engine.submit_answer(s, _answer_as(target_tags, state["question"]["qid"]))
    assert state["guess"]["name"] == "Mario"
    assert state["guess_number"] == 1
    assert len(queries) == 1 + sum(bool(a["answer"]) for a in s.asked)


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
