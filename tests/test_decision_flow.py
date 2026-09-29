"""Guess/question decision flow after the 2/5 live validation run.

All three misses (Makima, Megumin, Tifa) reached the turn cap and then spent
every guess back to back on the same evidence. These tests lock the fixes:
recovery turns after a wrong guess, in-cluster disambiguation, a forced
series question, and Laya-scored lookahead gain for question choice.
"""

from __future__ import annotations

import pytest

from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.nekomimi.session import Candidate


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Heuristic scoring only, and no search."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    monkeypatch.setattr(engine, "ONLINE", False)


def _arm_guess(sess, leader_id: str, turn: int = 8) -> None:
    """Two strong model judgments so an early guess is otherwise allowed."""
    sess.turn = turn
    for qid, p in (("gender_female", 0.9), ("age_teen", 0.8)):
        sess.evidence[qid] = (traits.QUESTIONS_BY_ID[qid], "yes")
        sess.match_cache[(leader_id, qid)] = p


def _crimson_demons(sess) -> None:
    """Arue leads at ~0.81, Yunyun ~0.12. The two share every rare look.

    Only an ordinary bank trait (Arue's small build) separates the leader
    from her clan, so near-twin's rare-look loop has nothing to ask.
    """
    sess.candidates = [
        Candidate(id="arue", name="Arue", series="KonoSuba", logodds=4.05,
                  popularity=300, tags=["female", "mage", "small"]),
        Candidate(id="yunyun", name="Yunyun", series="KonoSuba", logodds=2.1,
                  popularity=5000, tags=["female", "mage"]),
        Candidate(id="megumin", name="Megumin", series="KonoSuba", logodds=1.7,
                  popularity=60000, tags=["female", "mage", "eyepatch", "staff"]),
    ]


def test_wrong_guess_at_the_cap_asks_before_the_next_guess():
    """Guess 2 follows new evidence, not the runner-up of the same posterior."""
    sess = sess_mod.new_session()
    _crimson_demons(sess)
    _arm_guess(sess, "arue", turn=engine.MAX_TURNS)
    first = engine._advance(sess)
    assert first["stage"] == "guessing"
    assert first["guess"]["id"] == "arue"

    state = engine.submit_guess_result(sess, correct=False)
    assert sess.turn_cap() == engine.MAX_TURNS + sess_mod.RECOVERY_TURNS
    assert state["stage"] == "asking"
    assert state["question"]["max_turns"] == sess.turn_cap()

    for _ in range(sess_mod.RECOVERY_TURNS + 1):
        if state["stage"] != "asking":
            break
        state = engine.submit_answer(sess, "yes" if state["question"]["kind"] == "yesno"
                                     else state["question"]["options"][0]["key"])
    assert state["stage"] == "guessing"
    assert sess.turn <= sess.turn_cap()


def test_turn_cap_is_unchanged_without_a_wrong_guess():
    sess = sess_mod.new_session()
    assert sess.turn_cap() == engine.MAX_TURNS


def test_cluster_split_defers_a_clan_guess_once():
    """No rare look splits the clan, so any bank trait that does is asked once."""
    sess = sess_mod.new_session()
    _crimson_demons(sess)
    _arm_guess(sess, "arue")
    assert engine._should_guess(sess, None) is True
    state = engine._advance(sess)
    assert state["stage"] == "asking"
    qid = state["question"]["qid"]
    question = traits.QUESTIONS_BY_ID[qid]
    cast = [sess.by_id(i) for i in ("arue", "yunyun", "megumin")]
    has = {engine._candidate_has_trait(question, c) for c in cast}
    assert has == {True, False}
    # The same leader/runner pair is not deferred twice.
    again = engine._advance(sess)
    assert again["stage"] == "guessing"


def test_cluster_split_ignores_other_works():
    sess = sess_mod.new_session()
    sess.candidates = [
        Candidate(id="arue", name="Arue", series="KonoSuba", logodds=3.0, tags=["female"]),
        Candidate(id="tifa", name="Tifa Lockhart", series="Final Fantasy VII",
                  logodds=1.4, tags=["female", "fists"]),
    ]
    assert engine._cluster_split_question(sess, sess.by_id("arue")) is None


def test_series_is_pinned_once_one_franchise_holds_the_mass():
    """Tifa's round: after "Another series", FF7 gathered mass and was never offered."""
    sess = sess_mod.new_session()
    sess.candidates = [
        Candidate(id="tifa", name="Tifa Lockhart", series="Final Fantasy VII", logodds=1.0),
        Candidate(id="aerith", name="Aerith Gainsborough", series="Final Fantasy VII",
                  logodds=0.8),
        Candidate(id="bayo", name="Bayonetta", series="Bayonetta", logodds=0.9),
    ]
    sess.asked.append({
        "qid": "series", "text": "Which series is your character from?",
        "category": "series", "kind": "choice", "answer": "other", "detail": None,
        "options": {"s1": {"label": "Genshin Impact", "series_key": "genshinimpact"},
                    "other": {"label": "Another series"}},
    })
    sess.turn = engine.SERIES_PIN_TURN
    assert engine.candidate_questions(sess)[0]["id"] == "series_2"

    sess.turn = engine.SERIES_PIN_TURN - 1
    assert engine._series_pin_id(sess) == ""


def test_series_is_not_pinned_for_a_diffuse_pool():
    sess = sess_mod.new_session()
    sess.turn = engine.SERIES_PIN_TURN
    sess.candidates = [
        Candidate(id=f"c{i}", name=f"Name {i}", series=f"Work {i}", logodds=0.0)
        for i in range(5)
    ]
    assert engine._series_pin_id(sess) == ""


def _eig_session():
    """Two unpinned leads. Tags make kindness look like the split; Laya disagrees."""
    sess = sess_mod.new_session()
    sess.turn = 2
    sess.candidates = [
        Candidate(id="aya", name="Aya Kind", series="Work A", logodds=0.0,
                  tags=["kind"], blurb="She wears glasses."),
        Candidate(id="bea", name="Bea Plain", series="Work B", logodds=0.0),
    ]
    return sess


def test_laya_lookahead_picks_the_question_laya_can_split(monkeypatch):
    # Raw nouls: this is about the lookahead, not silent-profile calibration
    # (Bea's empty profile would otherwise pull her 0.05 toward the prior).
    monkeypatch.setattr(engine, "SILENT_WEIGHT", 1.0)
    sess = _eig_session()
    heuristic = engine.candidate_questions(sess)
    ids = [q["id"] for q in heuristic]
    assert "look_glasses" in ids and ids.index("pers_kind") < ids.index("look_glasses")
    calls: list[set] = []

    def ask(state, questions):
        calls.append(set(questions))
        profile = str(state.get("candidate") or "")
        if all(k.startswith("eig") for k in questions):
            out = {}
            for key, spec in questions.items():
                glasses = "glasses" in spec["instructions"].lower()
                p = (0.95 if "Aya" in profile else 0.05) if glasses else 0.5
                out[key] = {"noul": p, "action": {"act_probability": 0.7}}
            return out
        if "ready_to_guess" in questions:
            return {"ready_to_guess": {"noul": 0.1, "action": {"act_probability": 0.1}}}
        return None

    monkeypatch.setattr(laya_client, "ask", ask)
    chosen, _answers = engine._pick_question(sess)
    assert chosen["id"] == "look_glasses"
    assert sess.last_eig is not None and sess.last_eig > 0.5

    before = len(calls)
    engine.score_candidates(sess, chosen, "yes")
    # The lookahead judgments were made under the same rows: no new match call.
    assert len(calls) == before
    assert sess.match_cache[("aya", "look_glasses")] == pytest.approx(0.95)


def test_stale_lookahead_is_not_promoted():
    sess = _eig_session()
    question = traits.QUESTIONS_BY_ID["look_glasses"]
    sess.lookahead[("aya", "look_glasses")] = ((("gender_male", "yes", 1.0),), 0.95)
    sess.evidence["look_glasses"] = (question, "yes")
    engine._promote_lookahead(sess, question)
    assert ("aya", "look_glasses") not in sess.match_cache
    assert ("aya", "look_glasses") not in sess.lookahead


def test_exhausted_lookahead_gain_guesses_before_the_cap(monkeypatch):
    """Nothing left separates the top of the pool, so asking only spends turns."""
    sess = sess_mod.new_session()
    sess.candidates = [
        # One work: "Which series?" would otherwise be a real split.
        Candidate(id="lead", name="Lead Girl", series="Work A", logodds=0.3),
        Candidate(id="other", name="Other Girl", series="Work A", logodds=0.0),
    ]
    # Turn 5: past the minimum, before the series pin would take the pick.
    _arm_guess(sess, "lead", turn=engine.MIN_QUESTIONS_BEFORE_GUESS)
    # This test isolates the exhausted-EIG branch from silent-profile
    # calibration; the cache values are the support signal under test.
    monkeypatch.setattr(engine, "SILENT_WEIGHT", 1.0)

    def flat(state, questions):
        if "ready_to_guess" in questions:
            return {"ready_to_guess": {"noul": 0.1, "action": {"act_probability": 0.1}}}
        return {k: {"noul": 0.5, "action": {"act_probability": 0.5}} for k in questions}

    monkeypatch.setattr(laya_client, "ask", flat)
    state = engine._advance(sess)
    assert sess.last_eig is not None and sess.last_eig < engine.EIG_EXHAUSTED
    assert state["stage"] == "guessing"


def test_without_laya_the_exhausted_gate_stays_off():
    sess = sess_mod.new_session()
    sess.candidates = [
        # One work: "Which series?" would otherwise be a real split.
        Candidate(id="lead", name="Lead Girl", series="Work A", logodds=0.3),
        Candidate(id="other", name="Other Girl", series="Work A", logodds=0.0),
    ]
    # Turn 5: past the minimum, before the series pin would take the pick.
    _arm_guess(sess, "lead", turn=engine.MIN_QUESTIONS_BEFORE_GUESS)
    state = engine._advance(sess)
    assert sess.last_eig is None
    assert state["stage"] == "asking"
