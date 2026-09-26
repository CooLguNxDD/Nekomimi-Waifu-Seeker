"""Seed enrich, series-lock pins, and mid-game detail strength.

Enrich is a fixed phrase list (lexicon/enrich.yml). It records bank trait
ids the player already stated. It does not write questions or name a person.
A detail answer still does not score the closed question; the soft band is
what moves the posterior.
"""

from __future__ import annotations

import math

import pytest

from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.nekomimi.enrich import enrich_hits
from waifu_engine.nekomimi.lexicon import ENRICH_TEMPLATES
from waifu_engine.nekomimi.lexicon.load import LexiconError, build_enrich
from waifu_engine.nekomimi.session import popularity_prior


@pytest.fixture
def _offline(monkeypatch):
    """Heuristic scoring only, and no search."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    monkeypatch.setattr(engine, "ONLINE", True)


def test_templates_name_bank_questions_and_not_jobs():
    """Every template answer is a real bank option. Jobs stay off this list."""
    qids = {row["qid"] for row in ENRICH_TEMPLATES}
    assert "job_soldier" not in qids
    assert {"hair_color", "look_halo", "look_wings", "look_horns"} <= qids
    for row in ENRICH_TEMPLATES:
        question = traits.QUESTIONS_BY_ID[row["qid"]]
        if row["kind"] == "choice":
            assert row["answer"] in question["options"]
        else:
            assert row["answer"] == "yes"
            assert question["kind"] == "yesno"


def test_duplicate_enrich_marker_fails_load():
    """A repeated phrase would silently change which answer is stored."""
    with pytest.raises(LexiconError, match="repeats marker"):
        build_enrich({
            "templates": [
                {"id": "a", "qid": "hair_color", "answer": "blonde", "kind": "choice",
                 "markers": ["blonde hair"]},
                {"id": "b", "qid": "hair_color", "answer": "black", "kind": "choice",
                 "markers": ["blonde hair"]},
            ],
        })


def test_enrich_hits_record_colour_and_drop_ambiguity():
    """Blonde is one option. Two colours, or a negated colour, do not pick it."""
    assert dict(enrich_hits("blonde hair")) == {"hair_color": "blonde"}
    assert dict(enrich_hits("no wings")) == {"look_wings": "no"}
    assert "hair_color" not in dict(enrich_hits("blonde hair and black hair"))
    assert enrich_hits("not blonde hair") == []
    assert enrich_hits("soldier with a white coat") == []


def test_blonde_seed_is_answered_before_the_first_question(_offline):
    """The hair row exists before the pick, and hair colour is not asked."""
    sess = sess_mod.new_session("blonde hair")
    engine._score_free_text(sess, sess.seed, "clue_seed")
    rows = engine._answered_trait_pack(sess)["rows"]
    hair = next(row for row in rows if row[0] == "hair_color")
    assert hair[1] == "blonde"
    assert hair[2] == engine._SOFT_YES_SCORE
    assert hair[2] not in (0.0, 1.0)
    state = engine.start("blonde hair")
    started = sess_mod.get_session(state["session_id"])
    assert state["question"]["qid"] != "hair_color"
    packed = engine._answered_trait_pack(started)["rows"]
    assert ["hair_color", "blonde", engine._SOFT_YES_SCORE] in packed
    started.asked.append({
        "qid": "series", "text": "Which series?", "answer": "other", "kind": "choice",
        "detail": "fullmetal alchemist",
        "options": {"other": {"label": "Another series"}},
    })
    assert "hair_color" not in [q["id"] for q in engine.candidate_questions(started)]
    assert "look_halo" not in engine._pinned_question_ids(started)


def test_series_lock_leaves_unnamed_angel_kit_to_information_gain(_offline):
    """Without halo, wings or horns in the text, those questions are not pinned.

    Gender splits the cast and the kit does not, so information gain can ask
    gender first. A prosthetic that does split the cast is still pinned.
    """
    sess = sess_mod.new_session("a swordsman")
    sess.add_candidates([
        {"id": "ada", "name": "Ada", "series": "Fullmetal Alchemist", "medium": "anime",
         "tags": ["female"], "blurb": "She leads the guard.", "popularity": 10},
        {"id": "bran", "name": "Bran", "series": "Fullmetal Alchemist", "medium": "anime",
         "tags": ["male"], "blurb": "He keeps the gate.", "popularity": 10},
    ])
    sess.asked.append({
        "qid": "series", "text": "Which series?", "answer": "other", "kind": "choice",
        "detail": "fullmetal alchemist",
        "options": {"other": {"label": "Another series"}},
    })
    pins = engine._pinned_question_ids(sess)
    for qid in ("look_halo", "look_wings", "look_horns", "hair_color"):
        assert qid not in pins
    first = engine.candidate_questions(sess)[0]["id"]
    assert first not in {"look_halo", "look_wings", "look_horns"}
    assert first.startswith("gender_")

    split = sess_mod.new_session("blonde hair")
    engine._score_free_text(split, split.seed, "clue_seed")
    split.add_candidates([
        {"id": "ed", "name": "Edward Elric", "series": "Fullmetal Alchemist",
         "medium": "anime", "tags": ["male", "prosthetic"], "blurb": "A short alchemist."},
        {"id": "roy", "name": "Roy Mustang", "series": "Fullmetal Alchemist",
         "medium": "anime", "tags": ["male"], "blurb": "A flame alchemist."},
    ])
    split.asked.append({
        "qid": "series", "text": "Which series?", "answer": "other", "kind": "choice",
        "detail": "fullmetal alchemist",
        "options": {"other": {"label": "Another series"}},
    })
    pins = engine._pinned_question_ids(split)
    assert pins[0] == "look_prosthetic"
    assert "hair_color" not in pins
    for qid in ("look_halo", "look_wings", "look_horns"):
        assert qid not in pins


def test_midgame_detail_moves_the_leader_without_a_hard_wipe(_offline):
    """One extra detail changes top-1. Both candidates stay in the pool.

    The fame gap sits between the old overlap spread (~0.3 nats) and the
    current one (``OVERLAP_HIT`` vs ``OVERLAP_MISS``, about 0.67 nats).
    ``soft_delta`` is that spread. The closed question is not stored as a no.
    """
    gap = popularity_prior(200) - popularity_prior(1)
    old_spread = math.log(0.60) - math.log(0.45)
    new_spread = math.log(traits.OVERLAP_HIT) - math.log(traits.OVERLAP_MISS)
    assert old_spread < gap < new_spread
    state = engine.start("")
    sess = sess_mod.get_session(state["session_id"])
    pending = state["question"]["qid"]
    sess.add_candidates([
        {"id": "famous", "name": "Famous Knight", "series": "Crown Tale", "medium": "anime",
         "blurb": "A knight who patrols the capital.", "tags": ["male"], "popularity": 200},
        {"id": "coat", "name": "Coat Knight", "series": "Crown Tale", "medium": "anime",
         "blurb": "A knight in a white coat.", "tags": ["male"], "popularity": 1},
    ])
    before = len(sess.alive_candidates())
    out = engine.submit_answer(sess, "detail", "white coat")
    assert "error" not in out
    assert pending not in sess.evidence
    alive = sess.alive_candidates()
    assert len(alive) == before == 2
    assert all(math.isfinite(c.logodds) for c in alive)
    assert sess.posterior()[0][0].id == "coat"
    note = out["timing"]["soft_delta"]
    spreads = [float(part.split(":")[1]) for part in note.split("+") if ":" in part]
    assert max(spreads) >= 0.5
    assert traits.ANSWER_WEIGHT["detail"] == 0.0
