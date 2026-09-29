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


def test_negation_stops_at_a_clause_break():
    """A negator in an earlier clause does not reach the next trait."""
    hits = dict(enrich_hits("no wings, blonde hair"))
    assert hits == {"hair_color": "blonde", "look_wings": "no"}
    hits = dict(enrich_hits("no hat, blonde hair and horns"))
    assert hits["hair_color"] == "blonde"
    assert hits["look_horns"] == "yes"
    assert dict(enrich_hits("not blonde hair, golden hair")) == {"hair_color": "blonde"}
    assert dict(enrich_hits("no wings but halo")) == {"look_wings": "no", "look_halo": "yes"}
    assert traits._clue_requirements("no wings but halo") == [("halo", True), ("wings", False)]


def test_later_affirmative_mention_is_not_hidden_by_a_negated_one():
    """The first, negated, occurrence must not end the scan."""
    hits = dict(enrich_hits("not blonde hair as a child; now she has blonde hair"))
    assert hits == {"hair_color": "blonde"}


def test_halo_the_game_is_not_a_halo():
    """The series title does not record look_halo or a visual halo clue."""
    for text in ("a character from Halo", "Halo Infinite spartan", "plays Halo 3"):
        assert "look_halo" not in dict(enrich_hits(text))
        assert traits._clue_requirements(text) == []
    assert dict(enrich_hits("from Halo, she has a halo"))["look_halo"] == "yes"


def test_clue_covered_enrich_row_adds_no_nats(_offline):
    """Halo is scored once, by the visual clue, not again by the enrich row."""
    sess = sess_mod.new_session("halo")
    sess.add_candidates([
        {"id": "a", "name": "Angel A", "series": "S", "medium": "game",
         "blurb": "A girl with a halo.", "tags": ["halo"], "popularity": 10},
        {"id": "b", "name": "Plain B", "series": "S", "medium": "game",
         "blurb": "A girl.", "tags": [], "popularity": 10},
    ])
    engine._score_free_text(sess, "a halo", "clue_seed")
    question, answer = sess.evidence["look_halo"]
    assert question["soft_enrich"] and question["clue_covered"] and answer == "yes"
    with_row = {c.id: c.logodds for c in sess.alive_candidates()}
    sess.evidence.pop("look_halo")
    engine._rescore_candidates(sess)
    without_row = {c.id: c.logodds for c in sess.alive_candidates()}
    assert with_row == pytest.approx(without_row)
    sess.evidence["look_halo"] = (question, answer)
    assert "look_halo" not in [q["id"] for q in engine.candidate_questions(sess)]


def test_soft_chip_does_not_settle_a_bank_question(_offline):
    """Only enrich rows and real answers skip the ask. A chip is a nudge."""
    sess = sess_mod.new_session("")
    qid = next(q["id"] for q in traits.QUESTION_BANK if not traits.is_choice(q))
    chip = dict(traits.QUESTIONS_BY_ID[qid])
    chip["soft_chip"] = True
    sess.evidence[qid] = (chip, "yes")
    assert qid not in engine._settled_ids(sess)
    enriched = dict(chip, soft_enrich=True)
    sess.evidence[qid] = (enriched, "yes")
    assert qid in engine._settled_ids(sess)


def test_later_text_replaces_soft_enrich_and_drops_stale_votes(_offline):
    """Blonde, then "actually black hair": the old vote leaves with the row."""
    state = engine.start("blonde hair")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates([
        {"id": "gold", "name": "Gold Girl", "series": "S", "medium": "anime",
         "blurb": "A girl with blonde hair.", "tags": ["blonde"], "popularity": 10},
        {"id": "raven", "name": "Raven Girl", "series": "S", "medium": "anime",
         "blurb": "A girl with black hair.", "tags": ["black"], "popularity": 10},
    ])
    assert sess.evidence["hair_color"][1] == "blonde"
    # A cached Laya vote cast while the row said blonde.
    sess.match_cache[("gold", "hair_color")] = 0.95
    sess.choice_cache[("gold", "hair_color")] = {"blonde": 0.9, "black": 0.1}
    out = engine.submit_answer(sess, "detail", "actually black hair")
    assert "error" not in out
    question, answer = sess.evidence["hair_color"]
    assert question["soft_enrich"] and answer == "black"
    assert not [key for key in sess.match_cache if key[1] == "hair_color"]
    assert not [key for key in sess.choice_cache if key[1] == "hair_color"]
    note = out["timing"]["soft_delta"]
    assert "hair_color=black:" in note
    assert "hair_color=blonde" not in note
    assert sess.posterior()[0][0].id == "raven"
    assert "hair_color" not in [q["id"] for q in engine.candidate_questions(sess)]


def test_hard_answer_is_not_replaced_by_later_text(_offline):
    """A bank answer the player clicked stays; only soft_enrich rows move."""
    sess = sess_mod.new_session("")
    question = dict(traits.QUESTIONS_BY_ID["hair_color"])
    sess.evidence["hair_color"] = (question, "blonde")
    engine._score_enrich(sess, "black hair")
    assert sess.evidence["hair_color"][1] == "blonde"
    assert not sess.evidence["hair_color"][0].get("soft_enrich")
