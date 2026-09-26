"""Off-option details must not floor the candidate pool.

The live miss: seed "pink hair, white dress, princess", then "angel" on
"Is your character human?". A raw clue noul of 0 dropped every candidate
who did not literally match that word.
"""

from __future__ import annotations

import json

import pytest

from waifu_engine.nekomimi import engine, laya_client, session as sess_mod

PINK = (
    "She has long pink hair, golden eyes, and a white school ribbon. "
    "Her dress is described as pale, and she studies at an academy."
)


def _rows() -> list[dict]:
    """Four humans who share the seed's hair, plus one unrelated hero."""
    return [
        {"id": "aria", "name": "Aria", "series": "Crown Tale", "medium": "anime",
         "blurb": PINK + " She is a princess of the western castle.",
         "tags": ["female", "human", "pink"], "popularity": 100},
        {"id": "mio", "name": "Mio", "series": "Crown Tale", "medium": "anime",
         "blurb": PINK + " She is a knight in the same court.",
         "tags": ["female", "human", "pink"], "popularity": 80},
        {"id": "yuki", "name": "Yuki", "series": "Other Anime", "medium": "anime",
         "blurb": PINK + " She sings in the school choir.",
         "tags": ["female", "human", "pink"], "popularity": 50},
        {"id": "ken", "name": "Ken", "series": "Other Anime", "medium": "anime",
         "blurb": "A boy with black hair who fights demons in the city.",
         "tags": ["male", "human"], "popularity": 5000},
    ]


@pytest.fixture
def _harsh_laya(monkeypatch):
    """Any clue or match is a confident no. The old path then floored the pool."""
    def ask(state, questions):
        if "match" not in questions:
            return None
        return {"match": {"noul": 0.0, "action": {"act_probability": 0.1}}}

    monkeypatch.setattr(laya_client, "ask", ask)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    monkeypatch.setattr(engine, "ONLINE", True)


def _ask_human(sess) -> None:
    """Pend the human question the way the turn loop would."""
    q = engine.QUESTIONS_BY_ID["species_human"]
    sess.stage = "asking"
    sess.asked.append({
        "qid": q["id"], "text": q["text"], "category": q["category"],
        "instructions": q["instructions"], "tags_true": q["tags_true"],
        "tags_false": q["tags_false"], "prior": q["prior"], "kind": "yesno",
        "criteria": q["criteria"], "answer": None, "detail": None,
    })


def test_angel_on_the_human_question_keeps_the_pool(_harsh_laya):
    """"angel" is a soft species chip. Prior pink-hair fits stay viable."""
    state = engine.start("pink hair, white dress, princess")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates(_rows())
    engine._rescore_candidates(sess)
    _ask_human(sess)
    out = engine.submit_answer(sess, "angel")
    assert "error" not in out
    alive = {c.id for c in sess.alive_candidates()}
    assert alive >= {"aria", "mio", "yuki", "ken"}
    human = sess.evidence.get("species_human")
    assert human is None or human[1] != "no"
    chip = sess.evidence["species_angel"]
    assert chip[1] == "yes" and chip[0].get("soft_chip") is True
    ranked = {c.id: p for c, p in sess.posterior()}
    for cid in ("aria", "mio", "yuki"):
        assert ranked[cid] > 0.05
    assert max(ranked.values()) < 0.95


def test_detail_field_angel_matches_the_raw_answer(_harsh_laya):
    """The detail box and a bare "angel" reply take the same soft path."""
    state = engine.start("pink hair, white dress, princess")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates(_rows())
    engine._rescore_candidates(sess)
    _ask_human(sess)
    engine.submit_answer(sess, "detail", "angel")
    assert {c.id for c in sess.alive_candidates()} >= {"aria", "mio", "yuki"}
    assert sess.evidence["species_angel"][1] == "yes"


def test_mixed_chip_keeps_the_non_chip_words(_harsh_laya):
    """"princess in a white dress" still ranks a white dress, not only royalty."""
    dress = {
        "id": "seam", "name": "Seam", "series": "Crown Tale", "medium": "anime",
        "blurb": "She wears a white dress and sews costumes for the court.",
        "tags": ["female", "human"], "popularity": 10,
    }
    plain = {
        "id": "clerk", "name": "Clerk", "series": "Crown Tale", "medium": "anime",
        "blurb": "She works in a shop and wears a heavy coat.",
        "tags": ["female", "human"], "popularity": 10,
    }
    state = engine.start("princess in a white dress")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates([dress, plain, _rows()[0]])
    engine._rescore_candidates(sess)
    clues = [q.get("clues") or "" for q, _a in sess.evidence.values()]
    assert any("dress" in clue for clue in clues)
    assert sess.evidence["job_royalty"][1] == "yes"
    by = {c.id: c.logodds for c in sess.alive_candidates()}
    assert by["seam"] > by["clerk"]
    assert "aria" in by


def test_negated_chips_are_not_yes_evidence(_harsh_laya):
    """"not a demon" / "not a princess" must not boost those traits."""
    state = engine.start("not a demon")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates([
        {"id": "mio", "name": "Mio", "series": "Crown Tale", "medium": "anime",
         "blurb": "She is a knight in the same court.", "tags": ["female"], "popularity": 10},
        {"id": "imp", "name": "Imp", "series": "Crown Tale", "medium": "anime",
         "blurb": "She is a demon who hunts in the old city.", "tags": ["demon"], "popularity": 10},
    ])
    engine._rescore_candidates(sess)
    question, answer = sess.evidence["species_demon"]
    assert answer == "no" and question.get("soft_chip") is True
    demon = next(row for row in engine._answered_trait_pack(sess)["rows"] if row[0] == "species_demon")
    assert demon[2] == engine._SOFT_NO_SCORE
    by = {c.id: c.logodds for c in sess.alive_candidates()}
    assert by["mio"] > by["imp"]

    state = engine.start("not a princess")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates(_rows())
    engine._rescore_candidates(sess)
    assert sess.evidence["job_royalty"][1] == "no"
    by = {c.id: c.logodds for c in sess.alive_candidates()}
    assert by["mio"] > by["aria"]


def test_pink_hair_still_overlap_scores_the_dress(_harsh_laya):
    """A visual clue must not swallow the leftover words.

    "pink hair and a white dress" used to score only the hair. The dress
    stayed off the posterior. It stays a mild overlap, not a hard yes.
    """
    dress = {
        "id": "seam", "name": "Seam", "series": "Crown Tale", "medium": "anime",
        "blurb": "She has long pink hair and wears a white dress to court.",
        "tags": ["female", "pink"], "popularity": 10,
    }
    plain = {
        "id": "clerk", "name": "Clerk", "series": "Crown Tale", "medium": "anime",
        "blurb": "She has long pink hair and wears a heavy coat.",
        "tags": ["female", "pink"], "popularity": 10,
    }
    state = engine.start("pink hair and a white dress")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates([dress, plain])
    engine._rescore_candidates(sess)
    clues = {q["id"]: q.get("clues") or "" for q, _a in sess.evidence.values() if q.get("clues")}
    assert any("dress" in clue for clue in clues.values())
    assert "clue_seed_rest" in clues
    by = {c.id: c.logodds for c in sess.alive_candidates()}
    assert by["seam"] > by["clerk"]
    for question, answer in sess.evidence.values():
        if question.get("clues"):
            continue
        assert question.get("soft_chip") is True
        assert answer != "yes" or question["id"] != "hair_color"


def test_free_text_never_promotes_a_yaml_job_to_a_hard_yes(_harsh_laya):
    """"soldier" is a lexicon marker, not a chip. It must not answer job_soldier."""
    state = engine.start("soldier with a white coat")
    sess = sess_mod.get_session(state["session_id"])
    assert "job_soldier" not in sess.evidence
    for question, _answer in sess.evidence.values():
        assert question.get("clues") or question.get("soft_chip") is True
    rows = engine._answered_trait_pack(sess)["rows"]
    assert rows == [] or all(row[2] not in {0.0, 1.0} for row in rows)
    packed = engine._laya_state(sess, [])
    assert "soldier with a white coat" not in json.dumps(packed["answered_traits"])


def test_princess_chip_ranks_without_dropping_the_others(_harsh_laya):
    """Royalty is a nudge. Missing "princess" does not zero a pink-haired peer."""
    state = engine.start("pink hair, white dress, princess")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates(_rows())
    engine._rescore_candidates(sess)
    by = {c.id: c for c, _p in sess.posterior()}
    assert by["aria"].logodds > by["mio"].logodds
    assert by["mio"].alive and by["yuki"].alive
    assert "job_royalty" in sess.evidence
