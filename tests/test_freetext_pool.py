"""Off-option details must not floor the candidate pool.

The live miss: seed "pink hair, white dress, princess", then "angel" on
"Is your character human?". A raw clue noul of 0 dropped every candidate
who did not literally match that word.
"""

from __future__ import annotations

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
