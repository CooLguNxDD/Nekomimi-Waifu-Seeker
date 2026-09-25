"""Bench session memory keeps answers consistent and exportable."""

from __future__ import annotations

import json

import pytest

from waifu_engine.nekomimi import engine, laya_client, session as sess_mod
from waifu_engine.nekomimi.harness import SessionMemory
from waifu_engine.nekomimi.traits import QUESTIONS_BY_ID


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Heuristic scoring only, and no search."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    monkeypatch.setattr(engine, "ONLINE", False)


def test_memory_repeats_an_answer_and_forces_the_opposite_gender():
    """A later female question stays yes, and male is then no."""
    memory = SessionMemory(
        target="Lyra",
        traits={"species_elf": "yes", "gender_female": "yes"},
    )
    female = QUESTIONS_BY_ID["gender_female"]
    assert memory.answer_for(female) == "yes"
    memory.answers["gender_female"] = "yes"
    assert memory.answer_for(QUESTIONS_BY_ID["gender_male"]) == "no"
    assert memory.answer_for(female) == "yes"
    assert memory.answer_for(QUESTIONS_BY_ID["species_human"]) == "no"
    assert memory.answer_for(QUESTIONS_BY_ID["look_eyepatch"]) == "no"
    hair = QUESTIONS_BY_ID["hair_color"]
    assert memory.answer_for(hair) == "other"


def test_note_records_pool_chips_and_exports():
    """One honest turn stores the qid, a negated chip, the pool, and top names."""
    state = engine.start("not a demon")
    sess = sess_mod.get_session(state["session_id"])
    sess.add_candidates([
        {"id": "lyra", "name": "Lyra", "series": "Grove", "medium": "game",
         "tags": ["elf"], "blurb": "An elf archer.", "popularity": 20},
        {"id": "mara", "name": "Mara", "series": "Grove", "medium": "game",
         "tags": ["human"], "blurb": "A human knight.", "popularity": 10},
    ])
    engine._rescore_candidates(sess)
    question = QUESTIONS_BY_ID["species_elf"]
    sess.stage = "asking"
    sess.asked.append({
        "qid": question["id"], "text": question["text"], "category": question["category"],
        "kind": "yesno", "answer": None, "detail": None,
    })
    memory = SessionMemory(target="Lyra", seed=sess.seed, traits={"species_elf": "yes"})
    answer = memory.answer_for(question)
    engine.score_candidates(sess, question, answer)
    sess.asked[-1]["answer"] = answer
    record = memory.note(sess, answer=answer, detail="not a demon")
    assert record.qid == "species_elf"
    assert record.answer == "yes"
    assert record.pool_size == 2
    assert record.top[0]["name"] == "Lyra"
    assert "species_demon" in record.negated
    assert "demon" not in record.residual
    dumped = json.loads(memory.dumps())
    assert dumped["target"] == "Lyra"
    assert dumped["turns"][0]["stage"] == sess.stage
    assert dumped["answers"]["species_elf"] == "yes"
    assert dumped["answers"]["species_demon"] == "no"
    assert memory.answer_for(question) == "yes"
