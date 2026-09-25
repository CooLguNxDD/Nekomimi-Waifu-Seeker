"""Bench session memory keeps answers consistent and exportable."""

from __future__ import annotations

import json

import pytest

from waifu_engine import timing
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod
from waifu_engine.nekomimi.harness import SessionMemory, miss_pack
from waifu_engine.nekomimi.session import Candidate
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
    assert "miss" not in dumped


def _pool(*rows: tuple[str, str, str, float]) -> sess_mod.GuessSession:
    """A session whose candidates are ``(id, name, series, logodds)``."""
    sess = sess_mod.new_session()
    sess.candidates = [
        Candidate(id=cid, name=name, series=series, medium="anime", logodds=odds)
        for cid, name, series, odds in rows
    ]
    return sess


def _pin_series(sess, key: str = "fullmetalalchemist", fact: str = "Fullmetal Alchemist") -> None:
    """Record a hard series choice the way a confirmed chip stores it."""
    sess.asked.append({
        "qid": "series",
        "text": "Which series?",
        "answer": key,
        "detail": None,
        "options": {key: {"label": fact, "series_key": key, "fact": fact}},
    })


# Shape a 5-pack row keeps. Entropy of a 0.5/0.5 pair is exactly 1 bit.
_NEAR_TWIN_EXAMPLE = {
    "target": "Edward Elric",
    "target_in_shortlist": True,
    "top_two_posterior_margin": 0.0,
    "franchise_match_entropy": 1.0,
    "franchise_pair": "same-work",
    "trait_window": "ok",
    "defer": "look_prosthetic",
    "miss_label": "near-twin-fired",
}


def test_near_twin_miss_matches_the_example_json():
    """A same-work coin flip whose defer was a series_split is near-twin-fired."""
    sess = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 0.0),
        ("ed", "Edward Elric", "Fullmetal Alchemist", 0.0),
    )
    memory = SessionMemory(target="Edward Elric")
    memory.note(sess, timing={"defer": "look_prosthetic", "turn": 6})
    row = {"target": memory.target, **memory.record_miss(sess)}
    assert row == _NEAR_TWIN_EXAMPLE
    dumped = json.loads(memory.dumps())
    assert dumped["miss"]["miss_label"] == "near-twin-fired"
    assert dumped["miss"]["defer"] == "look_prosthetic"
    assert dumped["turns"][0]["defer"] == "look_prosthetic"


def test_uniform_same_work_pair_is_near_twin_even_when_the_margin_is_wide():
    """Renormalized pair mass under 0.75 counts when the raw margin is above 0.15."""
    import math

    sess = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", math.log(0.50)),
        ("ed", "Edward Elric", "Fullmetal Alchemist", math.log(0.30)),
        ("al", "Alphonse Elric", "Fullmetal Alchemist", math.log(0.20)),
    )
    memory = SessionMemory(target="Edward Elric", defer="look_prosthetic")
    miss = memory.record_miss(sess)
    assert miss["top_two_posterior_margin"] > 0.15
    assert miss["franchise_pair"] == "same-work"
    assert miss["franchise_match_entropy"] > 0.8
    assert miss["miss_label"] == "near-twin-fired"


def test_pin_thin_when_the_leader_is_clear_and_the_series_was_never_pinned():
    """In-pool target, no defer, no series chip: the pin is thin, not Laya."""
    sess = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 2.0),
        ("ed", "Edward Elric", "Fullmetal Alchemist", 0.0),
    )
    memory = SessionMemory(target="Edward Elric")
    miss = memory.record_miss(sess)
    assert miss["target_in_shortlist"] is True
    assert miss["top_two_posterior_margin"] > 0.15
    assert miss["defer"] is None
    assert miss["trait_window"] == "ok"
    assert miss["miss_label"] == "pin-thin"


def test_a_settled_pair_is_not_near_twin_just_because_defer_fired():
    """Defer counts only together with a close or near-uniform top two."""
    sess = _pool(
        ("mihawk", "Dracule Mihawk", "One Piece", 3.0),
        ("law", "Trafalgar Law", "One Piece", 0.0),
    )
    memory = SessionMemory(target="Trafalgar Law", defer="look_eyepatch")
    miss = memory.record_miss(sess)
    assert miss["top_two_posterior_margin"] > 0.15
    assert miss["franchise_pair"] == "same-work"
    assert miss["defer"] == "look_eyepatch"
    assert miss["miss_label"] == "laya-match"


def test_hard_series_pin_with_a_clear_leader_is_laya_match():
    """A confirmed series and no defer is the residual label, not a twin story."""
    sess = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 2.0),
        ("ed", "Edward Elric", "Fullmetal Alchemist", 0.0),
    )
    _pin_series(sess)
    memory = SessionMemory(target="Edward Elric")
    miss = memory.record_miss(sess)
    assert miss["miss_label"] == "laya-match"
    assert miss["target_in_shortlist"] is True


def test_truncation_wins_over_a_thin_pin_and_stays_on_the_record():
    """A truncated trait window is the label; the margin fields stay filled."""
    sess = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 2.0),
        ("ed", "Edward Elric", "Fullmetal Alchemist", 0.0),
    )
    memory = SessionMemory(target="Edward Elric")
    memory.note(sess, timing={"trait_window": "truncated"})
    memory.note(sess, timing={"turn": 4, "cands": "2/2"})
    miss = memory.record_miss(sess)
    assert memory.trait_window == "truncated"
    assert miss["trait_window"] == "truncated"
    assert miss["miss_label"] == "truncation"
    assert miss["target_in_shortlist"] is True


def test_absent_target_is_search_even_if_the_window_was_truncated():
    """Search wins when the known name never entered the shortlist."""
    sess = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 1.0),
        ("ed", "Edward Elric", "Fullmetal Alchemist", 0.0),
    )
    memory = SessionMemory(target="Frieren")
    memory.absorb_timing({"trait_window": "truncated"})
    miss = memory.finish(sess, correct=False)["miss"]
    assert miss["target_in_shortlist"] is False
    assert miss["trait_window"] == "truncated"
    assert miss["miss_label"] == "search"


def test_cross_work_top_two_still_export_entropy():
    """A pair from two works reports entropy and says so in franchise_pair."""
    sess = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 0.0),
        ("bruce", "Batman", "Batman", 0.0),
    )
    memory = SessionMemory(target="Batman")
    miss = memory.record_miss(sess)
    assert miss["franchise_pair"] == "cross-work"
    assert miss["franchise_match_entropy"] == 1.0
    assert miss["top_two_posterior_margin"] == 0.0
    assert miss["miss_label"] == "pin-thin"


def test_a_single_candidate_has_zero_margin_and_no_pair():
    """Fewer than two live names is margin 0, entropy 0, franchise_pair none."""
    sess = _pool(("ed", "Edward Elric", "Fullmetal Alchemist", 1.0))
    miss = SessionMemory(target="Edward Elric").record_miss(sess)
    assert miss["top_two_posterior_margin"] == 0.0
    assert miss["franchise_match_entropy"] == 0.0
    assert miss["franchise_pair"] == "none"
    assert miss["miss_label"] == "pin-thin"


def test_live_trace_notes_stick_and_a_hit_clears_the_miss():
    """timing.note during the turn is the channel; a later hit is not a miss."""
    sess = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 0.0),
        ("ed", "Edward Elric", "Fullmetal Alchemist", 0.0),
    )
    memory = SessionMemory(target="Edward Elric")

    @timing.traced("answer")
    def turn():
        timing.note(defer="look_animal_ears")
        memory.note(sess)
        return {}

    turn()
    assert memory.defer == "look_animal_ears"
    assert memory.record_miss(sess)["miss_label"] == "near-twin-fired"
    exported = memory.finish(sess, correct=True)
    assert "miss" not in exported


def test_miss_pack_lists_five_characters_with_stable_labels():
    """A 5-pack JSON has one row per miss, labels unchanged across dumps."""
    ed = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 0.0),
        ("ed", "Edward Elric", "Fullmetal Alchemist", 0.0),
    )
    holo = _pool(
        ("myuri", "Myuri", "Spice and Wolf", 2.0),
        ("holo", "Holo", "Spice and Wolf", 0.0),
    )
    batman = _pool(
        ("dick", "Dick Grayson", "Batman", 2.0),
        ("bruce", "Batman", "Batman", 0.0),
    )
    frieren = _pool(
        ("heinkel", "Heinkel", "Fullmetal Alchemist", 1.0),
        ("ed", "Edward Elric", "Fullmetal Alchemist", 0.0),
    )
    law = _pool(
        ("mihawk", "Dracule Mihawk", "One Piece", 2.0),
        ("law", "Trafalgar Law", "One Piece", 0.0),
    )
    _pin_series(law, key="onepiece", fact="One Piece")
    memories = [
        SessionMemory(target="Edward Elric", defer="look_prosthetic"),
        SessionMemory(target="Holo"),
        SessionMemory(target="Batman", trait_window="truncated"),
        SessionMemory(target="Frieren"),
        SessionMemory(target="Trafalgar Law"),
    ]
    sessions = [ed, holo, batman, frieren, law]
    for memory, sess in zip(memories, sessions):
        memory.record_miss(sess)
    pack = miss_pack(memories)
    assert [row["target"] for row in pack] == [
        "Edward Elric", "Holo", "Batman", "Frieren", "Trafalgar Law",
    ]
    assert [row["miss_label"] for row in pack] == [
        "near-twin-fired", "pin-thin", "truncation", "search", "laya-match",
    ]
    again = json.loads(json.dumps(pack, sort_keys=True))
    assert [row["miss_label"] for row in again] == [row["miss_label"] for row in pack]
    assert set(pack[0]) == {
        "target",
        "target_in_shortlist",
        "top_two_posterior_margin",
        "franchise_match_entropy",
        "franchise_pair",
        "trait_window",
        "defer",
        "miss_label",
    }
