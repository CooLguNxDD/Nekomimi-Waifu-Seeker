"""Decision prompts stay short; the round-level state stays inside the budget.

typed-decisions scored a 140-character instruction at 0.38 and the same
question in about 50 characters at 0.71. Elaboration belongs in the true/false
option text. The English checkpoint is not the default: its noul temperature
flattened scores, and on this checkpoint noul is the stronger primitive.
"""

import json

from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.nekomimi.session import Candidate


def test_match_instructions_stay_near_the_measured_length():
    """No bank instruction carries the long clause; criteria still do."""
    for question in traits.QUESTION_BANK:
        assert len(question["instructions"]) <= 64, question["id"]
        assert "`candidate`" in question["instructions"]
    soldier = traits.QUESTIONS_BY_ID["job_soldier"]
    assert soldier["instructions"] == "Is the character in `candidate` a soldier or fighter?"
    assert "mercenary" in soldier["criteria"]["true"]
    hair = traits.QUESTIONS_BY_ID["hair_long"]
    assert hair["instructions"].endswith("long-haired?")
    assert "long hair" in hair["criteria"]["true"]
    eyes = traits.QUESTIONS_BY_ID["eyes_heterochromia"]
    assert "two eye colours" not in eyes["instructions"]
    assert "two different eye colours" in eyes["criteria"]["true"]


def test_default_checkpoint_stays_typed_decisions(monkeypatch):
    """English noul at temperature 1.98 flattened character scores.

    An empty ``WAIFU_LAYA_SUBFOLDER`` is a set variable, so the default in
    ``getenv`` would not apply. Reload after unsetting it.
    """
    import importlib

    monkeypatch.delenv("WAIFU_LAYA_SUBFOLDER", raising=False)
    reloaded = importlib.reload(laya_client)
    assert reloaded.SUBFOLDER == "typed-decisions"
    assert reloaded.MAX_LEN == 512


def test_round_state_leads_with_answered_traits():
    """Profiles stay short. Trait rows lead so the window cuts them last."""
    sess = sess_mod.new_session("")
    blob = "alpha " * 200
    sess.candidates = [
        Candidate(id=f"c{i}", name=f"Name {i}", series="Example", medium="anime", blurb=blob)
        for i in range(10)
    ]
    for _ in range(12):
        sess.constraints.append("a confirmed fact that is not the latest one")
        sess.asked.append({
            "qid": f"q{_}", "text": "Is your character tall?", "answer": "yes", "detail": "",
        })
    state = engine._laya_state(sess, sess.scoring_pool())
    assert list(state)[:2] == ["answered_traits", "goal"]
    assert state["answered_traits"]["fields"] == ["id", "answer", "score"]
    assert len(state["characters"]) == 10
    assert all(len(profile) <= engine._LAYA_ROUND_PROFILE for profile in state["characters"].values())
    assert len(state["confirmed_facts"]) == engine._LAYA_ROUND_FACTS
    assert len(state["answer_history"]) == engine._LAYA_ROUND_HISTORY
    assert len(engine._POOL_FITS["pool_fits"]["instructions"]) <= 64


def _long_ids() -> list[str]:
    """Bank ids, longest first, so the trait block is as wide as it gets."""
    return sorted((q["id"] for q in traits.QUESTION_BANK), key=lambda qid: (-len(qid), qid))


def test_answered_traits_survive_the_english_window():
    """Ids and scores are unchanged after the 512-token right cut.

    The previous order put ten profiles first. The same rows, moved to the
    end of that JSON, do not survive — that was the truncation miss.
    """
    sess = sess_mod.new_session("")
    blob = "alpha " * 200
    sess.candidates = [
        Candidate(id=f"c{i}", name=f"Name {i} of a very long series title", series="Example",
                  medium="anime", blurb=blob)
        for i in range(10)
    ]
    for index, qid in enumerate(_long_ids()[:20]):
        question = dict(traits.QUESTIONS_BY_ID[qid])
        answer = "no" if index % 2 else "yes"
        if traits.is_choice(question):
            answer = next(iter(question["options"]))
        sess.evidence[qid] = (question, answer)
    # Soft chip: must stay off 1.0 / 0.0, and still survive the cut.
    chip = dict(traits.QUESTIONS_BY_ID["species_angel"])
    chip["soft_chip"] = True
    sess.evidence["species_angel"] = (chip, "yes")
    sess.llm_queries = ["rewrite prose the model must not treat as evidence"]
    for _ in range(12):
        sess.constraints.append("soft history padding " * 30)
        sess.asked.append({
            "qid": f"q{_}", "text": "Is your character tall? " * 8,
            "answer": "yes", "detail": "typed detail " * 12,
        })
    state = engine._laya_state(sess, sess.scoring_pool())
    full = engine._trait_signature(state)
    assert full
    assert ("species_angel", "yes", engine._SOFT_YES_SCORE) in full
    assert engine._traits_surviving_window(state) == full
    packed = json.dumps(state["answered_traits"])
    assert "rewrite prose" not in packed
    trailing = {
        "goal": state["goal"],
        "characters": state["characters"],
        "confirmed_facts": state["confirmed_facts"],
        "answer_history": state["answer_history"],
        "questions_asked": state["questions_asked"],
        "answered_traits": state["answered_traits"],
    }
    assert engine._traits_surviving_window(trailing) != full


def test_ready_state_carries_match_evidence_and_match_still_scores(monkeypatch):
    """A bank yes is in the shared window, and a later match still separates candidates."""
    sess = sess_mod.new_session("")
    sess.add_candidates([
        {"id": "miku", "name": "Hatsune Miku", "series": "Vocaloid", "medium": "game",
         "blurb": "A virtual singer.", "tags": ["female"], "popularity": 10},
        {"id": "mario", "name": "Mario", "series": "Super Mario", "medium": "game",
         "blurb": "A plumber.", "tags": ["male"], "popularity": 10},
    ])
    seen: list[dict] = []

    def ask(state, questions):
        seen.append({"state": state, "questions": set(questions)})
        if "match" in questions:
            profile = state.get("candidate") or ""
            return {"match": {"noul": 0.92 if "Miku" in profile else 0.08,
                              "action": {"act_probability": 0.7}}}
        if "pool_fits" in questions:
            return {"pool_fits": {"noul": 0.2, "action": {"act_probability": 0.4}}}
        return {"ready_to_guess": {"noul": 0.1, "action": {"act_probability": 0.2}}}

    monkeypatch.setattr(laya_client, "ask", ask)
    engine.score_candidates(sess, traits.QUESTIONS_BY_ID["gender_female"], "yes")
    assert sess.posterior()[0][0].id == "miku"
    engine.score_candidates(sess, traits.QUESTIONS_BY_ID["age_adult"], "yes")
    match_states = [item["state"] for item in seen if "match" in item["questions"]]
    assert match_states
    later = match_states[-1]
    assert later["answered_traits"]["rows"] == [["gender_female", "yes", 1.0]]
    assert "age_adult" not in [row[0] for row in later["answered_traits"]["rows"]]
    assert "candidate" in later and later["candidate"]
    _question, _answers = engine._pick_question(sess)
    ready = next(item["state"] for item in seen if "ready_to_guess" in item["questions"])
    assert engine._traits_surviving_window(ready) == engine._trait_signature(ready)
    ids = [row[0] for row in ready["answered_traits"]["rows"]]
    assert ids == ["gender_female", "age_adult"]
    stuck = engine._search_stuck(sess)
    assert stuck is True
    pool = next(item["state"] for item in seen if "pool_fits" in item["questions"])
    assert [row[0] for row in pool["answered_traits"]["rows"]] == ids
