"""Decision prompts stay short; the round-level state stays inside the budget.

typed-decisions scored a 140-character instruction at 0.38 and the same
question in about 50 characters at 0.71. Elaboration belongs in the true/false
option text. The English checkpoint is not the default: its noul temperature
flattened scores, and on this checkpoint noul is the stronger primitive.
"""

import json

from waifu_engine.nekomimi import context as laya_context, engine, laya_client, session as sess_mod, traits
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
    """Keep durable answer rows and profile identities inside the shared window."""
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
    assert len(state["confirmed_facts"]) <= engine._LAYA_ROUND_FACTS
    assert len(state["answer_history"]) <= engine._LAYA_ROUND_HISTORY
    assert laya_context._state_size(state, laya_client.tokenizer()) <= engine._shared_state_room()
    assert len(engine._POOL_FITS["pool_fits"]["instructions"]) <= 64


def _long_ids() -> list[str]:
    """Bank ids, longest first, so the trait block is as wide as it gets."""
    return sorted((q["id"] for q in traits.QUESTION_BANK), key=lambda qid: (-len(qid), qid))


def test_large_round_context_is_bounded_and_accounts_for_every_fact():
    """A 48-answer round fits the model window and logs its dropped fact IDs."""
    sess = sess_mod.new_session("")
    blob = "alpha " * 200
    sess.candidates = [
        Candidate(id=f"c{i}", name=f"Name {i} of a very long series title", series="Example",
                  medium="anime", blurb=blob)
        for i in range(10)
    ]
    for index in range(48):
        qid = f"synthetic_trait_{index:02d}"
        question = {
            "id": qid, "text": f"Synthetic question {index}?", "category": "synthetic",
            "kind": "yesno", "instructions": "Is `candidate` described this way?",
            "tags_true": [f"trait_{index}"], "tags_false": [], "prior": 0.5,
            "source": "button", "turn": index + 1, "fact_id": f"button:{qid}:{index + 1}",
        }
        answer = "no" if index % 2 else "yes"
        sess.evidence[qid] = (question, answer)
    # A soft chip remains a soft score and participates in the same audit.
    chip = dict(traits.QUESTIONS_BY_ID["species_angel"])
    chip.update({"soft_chip": True, "source": "inferred_chip", "turn": 49,
                 "fact_id": "chip:species_angel:49"})
    sess.evidence["species_angel"] = (chip, "yes")
    source_rows = engine._answered_trait_pack(sess)["rows"]
    sess.llm_queries = ["rewrite prose the model must not treat as evidence"]
    for _ in range(12):
        sess.constraints.append("soft history padding " * 30)
        sess.asked.append({
            "qid": f"q{_}", "text": "Is your character tall? " * 8,
            "answer": "yes", "detail": "typed detail " * 12,
        })
    state = engine._laya_state(sess, sess.scoring_pool())
    packed_rows = engine._trait_signature(state)
    assert len(sess.evidence) == 49
    assert packed_rows
    assert ["species_angel", "yes", engine._SOFT_YES_SCORE] in source_rows
    assert laya_context._state_size(state, laya_client.tokenizer()) <= engine._shared_state_room()
    context = sess.context_log["shared"]
    expected_ids = {f"button:synthetic_trait_{index:02d}:{index + 1}" for index in range(48)}
    expected_ids.add("chip:species_angel:49")
    included = set(context["included_fact_ids"])
    omitted = set(context["omitted_fact_ids"])
    assert included | omitted == expected_ids
    assert included & omitted == set()
    assert omitted
    packed = json.dumps(state["answered_traits"])
    assert "rewrite prose" not in packed


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


def test_match_cache_rejudges_after_model_or_profile_fingerprint_changes(monkeypatch):
    """Discard a judgment when either its model or bounded candidate profile changes."""
    sess = sess_mod.new_session()
    sess.candidates = [Candidate(id="c1", name="Mika", blurb="A pilot.")]
    calls = []
    monkeypatch.setattr(laya_client, "ask", lambda state, questions:
                        calls.append(state) or {"match": {"noul": 0.9}})
    question = traits.QUESTIONS_BY_ID["gender_female"]

    engine.score_candidates(sess, question, "yes")
    original = sess.match_fingerprints[("c1", "gender_female")]
    assert len(calls) == 1

    monkeypatch.setattr(laya_client, "MODEL_ID", "replacement-model")
    engine._rescore_candidates(sess)
    model_changed = sess.match_fingerprints[("c1", "gender_female")]
    assert model_changed != original
    assert len(calls) == 2

    sess.candidates[0].blurb = "Mika is described as a woman pilot."
    engine._rescore_candidates(sess)
    assert sess.match_fingerprints[("c1", "gender_female")] != model_changed
    assert len(calls) == 3


def test_input_fingerprint_tracks_the_model_used():
    """Include model identity in the cache key even when state and question match."""
    question = {"id": "q", "instructions": "Is `candidate` human?"}
    state = {"goal": "identify a character"}
    first = laya_context.build_bounded_context(
        state, [question], max_len=512, head_max_len=480,
        state_fingerprint_context={"model": "first"},
    )
    second = laya_context.build_bounded_context(
        state, [question], max_len=512, head_max_len=480,
        state_fingerprint_context={"model": "second"},
    )
    assert first.input_fingerprint(question) != second.input_fingerprint(question)
