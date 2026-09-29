import os
import json
import time
import pytest
from waifu_engine.nekomimi.memory import (
    SessionMemory, proposal_facts, refresh_session_memory, working_memory,
)
from waifu_engine.nekomimi.session import (
    GuessSession, new_session, save_session, load_session, purge_expired,
    Candidate, SESSION_DIR,
)
from waifu_engine.nekomimi.engine import score_candidates
from waifu_engine.sources.cache import get_entity, save_entity, CACHE_DIR
import math

def test_session_memory_serialization():
    mem = SessionMemory()
    mem.confirmed_traits["hair"] = "blonde"
    mem.soft_dispreferred.append("Not a demon")
    mem.player_details.append("has a big sword")
    mem.rejected_hypotheses.append({"name": "Cloud", "series": "FF7"})
    
    d = mem.to_dict()
    assert d["confirmed_traits"] == {"hair": "blonde"}
    assert "Cloud" in str(d["rejected_hypotheses"])
    
    mem2 = SessionMemory.from_dict(d)
    assert mem2.confirmed_traits == mem.confirmed_traits
    assert mem2.player_details == mem.player_details

def test_file_system_session_save_load_purge():
    sess = new_session()
    sess.memory.player_details.append("test_detail")
    
    save_session(sess)
    
    sess2 = load_session(sess.id)
    assert sess2 is not None
    assert sess2.id == sess.id
    assert sess2.memory.player_details == ["test_detail"]
    
    # purge
    path = os.path.join(SESSION_DIR, f"{sess.id}.json")
    old_time = time.time() - 4000
    sess.updated = old_time
    save_session(sess)
    os.utime(path, (old_time, old_time))
    
    purge_expired(now=time.time())
    
    assert load_session(sess.id) is None

def test_entity_cache():
    cand = {"name": "Test Char", "series": "Test Series", "tags": ["test"]}
    save_entity(cand)
    
    c = get_entity("Test Char")
    assert c is not None
    assert c["name"] == "Test Char"
    assert c["tags"] == ["test"]
    
def test_soft_penalty_and_detail_weight():
    """Keep soft negatives bounded and scale typed-detail evidence continuously."""
    sess = new_session()
    c = Candidate(id="test1", name="Test1")
    sess.candidates.append(c)
    
    question = {
        "id": "q1",
        "text": "Is she a test?",
        "category": "test",
        "instructions": "",
        "tags_true": ["test"],
        "tags_false": [],
        "prior": 0.5,
        "kind": "yesno"
    }
    # Test soft penalty (NO answer)
    c.logodds = 0.0
    c.tags = ["test"]
    sess.evidence["q1"] = (question, "no")
    sess.match_cache[("test1", "q1")] = 1.0 # bypass laya, simulate 1.0 probability
    
    from waifu_engine.nekomimi.engine import _rescore_candidates
    _rescore_candidates(sess)
    
    # since p_yes is 1.0 (has tag), likelihood of "no" is 0.0. Clamped to 0.18.
    # log(0.18) approx -1.714
    assert -1.8 <= c.logodds <= -1.6
    
    # Test detail boost (YES answer to detail clue)
    sess2 = new_session()
    c2 = Candidate(id="test2", name="Test2")
    sess2.candidates.append(c2)
    
    clue_q = {
        "id": "clue_1",
        "text": "detail text",
        "category": "clue",
        "instructions": "",
        "tags_true": [],
        "tags_false": [],
        "prior": 0.5,
        "clues": "detail text"
    }
    c2.blurb = "This is a detail text blurb."
    sess2.evidence["clue_1"] = (clue_q, "yes")
    # set visual match cache
    sess2.match_cache[("test2", "clue_1")] = 1.0
    
    _rescore_candidates(sess2)
    
    # A clue gets a continuous confidence multiplier, not the old 1.8 floor.
    expected = math.log(0.6) * (1.0 + 0.35 * abs(2.0 * 0.6 - 1.0))
    assert sess2.contrib[("test2", "clue_1")] == pytest.approx(expected)
    assert abs(c2.logodds) < 1.0


def test_working_memory_keeps_sources_turns_and_semantic_other_choices():
    """Keep each explicit answer traceable and omit rejected names from proposals."""
    sess = new_session("pink-haired student from Blue Archive, unlike Rejected Name")
    sess.asked = [
        {
            "turn": 3, "qid": "series", "text": "Which series is your character from?",
            "answer": "other", "detail": "has long teal hair",
            "options": {
                "s1": {"label": "Blue Archive", "series_key": "blue_archive", "fact": ""},
                "s2": {"label": "Violet Evergarden", "series_key": "violet_evergarden", "fact": ""},
                "other": {"label": "Another series", "series_key": "", "fact": ""},
            },
        },
        {
            "turn": 4, "qid": "gender_female", "text": "Is your character female?",
            "answer": "yes", "detail": None, "options": {},
        },
    ]
    sess.exclusion_log = [{
        "candidate_id": "wrong", "name": "Rejected Name", "series": "Other Work",
        "reason": "wrong_guess_identity", "turn": 5,
    }]

    facts = refresh_session_memory(sess)
    assert sess.memory.confirmed_traits == {"series": "other", "gender_female": "yes"}
    assert any(fact["source"] == "seed" and fact["turn"] == 0 for fact in facts)
    detail = next(fact for fact in facts if fact["source"] == "player_detail")
    assert (detail["value"], detail["turn"], detail["fact_id"]) == (
        "has long teal hair", 3, "detail:3",
    )
    series = next(fact for fact in facts if fact["question_id"] == "series")
    assert series["polarity"] == "none_of_offered"
    assert [row["label"] for row in series["offered"]] == [
        "Blue Archive", "Violet Evergarden", "Another series",
    ]
    sent = "\n".join(proposal_facts(sess))
    assert "Which series is your character from: none of these offered choices " \
           "(Blue Archive, Violet Evergarden)" in sent
    assert "pink-haired student from Blue Archive" in sent
    assert "has long teal hair" in sent and "Rejected Name" not in sent
    assert working_memory(sess) == facts


def test_repeat_visual_detail_does_not_count_already_answered_button_evidence():
    """Do not score one visual requirement again through typed detail text."""
    from waifu_engine.nekomimi import engine, traits

    sess = new_session()
    button = dict(traits.QUESTIONS_BY_ID["look_wings"])
    sess.evidence["look_wings"] = (button, "yes")
    candidate = Candidate(id="angel", name="Angel", blurb="A winged guardian.", tags=["wings"])
    clue = traits.clue_question("clue_1", "she has wings")
    assert engine._candidate_answer_likelihood(sess, clue, "yes", candidate) == 0.5

    earlier = traits.clue_question("clue_0", "she has wings")
    sess.evidence["clue_0"] = (earlier, "yes")
    repeated = traits.clue_question("clue_2", "she has wings")
    assert engine._candidate_answer_likelihood(sess, repeated, "yes", candidate) == 0.5


def test_session_restart_restores_evidence_trace_pending_action_and_cache(monkeypatch):
    """Reload every durable decision field while making a lost worker retryable."""
    from waifu_engine.nekomimi import engine, session as sess_mod, traits

    monkeypatch.setattr(engine.laya_client, "available", lambda: False)
    sess_mod._STORE.clear()
    sess = new_session("pink hair")
    sess.candidates = [Candidate(id="c1", name="Mika", series="Blue Archive", medium="anime")]
    sess.turn = 6
    sess.stage = "guessing"
    sess.pending_guess = "c1"
    sess.guesses_made = 1
    sess.asked = [{
        "qid": "gender_female", "text": "Is your character female?", "answer": "yes",
        "detail": "", "category": "gender", "kind": "yesno", "options": {},
    }]
    question = dict(traits.QUESTIONS_BY_ID["gender_female"])
    question.update({"source": "button", "turn": 6, "fact_id": "button:gender_female:6"})
    sess.evidence["gender_female"] = (question, "yes")
    sess.match_cache[("c1", "gender_female")] = 0.91
    sess.match_fingerprints[("c1", "gender_female")] = "match-fingerprint"
    sess.choice_cache[("c1", "hair_color")] = {"black": 0.2, "pink": 0.8}
    sess.choice_fingerprints[("c1", "hair_color")] = "choice-fingerprint"
    sess.lookahead[("c1", "look_halo")] = ((
        ("gender_female", "yes", 1.0),
    ), 0.83)
    sess.lookahead_fingerprints[("c1", "look_halo")] = "look-fingerprint"
    sess.contrib[("c1", "gender_female")] = -0.2
    sess.rank_log.append((6, "gender_female", "yes", [("c1", "Mika", 0.9)]))
    sess.guess_log.append({
        "id": "c1", "name": "Mika", "turn": 6, "probability": 0.9,
        "contrib": {("c1", "gender_female"): -0.2},
        "ranked": [("c1", "Mika", 0.9)],
    })
    sess.exclusion_log.append({"candidate_id": "old", "name": "Old", "reason": "wrong_guess_identity"})
    sess.merge_log.append({"candidate_id": "alias", "keeper_id": "c1", "reason": "same_character_identity"})
    sess.context_log["match"] = {
        "turn": 6, "fingerprint": "context-fingerprint",
        "included_fact_ids": ["button:gender_female:6"], "omitted_fact_ids": [],
    }
    sess.revision = 9
    sess.evidence_revision = 7
    sess.proposal_calls = 2
    sess.proposal_fingerprint = "proposal-fingerprint"
    sess.proposal_revision = 7
    sess.proposal_turn = 6
    sess.proposal_pending = True
    save_session(sess)

    sess_mod._STORE.clear()  # simulate a fresh process
    restored = load_session(sess.id)
    assert restored is not None
    assert restored.evidence["gender_female"][1] == "yes"
    assert restored.match_cache[("c1", "gender_female")] == 0.91
    assert restored.match_fingerprints[("c1", "gender_female")] == "match-fingerprint"
    assert restored.choice_cache[("c1", "hair_color")] == {"black": 0.2, "pink": 0.8}
    assert restored.lookahead[("c1", "look_halo")][1] == 0.83
    assert restored.rank_log[0][1:3] == ("gender_female", "yes")
    assert restored.guess_log[0]["contrib"][("c1", "gender_female")] == -0.2
    assert restored.exclusion_log[0]["reason"] == "wrong_guess_identity"
    assert restored.merge_log[0]["candidate_id"] == "alias"
    assert restored.context_log["match"]["included_fact_ids"] == ["button:gender_female:6"]
    assert restored.proposal_pending is False  # process-local worker cannot survive restart
    assert restored.proposal_calls == 1  # pending request is made retryable
    assert engine.state_payload(restored)["guess"]["id"] == "c1"
    assert engine.trace_payload(restored, "Mika")["turns"][0]["target"]["name"] == "Mika"


def test_legacy_session_schema_loads_and_discards_unfingerprinted_judgments():
    """Upgrade old sessions without trusting judgments that lack input fingerprints."""
    from waifu_engine.nekomimi.session import _from_json_dict

    legacy = _from_json_dict({
        "id": "legacy", "created": 1, "updated": 1,
        "candidates": [{"id": "c", "name": "Character"}],
        "match_cache": {"('c', 'gender_female')": 0.9},
        "choice_cache": {"('c', 'hair_color')": {"pink": 1.0}},
        "rank_log": [[1, "gender_female", "yes", [["c", "Character", 1.0]]]],
    })
    assert legacy.id == "legacy" and legacy.schema_version == 2
    assert legacy.match_cache == {} and legacy.choice_cache == {}
    assert legacy.match_fingerprints == {} and legacy.lookahead == {}
    assert legacy.rank_log == [(1, "gender_female", "yes", [("c", "Character", 1.0)])]
