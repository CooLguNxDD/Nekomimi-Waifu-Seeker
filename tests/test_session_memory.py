import os
import json
import time
from waifu_engine.nekomimi.memory import SessionMemory
from waifu_engine.nekomimi.session import GuessSession, new_session, save_session, load_session, purge_expired, Candidate, SESSION_DIR
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
    
    # detail weight boosts to 1.8 log odds
    assert c2.logodds >= 1.7
