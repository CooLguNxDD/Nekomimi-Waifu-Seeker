"""Decision prompts stay short; the round-level state stays inside the budget.

typed-decisions scored a 140-character instruction at 0.38 and the same
question in about 50 characters at 0.71. Elaboration belongs in the true/false
option text. The English checkpoint is not the default: its noul temperature
flattened scores, and on this checkpoint noul is the stronger primitive.
"""

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


def test_default_checkpoint_stays_typed_decisions():
    """English noul at temperature 1.98 flattened character scores."""
    assert laya_client.SUBFOLDER == "typed-decisions"
    assert laya_client.MAX_LEN == 512


def test_round_state_leads_with_short_profiles():
    """Ten full blurbs used to be truncated before the question could see them."""
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
    assert list(state)[:2] == ["goal", "characters"]
    assert len(state["characters"]) == 10
    assert all(len(profile) <= engine._LAYA_ROUND_PROFILE for profile in state["characters"].values())
    assert len(state["confirmed_facts"]) == engine._LAYA_ROUND_FACTS
    assert len(state["answer_history"]) == engine._LAYA_ROUND_HISTORY
    assert len(engine._POOL_FITS["pool_fits"]["instructions"]) <= 64
