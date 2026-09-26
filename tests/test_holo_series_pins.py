"""Spice and Wolf look pins: Holo versus Kraft Lawrence.

Empty-seed benches guessed the merchant while Holo was still in the pool
(pin-thin, defer null, margin about 0.31, entropy about 0.84). The lexicon
pin makes that pair a series_split defer, and wolf-deity answers prefer
Holo inside the cast. The question bank and the miss-log labels stay put.
"""

from __future__ import annotations

import pytest

from waifu_engine import web_search
from waifu_engine.names import identity_id, same_character, same_series
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.nekomimi.harness import MISS_KEYS, SessionMemory
from waifu_engine.nekomimi.session import Candidate


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Heuristic scoring only, unless a test installs its own Laya stub."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)


def _pair(holo_series="Spice and Wolf", lawrence_series="Spice and Wolf"):
    """Thin Holo blurb, merchant page that talks about her ears, Lawrence more famous."""
    return [
        {"id": "holo", "name": "Holo", "series": holo_series, "medium": "anime",
         "popularity": 1000, "tags": ["female"], "blurb": "A traveler."},
        {"id": "lawrence", "name": "Kraft Lawrence", "series": lawrence_series,
         "medium": "anime", "popularity": 80000,
         "tags": ["male", "animal-ears", "tail", "god", "beast"],
         "blurb": "A merchant who travels with the wise wolf. She has a wolf tail and ears."},
    ]


def _rank(rows, asked):
    """Posterior names after replaying ``asked`` over ``rows``."""
    sess = sess_mod.new_session()
    sess.add_candidates(rows)
    for qid, answer in asked:
        engine.score_candidates(sess, traits.QUESTIONS_BY_ID[qid], answer)
    return [cand.name for cand, _prob in sess.posterior()], sess


def _arm_guess(sess, leader_id: str) -> None:
    """Two strong model judgments and the stable-leader streak.

    One check only records the streak. The second is what ``_should_guess``
    accepts when the posterior is under the single-check bar of 0.80, which
    is the empty-seed shape this pair actually missed on.
    """
    sess.turn = 8
    for qid, prob in (("gender_male", 0.9), ("age_adult", 0.8)):
        question = traits.QUESTIONS_BY_ID[qid]
        sess.evidence[qid] = (question, "yes")
        sess.match_cache[(leader_id, qid)] = prob
    assert engine._should_guess(sess, None) is False
    assert engine._should_guess(sess, None) is True


def test_wise_wolf_phrases_name_the_work_and_holo():
    """Typed wise-wolf wording is the Spice and Wolf headliner, who is Holo."""
    assert engine._typed_franchise(sess_mod.new_session("wise wolf")) == "Spice and Wolf"
    assert engine._typed_franchise(sess_mod.new_session("wolf harvest deity")) == "Spice and Wolf"
    hit = web_search.headliner_from_texts(["she is the wise wolf"])
    assert hit is not None
    assert hit[0] == "Spice and Wolf"
    assert hit[1] == ("Holo",)
    assert web_search.fulltext_pin_parts(["ookami to koushinryou"]) == ["Spice and Wolf", "Holo"]
    assert "Holo" in web_search.fulltext_pin_parts(["holo the wise wolf"])


def test_series_spellings_are_one_work():
    """Ampersand and romaji titles fold onto Spice and Wolf. Other shows do not."""
    assert same_series("Spice & Wolf", "Spice and Wolf")
    assert same_series("Ookami to Koushinryou", "Spice and Wolf")
    assert same_series("Spice and Wolf: Merchant Meets the Wise Wolf", "Spice and Wolf")
    assert not same_series("Spice and Wolf", "Fullmetal Alchemist")
    assert same_series("Final Fantasy VII", "Final Fantasy")


def test_spellings_collapse_and_a_different_show_is_not_the_pin():
    """Holo the Wise Wolf is Holo. Craft Lawrence is Kraft. Another Holo is not."""
    sess = sess_mod.new_session()
    assert sess.add_candidates([
        {"id": "holo", "name": "Holo the Wise Wolf", "series": "Spice and Wolf",
         "popularity": 10, "blurb": "A traveler."},
        {"id": "horo", "name": "Holo", "series": "Spice & Wolf", "popularity": 10},
        {"id": "craft", "name": "Craft Lawrence", "series": "Ookami to Koushinryou",
         "popularity": 10},
        {"id": "kraft", "name": "Kraft Lawrence", "series": "Spice and Wolf",
         "popularity": 10},
    ]) == 2
    assert [c.name for c in sess.alive_candidates()] == ["Holo", "Kraft Lawrence"]
    assert identity_id("Holo the Wise Wolf") == "holo"
    assert same_character("Craft Lawrence", "Kraft Lawrence")
    assert not same_character("Holo", "Kraft Lawrence")
    ears = traits.QUESTIONS_BY_ID["look_animal_ears"]
    other = Candidate(id="bar", name="Holo", series="Other Show", blurb="A bartender.")
    assert engine._candidate_has_trait(ears, other) is False


def test_pin_beats_a_partner_blurb_and_a_thin_holo_page():
    """Lawrence's mention of her ears is not his trait. Holo's empty blurb still is."""
    ears = traits.QUESTIONS_BY_ID["look_animal_ears"]
    tail = traits.QUESTIONS_BY_ID["look_tail"]
    god = traits.QUESTIONS_BY_ID["species_god"]
    holo = Candidate(id="holo", name="Holo", series="Unknown", blurb="A traveler.")
    merchant = Candidate(
        id="lawrence", name="Kraft Lawrence", series="Spice and Wolf",
        tags=["animal-ears", "tail", "god", "beast"],
        blurb="He travels with Holo, the wise wolf, who has a wolf tail and ears.",
    )
    assert "animal-ears" not in web_search.mine_trait_slugs(holo.blurb)
    assert engine._candidate_has_trait(ears, holo) is True
    assert engine._candidate_has_trait(tail, holo) is True
    assert engine._candidate_has_trait(god, holo) is True
    assert engine._candidate_has_trait(ears, merchant) is False
    assert engine._candidate_has_trait(tail, merchant) is False
    assert engine._candidate_has_trait(god, merchant) is False
    assert "animal-ears" in web_search.mine_trait_slugs("She is the wise wolf.")
    assert "beast" in web_search.mine_trait_slugs("She is the wise wolf.")
    assert "god" in web_search.mine_trait_slugs("A wolf harvest deity.")
    assert "animal-ears" in web_search.mine_trait_slugs("A wolf deity of the north.")


def test_wolf_deity_answers_prefer_holo_over_the_famous_merchant():
    """God, ears and tail put Holo first even when the merchant page wears her tags."""
    names, sess = _rank(_pair(), [
        ("species_god", "yes"),
        ("look_animal_ears", "yes"),
        ("look_tail", "yes"),
        ("species_beast", "yes"),
    ])
    assert names[0] == "Holo"
    assert "Kraft Lawrence" in names
    holo = sess.by_id("holo")
    lawrence = sess.by_id("lawrence")
    assert holo is not None and lawrence is not None
    assert "animal-ears" not in holo.tags
    assert holo.logodds > lawrence.logodds


def test_a_leading_catgirl_is_not_replaced_by_the_pin():
    """The lift stops under someone already ahead of both Spice and Wolf rows."""
    rows = _pair() + [
        {"id": "mika", "name": "Misono Mika", "series": "Blue Archive",
         "medium": "game", "popularity": 200000, "tags": ["female", "animal-ears"],
         "blurb": "A student with animal ears."},
    ]
    names, sess = _rank(rows, [("look_animal_ears", "yes")])
    assert names[0] == "Misono Mika"
    assert sess.by_id("holo").logodds > sess.by_id("lawrence").logodds


def test_merchant_answers_do_not_hand_the_cast_to_holo():
    """No wolf-deity yes, and a male answer, leaves the famous merchant ahead."""
    names, _sess = _rank(_pair(), [("gender_male", "yes")])
    assert names[0] == "Kraft Lawrence"


def test_mushy_laya_does_not_erase_the_pin(monkeypatch):
    """A flat match noul is not a reason to keep the merchant over the wolf."""
    def _flat(state, questions):
        return {"match": {"noul": 0.55, "confidence": 0.2, "action": {"act_probability": 0.4}}}

    monkeypatch.setattr(laya_client, "ask", _flat)
    names, _sess = _rank(_pair(), [("species_god", "yes"), ("look_animal_ears", "yes")])
    assert names[0] == "Holo"


def test_lawrence_lead_defers_on_ears_instead_of_a_pin_thin_guess():
    """Same-work, Holo in the pool, margin wide enough to guess: ask ears once.

    The labelled miss was pin-thin because defer was null. The pin makes the
    ear question the defer, which the miss log calls near-twin-fired.
    """
    assert MISS_KEYS == (
        "target_in_shortlist",
        "top_two_posterior_margin",
        "franchise_match_entropy",
        "franchise_pair",
        "trait_window",
        "defer",
        "miss_label",
    )
    sess = sess_mod.new_session()
    sess.add_candidates(_pair())
    holo = sess.by_id("holo")
    lawrence = sess.by_id("lawrence")
    assert holo is not None and lawrence is not None
    # Posterior about 0.65/0.35: margin above 0.15, pair mass under 0.75
    # (the labelled miss shape). The stable-leader gate needs two checks
    # before it will guess, which is what makes the defer reachable.
    lawrence.logodds = 0.0
    holo.logodds = -0.62
    _arm_guess(sess, "lawrence")
    before = SessionMemory(target="Holo").record_miss(sess)
    assert before["miss_label"] == "pin-thin"
    assert before["defer"] is None
    assert before["target_in_shortlist"] is True
    assert before["trait_window"] == "ok"
    assert before["franchise_pair"] == "same-work"
    assert before["top_two_posterior_margin"] > 0.15
    assert before["franchise_match_entropy"] > 0.8
    state = engine._advance(sess)
    assert state["stage"] == "asking"
    assert state["question"]["qid"] == "look_animal_ears"
    after = SessionMemory(target="Holo", defer="look_animal_ears").record_miss(sess)
    assert after["miss_label"] == "near-twin-fired"
    assert after["defer"] == "look_animal_ears"
    assert set(after) == {
        "target_in_shortlist",
        "top_two_posterior_margin",
        "franchise_match_entropy",
        "franchise_pair",
        "trait_window",
        "defer",
        "miss_label",
    }


def test_romaji_series_still_defers_the_same_pair():
    """Spice & Wolf and the romaji title are one cast, so ears still defer."""
    sess = sess_mod.new_session()
    sess.add_candidates(_pair("Spice & Wolf", "Ookami to Koushinryou"))
    lawrence = sess.by_id("lawrence")
    holo = sess.by_id("holo")
    assert lawrence is not None and holo is not None
    lawrence.logodds = 0.0
    holo.logodds = -0.62
    _arm_guess(sess, "lawrence")
    state = engine._advance(sess)
    assert state["stage"] == "asking"
    assert state["question"]["qid"] == "look_animal_ears"


def test_a_no_on_ears_guesses_the_merchant_without_a_second_defer():
    """The pair is deferred once. Answering no then commits Lawrence."""
    sess = sess_mod.new_session()
    sess.add_candidates(_pair())
    lawrence = sess.by_id("lawrence")
    holo = sess.by_id("holo")
    assert lawrence is not None and holo is not None
    lawrence.logodds = 0.0
    holo.logodds = -0.62
    _arm_guess(sess, "lawrence")
    first = engine._advance(sess)
    assert first["question"]["qid"] == "look_animal_ears"
    sess.asked[-1]["answer"] = "no"
    engine.score_candidates(sess, traits.QUESTIONS_BY_ID["look_animal_ears"], "no")
    assert sess.posterior()[0][0].name == "Kraft Lawrence"
    second = engine._advance(sess)
    assert second["stage"] == "guessing"
    assert second["guess"]["name"] == "Kraft Lawrence"


def test_question_bank_did_not_grow_a_wolf_question():
    """Ears and tail stay the series_split questions. No wise-wolf row was added."""
    ids = {q["id"] for q in traits.QUESTION_BANK}
    assert "look_animal_ears" in ids and "look_tail" in ids
    assert "look_wise_wolf" not in ids
    assert "species_wolf" not in ids
    assert engine._SERIES_SPLIT_QIDS == (
        "look_prosthetic", "look_animal_ears", "look_eyepatch", "look_tail",
    )
