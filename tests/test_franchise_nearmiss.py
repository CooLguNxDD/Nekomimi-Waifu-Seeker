"""Same-franchise near-misses and short-name collisions from the diverse bench.

Empty-seed runs guessed Kylo for Vader, Gil for Homer, and Mario Rossi for
Mario. These tests lock the priors that separate a franchise lead from a
side character, and an exact short name from a longer unrelated namesake.
"""

from __future__ import annotations

import pytest

from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.sources import wikipedia
from waifu_engine.web_search import _guess_series


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Heuristic scoring only: these priors must hold with no model."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)


def _rank(rows, asked=None, detail=""):
    """Posterior names after replaying ``asked`` over ``rows``."""
    s = sess_mod.new_session()
    s.add_candidates(rows)
    if detail:
        s.asked.append({
            "qid": "series", "text": "Which series is your character from?",
            "answer": "other", "kind": "choice", "detail": detail,
            "options": {"other": {"label": "Another series"}},
        })
    for qid, answer in asked or []:
        engine.score_candidates(s, traits.QUESTIONS_BY_ID[qid], answer)
    if not asked:
        engine._rescore_candidates(s)
    return [c.name for c, _ in s.posterior()], s


def test_typed_franchise_leads_the_search_terms():
    s = sess_mod.new_session()
    s.asked.append({
        "qid": "series", "answer": "other", "kind": "choice", "detail": "cowboy bebop",
        "options": {"other": {"label": "Another series"}}, "text": "Which series?",
    })
    assert engine._typed_franchise(s) == "Cowboy Bebop"
    assert engine._search_terms(s)[0] == "Cowboy Bebop"
    assert engine._series_trait_anchor(s).startswith("Cowboy Bebop")


def test_spike_leads_ed_even_when_his_series_field_is_missing():
    rows = [
        {"id": "ed", "name": "Ed", "series": "Cowboy Bebop", "medium": "anime",
         "popularity": 20000, "tags": ["male"],
         "blurb": "Edward is a hacker on the Bebop crew in Cowboy Bebop."},
        {"id": "spike", "name": "Spike Spiegel", "series": "Unknown", "medium": "anime",
         "popularity": 60000, "tags": ["protagonist", "male"],
         "blurb": "Spike Spiegel is a bounty hunter in Cowboy Bebop."},
    ]
    names, _s = _rank(rows, [("role_protagonist", "yes")], detail="cowboy bebop")
    assert names[0] == "Spike Spiegel"


def test_homer_outranks_a_simpsons_side_character_once_he_is_the_lead():
    rows = [
        {"id": "homer", "name": "Homer Simpson", "series": "The Simpsons",
         "medium": "tv", "popularity": 80000, "tags": ["protagonist", "male"],
         "blurb": "The main character of the animated sitcom The Simpsons."},
        {"id": "gil", "name": "Gil Gunderson", "series": "The Simpsons",
         "medium": "tv", "popularity": 400, "tags": ["male"],
         "blurb": "A salesman who appears in The Simpsons."},
    ]
    names, _s = _rank(rows, [("role_protagonist", "yes")], detail="simpsons")
    assert names[0] == "Homer Simpson"


def test_vader_outranks_kylo_by_fame_inside_star_wars():
    rows = [
        {"id": "vader", "name": "Darth Vader", "series": "Star Wars",
         "medium": "movie", "popularity": 200000, "tags": ["antagonist", "male"],
         "blurb": "A Star Wars antagonist."},
        {"id": "kylo", "name": "Kylo Ren", "series": "Star Wars",
         "medium": "movie", "popularity": 20000, "tags": ["antagonist", "male"],
         "blurb": "A Star Wars antagonist."},
        {"id": "tank", "name": "Bacta Tank Darth Vader / Unmasked Anakin Skywalker",
         "series": "Star Wars", "medium": "movie", "popularity": 50,
         "tags": ["male"], "blurb": "A merchandise page."},
    ]
    names, _s = _rank(rows, [("role_antagonist", "yes")], detail="star wars")
    assert names[0] == "Darth Vader"
    assert names[-1].startswith("Bacta Tank")


def test_mario_outranks_mario_rossi_when_the_clue_is_the_short_name():
    rows = [
        {"id": "mario", "name": "Mario", "series": "Super Mario", "medium": "game",
         "popularity": 90000, "tags": ["protagonist", "male", "game"],
         "blurb": "The hero of Super Mario."},
        {"id": "rossi", "name": "Mario Rossi", "series": "Captain Tsubasa",
         "medium": "unknown", "popularity": 5000, "tags": ["male"],
         "blurb": "A footballer named Mario Rossi."},
    ]
    names, s = _rank(rows, [], detail="mario")
    assert names[0] == "Mario"
    assert s.by_id("rossi").logodds < s.by_id("mario").logodds


def test_publisher_bucket_is_not_a_series_option():
    s = sess_mod.new_session()
    s.add_candidates([
        {"id": "abom", "name": "Abomination", "series": "Marvel Comics",
         "medium": "comic", "popularity": 100},
        {"id": "agent", "name": "Agent X", "series": "Marvel Comics",
         "medium": "comic", "popularity": 80},
        {"id": "spidey", "name": "Spider-Man", "series": "Spider-Man",
         "medium": "comic", "popularity": 50000, "tags": ["protagonist"]},
        {"id": "batman", "name": "Batman", "series": "Batman",
         "medium": "comic", "popularity": 40000},
    ])
    labels = [o["label"] for o in engine._series_question(s)["options"].values()]
    assert "Marvel Comics" not in labels
    assert "Spider-Man" in labels


def test_wikipedia_series_skips_the_publisher_bucket():
    series = wikipedia._series_of(
        "Spider-Man",
        ["Category:Marvel Comics superheroes", "Category:Spider-Man characters"],
        "Spider-Man is a superhero.",
    )
    assert series == "Spider-Man"
    assert wikipedia._series_of(
        "Abomination (character)",
        ["Category:Marvel Comics supervillains"],
        "Abomination is a Marvel Comics supervillain.",
    ) == "Unknown"


def test_guess_series_keeps_spider_man_off_the_publisher_label():
    assert _guess_series("Peter Parker is Spider-Man") == "Spider-Man"
    assert _guess_series("Spike Spiegel from Cowboy Bebop") == "Cowboy Bebop"
    assert _guess_series("a Star Wars villain") == "Star Wars"


def test_mario_alias_needs_the_whole_clue():
    s = sess_mod.new_session("Mario Rossi")
    assert engine._typed_franchise(s) == ""
    s = sess_mod.new_session("mario")
    assert engine._typed_franchise(s) == "Super Mario"


def test_a_no_on_protagonist_does_not_boost_the_series_lead():
    rows = [
        {"id": "homer", "name": "Homer Simpson", "series": "The Simpsons",
         "medium": "tv", "popularity": 1000, "tags": ["protagonist", "male"],
         "blurb": "The main character of the animated sitcom The Simpsons."},
        {"id": "gil", "name": "Gil Gunderson", "series": "The Simpsons",
         "medium": "tv", "popularity": 1000, "tags": ["male"],
         "blurb": "A salesman who appears in The Simpsons."},
    ]
    names, _s = _rank(rows, [("role_protagonist", "no")], detail="simpsons")
    assert names[0] == "Gil Gunderson"
