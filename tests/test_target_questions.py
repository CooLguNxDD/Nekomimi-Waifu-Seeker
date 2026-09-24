"""'Where is your character from?' (with movie/TV) and 'Which series?'."""

from __future__ import annotations

import pytest

from waifu_engine import sources, web_search
from waifu_engine.names import same_series, series_key
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.sources import wikipedia

MEDIUM = traits.QUESTIONS_BY_ID["medium"]

POOL = [
    {"id": "koharu", "name": "Koharu Shimoe", "series": "Blue Archive", "medium": "game",
     "tags": ["game", "female"]},
    {"id": "hifumi", "name": "Hifumi Ajitani", "series": "Blue Archive (video game)",
     "medium": "game", "tags": ["game", "female"]},
    {"id": "klee", "name": "Klee", "series": "Genshin Impact", "medium": "game",
     "tags": ["game", "female"]},
    {"id": "ripley", "name": "Ellen Ripley", "series": "Alien", "medium": "movie", "tags": []},
    {"id": "buffy", "name": "Buffy Summers", "series": "Buffy the Vampire Slayer",
     "medium": "tv", "tags": []},
    {"id": "makima", "name": "Makima", "series": "Chainsaw Man", "medium": "anime",
     "tags": ["anime"]},
    {"id": "mystery", "name": "Mystery Girl", "series": "Web result", "medium": "unknown",
     "tags": []},
]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    calls = []
    monkeypatch.setattr(laya_client, "ask", lambda state, q: calls.append(q) or None)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    monkeypatch.setattr(engine, "ONLINE", True)
    return calls


def _session() -> sess_mod.GuessSession:
    s = sess_mod.new_session()
    s.add_candidates(POOL)
    return s


def _alive(s):
    return {c.id for c in s.alive_candidates()}


# --- medium ---------------------------------------------------------------


def test_medium_is_one_question_with_movie_and_tv():
    assert list(MEDIUM["options"]) == ["anime", "game", "comic", "movie", "tv", "other"]
    assert MEDIUM["options"]["movie"]["label"] == "Movie"
    assert MEDIUM["options"]["tv"]["label"] == "TV series"
    for gone in ("medium_game", "medium_anime", "medium_comic"):
        assert gone not in traits.QUESTIONS_BY_ID
    assert "the character is from a live-action or animated film" in MEDIUM["criteria"].values()


def test_medium_competes_by_information_gain():
    """Mixed media keep medium in the ranking; it is not forced as the only ask.

    Series often outranks it: naming one series among several is a finer split
    than naming the medium. Both still beat traits the pool barely knows.
    """
    s = _session()
    picked = engine.candidate_questions(s)
    ids = [q["id"] for q in picked]
    assert "medium" in ids and any(i.startswith("series") for i in ids)
    assert len(picked) > 1
    top = ids[:2]
    assert "medium" in top and any(i.startswith("series") for i in top)


def test_shared_medium_lets_series_outrank_medium():
    """When every candidate is already one medium, series is the better ask."""
    s = sess_mod.new_session()
    s.add_candidates([c for c in POOL if c["medium"] == "game"])
    picked = engine.candidate_questions(s)
    ids = [q["id"] for q in picked]
    assert "medium" in ids  # still available, just not forced first
    assert ids[0].startswith("series")
    assert "medium" != ids[0]


@pytest.mark.parametrize("pick, alive", [
    ("game", {"koharu", "hifumi", "klee", "mystery"}),
    ("movie", {"ripley", "buffy", "mystery"}),        # film and TV franchises cross over
    ("tv", {"ripley", "buffy", "mystery"}),
    ("anime", {"makima", "mystery"}),
    ("other", {"mystery"}),                              # every known medium is ruled out
])
def test_a_medium_pick_removes_known_other_media_but_never_unknowns(pick, alive):
    s = _session()
    engine.score_candidates(s, MEDIUM, pick)
    assert _alive(s) == alive


def test_movie_and_tv_stay_apart_by_soft_evidence():
    s = _session()
    engine.score_candidates(s, MEDIUM, "movie")
    assert s.by_id("ripley").logodds > s.by_id("buffy").logodds


def test_known_media_never_reach_laya(_offline):
    s = _session()
    engine.score_candidates(s, MEDIUM, "game")
    asked_about = [q for q in _offline if "match" in q]
    assert len(asked_about) == 1  # only the unknown-medium candidate


def test_medium_hint_and_search_suffix_for_movie_and_tv():
    s = _session()
    s.asked.append({"qid": "medium", "answer": "tv", "kind": "choice",
                    "options": MEDIUM["options"], "text": "?"})
    assert engine._medium_hint(s) == "tv"
    assert sources._queries(["vampire slayer"], "tv") == ["vampire slayer TV series character"]
    assert sources._queries(["xenomorph"], "movie") == ["xenomorph film character"]
    s.asked[0]["answer"] = "other"
    assert engine._medium_hint(s) is None


def test_film_and_tv_pages_are_classified():
    assert wikipedia._medium_of(["Category:Female characters in film"], "") == "movie"
    assert wikipedia._medium_of(["Category:Sitcom characters"], "") == "tv"
    # An anime TV series stays anime.
    assert wikipedia._medium_of(["Category:Anime and manga characters",
                                 "Category:Television characters"], "") == "anime"
    assert web_search._guess_medium("Walter White", "", "a character in the TV series") == "tv"
    assert web_search._guess_medium("Rem", "", "Re:Zero anime TV series") == "anime"
    # IMDb hosts television; the domain alone is not a movie.
    assert web_search._guess_medium(
        "Jesse Pinkman", "https://www.imdb.com/title/tt0903747/", "Breaking Bad (TV Series)") == "tv"
    assert web_search._guess_medium("Someone", "https://www.imdb.com/name/nm1/", "a biography") == "unknown"
    # Substring hits used to tag these movie/tv and then eliminate them.
    assert web_search._guess_medium("Neighbor", "", "the neighbourhood was filmed") == "unknown"


# --- series ---------------------------------------------------------------


def test_series_keys():
    assert series_key("Blue Archive (video game)") == series_key("blue archive") == "bluearchive"
    assert series_key("The Office (TV series)") == "office"
    american = series_key("The Office (American TV series)")
    british = series_key("The Office (British TV series)")
    assert american != british and american and british
    assert not same_series("The Office (American TV series)", "The Office (British TV series)")
    assert series_key("Web result") == series_key("") == ""
    assert same_series("Re:Zero", "Re:Zero - Starting Life in Another World")
    assert not same_series("Fate", "Fate/Grand Order")  # too short to prefix-match
    assert not same_series("Web result", "Web result")


def _after_medium(pick="game"):
    s = _session()
    s.asked.append({"qid": "medium", "answer": pick, "kind": "choice",
                    "options": MEDIUM["options"], "text": "Where is your character from?",
                    "category": "medium_kind"})
    engine.score_candidates(s, MEDIUM, pick)
    return s


def test_series_options_come_from_the_leading_series():
    s = _after_medium()
    q = engine._series_question(s)
    labels = [o["label"] for o in q["options"].values()]
    assert labels[:2] == ["Blue Archive", "Genshin Impact"]  # both Blue Archives merged
    assert labels[-1] == "Another series"
    assert "Web result" not in labels
    assert q["text"] == "Which series is your character from?"
    assert q["id"] in {x["id"] for x in engine.candidate_questions(s)}


def test_series_needs_two_series():
    s = sess_mod.new_session()
    s.add_candidates([POOL[0], POOL[1]])  # one series under two spellings
    assert engine._series_question(s) is None


def test_series_labels_are_sanitised():
    s = sess_mod.new_session()
    s.add_candidates([
        {"id": "a", "name": "A", "series": "<b>Evil</b> Series (game)", "medium": "game"},
        {"id": "b", "name": "B", "series": "Genshin Impact", "medium": "game"},
    ])
    labels = [o["label"] for o in engine._series_question(s)["options"].values()]
    assert all("<" not in label and ">" not in label for label in labels)


def test_series_pick_promotes_that_series_without_model_calls(_offline):
    s = _after_medium()
    q = engine._series_question(s)
    _offline.clear()
    engine.score_candidates(s, q, "s1")  # Blue Archive
    ranked = [c.id for c, _ in s.posterior()]
    assert set(ranked[:2]) == {"koharu", "hifumi"}
    assert s.by_id("klee").alive  # soft: never eliminates
    # Only the candidate with no series needs Laya for the series question.
    series_calls = [x for x in _offline
                    if "match" in x and "Which series" in x["match"]["instructions"]]
    assert len(series_calls) == 1


def test_confirmed_series_becomes_a_specific_search_term(monkeypatch):
    s = _after_medium()
    q = engine._series_question(s)
    s.asked.append({"qid": q["id"], "text": q["text"], "category": "series", "answer": "s1",
                    "kind": "choice", "options": q["options"], "detail": None})
    assert "Blue Archive" in engine._search_terms(s)
    seen = []
    monkeypatch.setattr(engine.sources, "find_candidates",
                        lambda terms, **k: seen.append((terms, k["specific"])) or [])
    engine.refresh_candidates(s)
    assert "Blue Archive" in seen[0][0] and seen[0][1] is True
    assert engine._series_question(s) is None  # settled: not asked again


def test_another_series_asks_once_more_without_the_offered_ones():
    s = _after_medium()
    first = engine._series_question(s)
    s.asked.append({"qid": first["id"], "text": first["text"], "category": "series",
                    "answer": "other", "kind": "choice", "options": first["options"]})
    s.add_candidates([{"id": "fgo", "name": "Mash Kyrielight", "series": "Fate/Grand Order",
                       "medium": "game"},
                      {"id": "ak", "name": "Amiya", "series": "Arknights", "medium": "game"}])
    second = engine._series_question(s)
    assert second["id"] == "series_2"
    labels = [o["label"] for o in second["options"].values()]
    assert "Blue Archive" not in labels and "Genshin Impact" not in labels
    assert {"Fate/Grand Order", "Arknights"} <= set(labels)
    s.asked.append({"qid": "series_2", "answer": "other", "options": second["options"]})
    assert engine._series_question(s) is None  # at most twice


def test_page_hides_the_web_result_placeholder():
    s = _session()
    assert s.by_id("mystery").public()["series"] == ""
    assert s.by_id("koharu").public()["series"] == "Blue Archive"
