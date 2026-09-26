"""Empty-seed search asks for Edward Elric by name.

Button answers never contain his name, and AniList only matches names. The
popular page then filled the shortlist with Naruto and the miss was labelled
search. Coverage name searches run on that path. They are not a series pin.
"""

from waifu_engine import sources, web_search
from waifu_engine.names import same_character
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.nekomimi.lexicon.load import LexiconError, build_coverage, load_document
from waifu_engine.sources import anilist, wikipedia

import pytest


_ED_QUERIES = [
    "Edward Elric",
    "Ed Elric",
    "Fullmetal Alchemist",
    "Full Metal Alchemist",
    "Hagane no Renkinjutsushi",
    "automail",
]


def _yes(qid: str) -> dict:
    """A yes answer stored the way ``submit_answer`` leaves it."""
    question = traits.QUESTIONS_BY_ID[qid]
    return {
        "qid": question["id"], "text": question["text"], "category": question["category"],
        "prior": question["prior"], "kind": question.get("kind", "yesno"),
        "answer": "yes", "detail": None, "options": question.get("options"),
    }


def _choice(qid: str, answer: str) -> dict:
    """A choice answer with the option the player picked."""
    question = traits.QUESTIONS_BY_ID[qid]
    return {
        "qid": question["id"], "text": question["text"], "category": question["category"],
        "prior": question["prior"], "kind": "choice", "answer": answer,
        "detail": None, "options": question["options"],
    }


def test_prosthetic_and_blonde_asks_for_edward_and_the_fma_aliases():
    """The empty-seed pair the bank can confirm expands to Ed's name searches."""
    facts = ["male", "blonde hair", "have a prosthetic or mechanical limb"]
    assert web_search.coverage_queries(facts) == _ED_QUERIES
    assert web_search.coverage_queries(["have a prosthetic or mechanical limb"]) == []
    assert web_search.coverage_queries(["blonde hair"]) == []
    assert web_search.coverage_queries(
        ["Not true: have a prosthetic or mechanical limb", "blonde hair"]
    ) == []


def test_thin_seed_aliases_and_a_blond_braid_use_the_same_name_searches():
    """A typed alias or a blond braid still searches Edward when no series chip exists."""
    for fact in (
        "automail",
        "state alchemist",
        "hagane no renkinjutsushi",
        "full metal alchemist",
        "ed elric",
        "blond braid",
        "blonde braided hair",
    ):
        assert web_search.coverage_queries([fact]) == _ED_QUERIES
    assert web_search.franchise_label("hagane no renkinjutsushi") == "Fullmetal Alchemist"
    assert web_search.franchise_label("full metal alchemist") == "Fullmetal Alchemist"
    assert web_search.title_character_queries(["hagane no renkinjutsushi"]) == ["Edward Elric"]
    assert web_search.is_non_character("Fullmetal Alchemist")
    assert web_search.is_non_character("Hagane no Renkinjutsushi")


def test_empty_seed_answers_put_edward_in_the_pool_ahead_of_naruto(monkeypatch):
    """Popular fill still runs, and Edward is in the session the player can guess from."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    wiki: list[str] = []
    ani: list[str] = []

    def wiki_search(query, limit=10):
        wiki.append(query)
        return []

    def ani_search(query, limit=10):
        ani.append(query)
        if query == "Edward Elric":
            return [{
                "id": "ed", "name": "Edward Elric", "series": "Fullmetal Alchemist",
                "medium": "anime", "popularity": 40,
                "blurb": "The Fullmetal Alchemist, a boy with an automail arm.",
                "tags": ["prosthetic", "blonde"],
            }]
        return []

    monkeypatch.setattr(wikipedia, "search_characters", wiki_search)
    monkeypatch.setattr(anilist, "search_characters", ani_search)
    monkeypatch.setattr(sources, "popular_characters", lambda medium, n: [{
        "id": "naruto", "name": "Naruto Uzumaki", "series": "Naruto",
        "medium": "anime", "popularity": 99999, "blurb": "", "tags": ["blonde"],
    }])

    sess = sess_mod.new_session("")
    sess.asked = [
        _choice("medium", "anime"),
        _choice("hair_color", "blonde"),
        _yes("look_prosthetic"),
    ]
    engine.refresh_candidates(sess)
    assert ani == _ED_QUERIES
    assert wiki == _ED_QUERIES
    names = [cand.name for cand in sess.candidates]
    assert any(same_character("Edward Elric", name) for name in names)
    assert any(same_character("Naruto Uzumaki", name) for name in names)


def test_rewrite_cannot_drop_the_edward_name_search(monkeypatch):
    """A generic rewrite occupies the template window. The name search still runs."""
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    ani: list[str] = []

    def ani_search(query, limit=10):
        ani.append(query)
        if query == "Edward Elric":
            return [{"id": "ed", "name": "Edward Elric", "popularity": 40}]
        return [{"id": query, "name": query, "popularity": 1}]

    monkeypatch.setattr(anilist, "search_characters", ani_search)
    monkeypatch.setattr(wikipedia, "search_characters", lambda query, limit=10: [])
    rewritten = [
        "blonde boy anime character",
        "prosthetic anime character",
        "young alchemist",
        "short blonde anime",
        "military boy anime",
    ]
    sources.find_candidates(
        ["automail", "blonde braid"],
        medium_hint="anime",
        limit=8,
        use_ddg=False,
        specific=True,
        rewritten=rewritten,
        focus=["blonde braid"],
    )
    assert ani[:len(_ED_QUERIES)] == _ED_QUERIES
    assert "blonde boy anime character" in " ".join(ani)


def test_coverage_loader_rejects_a_duplicate_cluster():
    """A repeated coverage id fails the load instead of merging two query lists."""
    doc = load_document("coverage.yml")
    broken = {
        "clusters": [dict(doc["clusters"][0]), dict(doc["clusters"][0])],
    }
    with pytest.raises(LexiconError, match="duplicate id"):
        build_coverage(broken)
