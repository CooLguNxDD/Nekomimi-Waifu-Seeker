"""Offline regression tests for search-driven candidate discovery."""

from waifu_engine import sources, web_search
from waifu_engine.sources import wikipedia


def test_queries_keep_anchor_and_retry_shorter_clue_combinations():
    queries = sources._queries(
        ["Nintendo", "The character is male", "Not true: Is your character female?",
         "red cap", "Italian plumber"], "game")
    assert "Nintendo" in queries[0]
    assert "Italian plumber" in queries[0]
    assert all("female" not in q and "Not true" not in q for q in queries)
    assert queries[-1] == "Italian plumber video game character"
    assert len(queries) == 3


def test_retrieval_preserves_relevance_and_retries_empty_query(monkeypatch):
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    queries = []

    def wiki(query, limit):
        queries.append(query)
        if len(queries) == 1:
            return []
        return [{"id": "match", "name": "Relevant character", "popularity": 0},
                {"id": "famous", "name": "Famous character", "popularity": 100000}]

    monkeypatch.setattr(wikipedia, "search_characters", wiki)
    result = sources.find_candidates(["Nintendo", "male", "plumber"],
                                     medium_hint="game", limit=2, use_ddg=False)
    assert len(queries) == 2
    assert [c["id"] for c in result] == ["match", "famous"]


def test_wikipedia_failure_does_not_prevent_ddg_discovery(monkeypatch):
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    monkeypatch.setattr(web_search, "_want_ddg_fill", lambda *a, **k: True)

    def broken(*a, **k):
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(wikipedia, "search_characters", broken)
    monkeypatch.setattr(web_search, "search_by_constraints", lambda *a, **k:
                        [{"id": "m", "name": "Mario"}])
    assert sources.find_candidates(["plumber"], medium_hint="game", limit=1)[0]["name"] == "Mario"


def test_wikipedia_search_requests_all_extracts_and_preserves_rank(monkeypatch):
    def query(params):
        assert params["exlimit"] == "max"
        return [{"index": 2, "name": "Second"}, {"index": 1, "name": "First"}]

    monkeypatch.setattr(wikipedia, "_query", query)
    monkeypatch.setattr(wikipedia, "_to_candidate", lambda p: p)
    assert [p["name"] for p in wikipedia.search_characters("plumber")] == ["First", "Second"]


def test_character_indexes_reject_game_and_anime_pages():
    for url in ["https://vndb.org/v382", "https://myanimelist.net/anime/1/Cowboy_Bebop",
                "https://anilist.co/anime/1"]:
        assert web_search._to_candidate({"title": "CHAOS;HEAD", "href": url,
                                        "body": "A protagonist is a playable character."}, 1) is None
    assert web_search._to_candidate({"title": "Makise Kurisu", "href": "https://vndb.org/c123",
                                    "body": "A fictional character."}, 1) is not None
