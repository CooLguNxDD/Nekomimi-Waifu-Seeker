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
    monkeypatch.setattr(web_search, "ddg_quick", lambda *a, **k:
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


# --- DuckDuckGo: capped, gated, backgrounded, one client ------------------

import time as _time

import pytest

from waifu_engine.sources import anilist


def _ddg_item(name):
    return {"title": f"{name} | Fandom", "href": f"https://x.fandom.com/wiki/{name}",
            "body": f"{name} is a character in the game."}


@pytest.fixture
def quiet_sources(monkeypatch):
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    monkeypatch.setattr(web_search, "_want_ddg_fill", lambda *a, **k: True)
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit: [])
    monkeypatch.setattr(anilist, "search_characters", lambda q, limit: [])
    with sources._BG_LOCK:
        sources._BG_READY.clear()
        sources._BG_PENDING.clear()


def test_ddg_quick_caps_requests_and_uses_one_generic_query_each(monkeypatch):
    sent = []
    monkeypatch.setattr(web_search, "_search_once",
                        lambda q, max_results=8: sent.append(q) or [])
    web_search.ddg_quick([f"q{i}" for i in range(5)], limit=12, max_requests=3, budget=60)
    assert sent == ["q0 wiki", "q1 wiki", "q2 wiki"]
    assert not any("site:" in q for q in sent)


def test_ddg_quick_stops_starting_requests_after_its_budget(monkeypatch):
    sent = []

    def slow(q, max_results=8):
        sent.append(q)
        _time.sleep(0.05)
        return []

    monkeypatch.setattr(web_search, "_search_once", slow)
    web_search.ddg_quick(["a", "b", "c"], limit=12, max_requests=3, budget=0.01)
    assert sent == ["a wiki"]


def test_ddg_client_is_reused_and_dropped_after_errors(monkeypatch):
    made = []

    class FakeDDGS:
        def __init__(self, timeout=None):
            made.append(timeout)
            self.fail = False

        def text(self, q, max_results=8):
            if q == "boom":
                raise RuntimeError("ratelimit")
            return [_ddg_item("Nami")]

    monkeypatch.setattr(web_search, "_ddgs", lambda: FakeDDGS)
    monkeypatch.setattr(web_search, "_DDG_CLIENT", None)
    web_search._search_once("a")
    web_search._search_once("b")
    assert len(made) == 1
    with pytest.raises(RuntimeError):
        web_search._search_once("boom")
    web_search._search_once("c")
    assert len(made) == 2


def test_ddg_skipped_when_laya_says_the_pool_fits(monkeypatch, quiet_sources):
    called = []
    monkeypatch.setattr(web_search, "ddg_quick", lambda *a, **k: called.append(1) or [])
    sources.find_candidates(["plumber"], medium_hint="game", limit=5,
                            exclude_names={"Mario"}, background_key="s1",
                            ddg_gate=lambda: False)
    assert called == []


def test_ddg_runs_in_background_when_the_pool_has_candidates(monkeypatch, quiet_sources):
    gate = __import__("threading").Event()

    def slow(queries, limit, **k):
        gate.wait(5)
        return [{"id": "ddg_nami", "name": "Nami"}]

    monkeypatch.setattr(web_search, "ddg_quick", slow)
    t0 = _time.perf_counter()
    first = sources.find_candidates(["navigator"], limit=5, exclude_names={"Luffy"},
                                    background_key="s2", ddg_gate=lambda: True)
    assert _time.perf_counter() - t0 < 1.0
    assert first == [] and sources.background_pending("s2")
    # A second search while the first fill runs does not queue another one.
    assert sources._bg_start("s2", ["x"], 5) is False
    gate.set()
    for _ in range(100):
        if not sources.background_pending("s2"):
            break
        _time.sleep(0.02)
    second = sources.find_candidates(["navigator"], limit=5, exclude_names={"Luffy"},
                                     background_key="s2", ddg_gate=lambda: False)
    assert [c["name"] for c in second] == ["Nami"]
    assert sources.take_background("s2") == []  # consumed once


def test_ddg_runs_inline_when_nothing_else_was_found(monkeypatch, quiet_sources):
    monkeypatch.setattr(web_search, "ddg_quick",
                        lambda *a, **k: [{"id": "ddg_mario", "name": "Mario"}])
    out = sources.find_candidates(["plumber"], medium_hint="game", limit=5,
                                  background_key="s3", ddg_gate=lambda: True)
    assert [c["name"] for c in out] == ["Mario"]
    assert not sources.background_pending("s3")
