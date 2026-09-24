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


# --- cold start: broad facts use the popular pool; errors are visible -------

from waifu_engine import timing
from waifu_engine.sources import _http


def _named(*names, medium="anime"):
    return [{"id": n.lower(), "name": n, "medium": medium, "popularity": i}
            for i, n in enumerate(names)]


def test_broad_facts_skip_name_search_and_use_the_popular_pool(monkeypatch, quiet_sources):
    called = []
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit: called.append("wiki") or [])
    monkeypatch.setattr(anilist, "search_characters", lambda q, limit: called.append("anilist") or [])
    monkeypatch.setattr(web_search, "ddg_quick", lambda *a, **k: called.append("ddg") or [])
    monkeypatch.setattr(sources, "popular_characters",
                        lambda medium, n: _named("Miku", "Rem", "Saber")[:n])
    out = sources.find_candidates(["female video game character"], limit=12,
                                  specific=False, ddg_gate=lambda: True)
    assert [c["name"] for c in out] == ["Miku", "Rem", "Saber"]
    assert called == []  # no name search, and no inline DuckDuckGo for broad traits


def test_anilist_runs_for_a_game_without_relabeling_anime(monkeypatch, quiet_sources):
    called = []

    def search(query, limit):
        called.append(query)
        return [{"id": "al_mika", "name": "Mika Misono", "series": "Blue Archive",
                 "medium": "anime", "blurb": "Long pink hair and angel wings.",
                 "tags": ["anilist", "anime"]}]

    monkeypatch.setattr(anilist, "search_characters", search)
    monkeypatch.setattr(web_search, "_want_ddg_fill", lambda *a, **k: False)
    out = sources.find_candidates(["pink hair halo wings"], medium_hint="game",
                                  limit=5, use_ddg=False)
    assert called
    mika = next(c for c in out if c["name"] == "Mika Misono")
    assert mika["medium"] == "anime"
    assert "game" not in mika["tags"]


def test_popular_pool_stops_growing_at_its_cap(monkeypatch, quiet_sources):
    monkeypatch.setenv("WAIFU_POPULAR_POOL", "2")
    asked = []
    monkeypatch.setattr(sources, "popular_characters",
                        lambda medium, n: asked.append(n) or _named("A", "B", "C")[:n])
    assert len(sources.find_candidates(["x"], specific=False)) == 2
    assert sources.find_candidates(["x"], specific=False, exclude_names={"A", "B"}) == []
    assert asked == [2]


def test_popular_characters_follow_the_medium(monkeypatch):
    monkeypatch.setattr(anilist, "top_characters",
                        lambda page=1, per_page=50: _named("Lelouch", "Levi", "Gojo", "Rem")[:per_page])
    members = {"Category:Female characters in video games": ["Lara Croft", "Samus Aran"],
               "Category:Male characters in video games": ["Mario", "Link"],
               "Category:Marvel Comics superheroes": ["Spider-Man"],
               "Category:DC Comics superheroes": ["Batman"],
               "Category:Female characters in film": ["Ellen Ripley"],
               "Category:Male characters in film": ["Indiana Jones"],
               "Category:Female characters in television": ["Buffy Summers"],
               "Category:Male characters in television": ["Walter White"]}
    monkeypatch.setattr(wikipedia, "category_members", lambda cat, limit=100: members[cat])
    media = {"Spider-Man": "comic", "Batman": "comic", "Ellen Ripley": "movie",
             "Indiana Jones": "movie", "Buffy Summers": "tv", "Walter White": "tv"}
    views = {"Samus Aran": 900, "Mario": 500, "Link": 400, "Lara Croft": 300}
    monkeypatch.setattr(wikipedia, "pages_by_title", lambda titles: [
        {"id": t, "name": t, "medium": media.get(t, "game"), "popularity": views.get(t, 1)}
        for t in titles])
    assert [c["name"] for c in sources.popular_characters("anime", 2)] == ["Lelouch", "Levi"]
    games = sources.popular_characters("game", 4)
    assert {c["name"] for c in games} == {"Lara Croft", "Samus Aran", "Mario", "Link"}
    assert [c["name"] for c in games][0] == "Samus Aran"  # most viewed first
    mixed = sources.popular_characters(None, 8)
    assert {c["medium"] for c in mixed} == {"anime", "game", "comic", "movie", "tv"}
    assert len(mixed) <= 8
    assert {c["name"] for c in sources.popular_characters("tv", 2)} == {"Buffy Summers", "Walter White"}


def test_network_errors_are_recorded_not_mistaken_for_no_results(monkeypatch, quiet_sources):
    import urllib.error

    def refuse(req, timeout=None, context=None):
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(_http.urllib.request, "urlopen", refuse)
    _http.clear_cache()
    monkeypatch.setattr(wikipedia, "search_characters",
                        lambda q, limit: wikipedia._query({"list": "search", "srsearch": q}))

    @timing.traced("t")
    def run():
        sources.find_candidates(["Vocaloid"], limit=5)
        return {}

    out = run()
    assert any("403" in e for e in web_search.last_search_meta()["errors"])
    assert "403" in out["timing"]["err"]
    assert out["timing"]["hits"].startswith("wikipedia:0")


def test_popular_pool_refills_by_candidates_still_in_play(monkeypatch, quiet_sources):
    monkeypatch.setenv("WAIFU_POPULAR_POOL", "3")
    monkeypatch.setattr(sources, "popular_characters",
                        lambda medium, n: _named("A", "B", "C", "D", "E", "F")[:n])
    # A-C were added earlier; two were ruled out, one is still in play.
    out = sources.find_candidates(["x"], specific=False,
                                  exclude_names={"A", "B", "C"}, pool_size=1)
    assert [c["name"] for c in out] == ["D", "E"]


def test_background_ddg_logs_why_it_found_nothing(monkeypatch, quiet_sources):
    import logging

    lines = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = Grab()
    timing.log.addHandler(handler)
    try:
        def ratelimited(q, max_results=8):
            raise RuntimeError("202 Ratelimit")

        monkeypatch.setattr(web_search, "_search_once", ratelimited)
        sources._bg_run("k1", ["x"], 5)
    finally:
        timing.log.removeHandler(handler)
    assert "found=0" in lines[-1] and "Ratelimit" in lines[-1]


def test_background_queue_is_bounded_across_sessions(monkeypatch, quiet_sources):
    import threading

    monkeypatch.setenv("WAIFU_DDG_BG_MAX_PENDING", "1")
    gate = threading.Event()
    monkeypatch.setattr(web_search, "ddg_quick", lambda *a, **k: gate.wait(5) and [])
    try:
        assert sources._bg_start("sess-a", ["q"], 5) is True
        assert sources._bg_start("sess-b", ["q"], 5) is False  # dropped, not queued
    finally:
        gate.set()
    for _ in range(100):
        if not sources.background_pending("sess-a"):
            break
        _time.sleep(0.02)
    assert sources._bg_start("sess-b", ["q"], 5) is True  # room again


def test_colour_option_tags_are_mined_from_snippets():
    from waifu_engine.nekomimi import traits

    slugs = set(web_search.mine_trait_slugs(
        "She has green hair and amber eyes; her rival is brown-eyed with purple hair."))
    assert {"green", "purple", "eyes-gold", "eyes-brown"} <= slugs
    assert "blue" not in web_search.mine_trait_slugs("She has blue eyes.")  # not hair
    mined = {slug for slug, _ in web_search.TRAIT_PATTERNS}
    for qid in ("hair_color", "eye_color"):
        for option in traits.QUESTIONS_BY_ID[qid]["options"].values():
            assert set(option["tags"]) <= mined, (qid, option["tags"])
