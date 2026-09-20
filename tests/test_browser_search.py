"""Offline tests for Playwright search parsers and the DDG fill waterfall.

No Chromium, no network, no weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from waifu_engine import browser_search, web_search
from waifu_engine.sources import find_candidates

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def _no_chromium(*_a, **_k):
        raise AssertionError("tests must not launch Chromium")

    def _no_ddg(*_a, **_k):
        raise AssertionError("tests must not hit DuckDuckGo")

    monkeypatch.setattr(browser_search, "_launch_browser", _no_chromium)
    monkeypatch.setattr(browser_search, "_fetch_html", lambda *_a, **_k: None)
    monkeypatch.setattr(web_search, "_search_once", _no_ddg)
    monkeypatch.setenv("WAIFU_PLAYWRIGHT_ENRICH", "0")
    browser_search.reset()
    yield
    browser_search.reset()


def _html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _pw_hit(name: str, n: int, series: str = "One Piece") -> dict:
    slug = name.replace(" ", "_")
    return {
        "title": f"{name} - MyAnimeList.net",
        "href": f"https://myanimelist.net/character/{n}/{slug}",
        "body": f"{name} is a character from {series}. A pirate and fighter.",
    }


# --- parsers ---------------------------------------------------------


def test_mal_search_and_character_parse_to_name_series_blurb_tags():
    hits = browser_search.parse_mal_search(_html("mal_search.html"))
    assert hits
    cand = web_search._to_candidate(hits[0], 1, source="playwright")
    assert cand is not None
    assert cand["name"] == "Sanji"
    assert cand["series"] == "One Piece"
    assert "cook" in cand["blurb"].lower() or "pirate" in cand["blurb"].lower()
    assert cand["source"] == "playwright"
    assert "playwright" in cand["tags"]
    assert "male" in cand["tags"]

    page = browser_search.parse_mal_character(_html("mal_character.html"))
    assert page["name"] == "Sanji"
    assert page["series"] == "One Piece"
    assert "male" in web_search.mine_trait_slugs(page["blurb"])
    assert page["image_url"].startswith("https://cdn.myanimelist.net/")


def test_fandom_search_and_page_parse_to_name_series_blurb_tags():
    hits = browser_search.parse_fandom_search(_html("fandom_search.html"))
    assert hits
    cand = web_search._to_candidate(hits[0], 1, source="playwright")
    assert cand is not None
    assert cand["name"] == "Hatsune Miku"
    assert cand["series"] == "Vocaloid"
    assert cand["source"] == "playwright"
    assert "female" in cand["tags"]
    assert "idol" in cand["tags"]

    page = browser_search.parse_fandom_page(_html("fandom_page.html"))
    assert page["name"] == "Hatsune Miku"
    assert page["series"] == "Vocaloid"
    slugs = web_search.mine_trait_slugs(page["blurb"])
    assert "female" in slugs
    assert "idol" in slugs


def test_vndb_search_and_character_parse_to_name_series_blurb_tags():
    hits = browser_search.parse_vndb_search(_html("vndb_search.html"))
    assert hits
    cand = web_search._to_candidate(hits[0], 1, source="playwright")
    assert cand is not None
    assert cand["name"] == "Ayanami Rei"
    assert cand["series"] == "Neon Genesis Evangelion"
    assert cand["source"] == "playwright"
    assert "female" in cand["tags"]
    assert "protagonist" in cand["tags"]

    page = browser_search.parse_vndb_character(_html("vndb_character.html"))
    assert page["name"] == "Ayanami Rei"
    assert page["series"] == "Neon Genesis Evangelion"
    slugs = web_search.mine_trait_slugs(page["blurb"])
    assert "female" in slugs
    assert "protagonist" in slugs


def test_parse_character_page_dispatches_by_url():
    info = browser_search.parse_character_page(
        "https://myanimelist.net/character/1887/Sanji",
        _html("mal_character.html"),
    )
    assert info and info["name"] == "Sanji"


# --- available() -----------------------------------------------------


def test_available_false_when_playwright_import_fails(monkeypatch):
    def boom():
        raise ImportError("No module named 'playwright'")

    monkeypatch.setattr(browser_search, "_import_sync_playwright", boom)
    browser_search.reset()
    assert browser_search.available() is False


def test_available_false_when_launch_fails(monkeypatch):
    class _Fake:
        def start(self):
            return self

    monkeypatch.setattr(browser_search, "_import_sync_playwright", lambda: (lambda: _Fake()))
    monkeypatch.setattr(
        browser_search,
        "_launch_browser",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no chromium")),
    )
    browser_search.reset()
    assert browser_search.available() is False


# --- waterfall -------------------------------------------------------


def test_playwright_partial_fills_remainder_with_ddg(monkeypatch):
    monkeypatch.setenv("WAIFU_SEARCH_BACKEND", "auto")
    monkeypatch.setattr(browser_search, "available", lambda: True)
    monkeypatch.setattr(
        browser_search,
        "search",
        lambda *_a, **_k: [_pw_hit("Sanji", 1887), _pw_hit("Zoro", 1372)],
    )
    ddg_called: list[tuple] = []

    def fake_ddg(query, max_results=8):
        ddg_called.append((query, max_results))
        return [
            _pw_hit("Nami", 64),
            _pw_hit("Usopp", 40),
            _pw_hit("Robin", 62),
        ]

    monkeypatch.setattr(web_search, "_search_once", fake_ddg)

    out = web_search.search_by_constraints(["pirate cook"], limit=5)
    assert len(out) == 5
    assert ddg_called
    sources = [c["source"] for c in out]
    assert sources.count("playwright") == 2
    assert sources.count("duckduckgo") == 3
    meta = web_search.last_search_meta()
    assert meta["search_state"] == "done"
    assert meta["search_state"] in web_search.SEARCH_STATES
    assert meta["backend"] == "auto"


def test_playwright_raises_uses_ddg_only(monkeypatch):
    monkeypatch.setenv("WAIFU_SEARCH_BACKEND", "auto")
    monkeypatch.setattr(browser_search, "available", lambda: True)

    def boom(*_a, **_k):
        raise RuntimeError("chromium died")

    monkeypatch.setattr(browser_search, "search", boom)
    ddg_called: list = []

    def fake_ddg(query, max_results=8):
        ddg_called.append(query)
        return [_pw_hit("Nami", 64), _pw_hit("Usopp", 40)]

    monkeypatch.setattr(web_search, "_search_once", fake_ddg)

    out = web_search.search_by_constraints(["navigator"], limit=5)
    assert ddg_called
    assert out
    assert all(c["source"] == "duckduckgo" for c in out)
    meta = web_search.last_search_meta()
    assert meta["search_state"] == "done"
    assert any("playwright" in e.lower() for e in meta["errors"])


def test_backend_ddg_never_calls_playwright(monkeypatch):
    monkeypatch.setenv("WAIFU_SEARCH_BACKEND", "ddg")
    called: list[str] = []

    def pw_search(*_a, **_k):
        called.append("pw")
        return [_pw_hit("Sanji", 1887)]

    monkeypatch.setattr(browser_search, "available", lambda: True)
    monkeypatch.setattr(browser_search, "search", pw_search)
    monkeypatch.setattr(web_search, "_search_once", lambda *_a, **_k: [_pw_hit("Nami", 64)])

    out = web_search.search_by_constraints(["navigator"], limit=3)
    assert "pw" not in called
    assert out
    assert all(c["source"] == "duckduckgo" for c in out)
    meta = web_search.last_search_meta()
    assert meta["backend"] == "ddg"
    assert meta["search_state"] == "done"
    assert meta["search_state"] in web_search.SEARCH_STATES


def test_search_state_follows_enum(monkeypatch):
    monkeypatch.setenv("WAIFU_SEARCH_BACKEND", "ddg")
    monkeypatch.setattr(web_search, "_search_once", lambda *_a, **_k: [_pw_hit("Nami", 64)])
    web_search.search_by_constraints(["navigator"], limit=1)
    meta = web_search.last_search_meta()
    assert meta["search_state"] in web_search.SEARCH_STATES
    assert set(web_search.SEARCH_STATES) == {
        "idle",
        "playwright_search",
        "fill_ddg",
        "enrich",
        "done",
    }


def test_multiround_logs_include_search_state(monkeypatch):
    monkeypatch.setenv("WAIFU_SEARCH_BACKEND", "auto")
    monkeypatch.setattr(browser_search, "available", lambda: True)
    monkeypatch.setattr(browser_search, "search", lambda *_a, **_k: [_pw_hit("Sanji", 1887)])
    monkeypatch.setattr(web_search, "_search_once", lambda *_a, **_k: [_pw_hit("Nami", 64)])
    _cands, logs = web_search.search_characters_multiround("pirate cook", rounds=1, per_round=4)
    assert logs
    assert logs[-1]["search_state"] == "done"
    assert logs[-1]["search_state"] in web_search.SEARCH_STATES
    assert logs[-1]["backend"] == "auto"


def test_find_candidates_playwright_then_ddg_fill(monkeypatch):
    monkeypatch.setenv("WAIFU_SEARCH_BACKEND", "auto")
    monkeypatch.setattr(browser_search, "available", lambda: True)
    monkeypatch.setattr(
        browser_search,
        "search",
        lambda *_a, **_k: [_pw_hit("Sanji", 1887), _pw_hit("Zoro", 1372)],
    )
    monkeypatch.setattr("waifu_engine.sources.wikipedia.search_characters", lambda *_a, **_k: [])
    monkeypatch.setattr("waifu_engine.sources.anilist.search_characters", lambda *_a, **_k: [])
    ddg_called: list = []

    def fake_ddg(query, max_results=8):
        ddg_called.append(query)
        return [_pw_hit("Nami", 64), _pw_hit("Usopp", 40), _pw_hit("Robin", 62)]

    monkeypatch.setattr(web_search, "_search_once", fake_ddg)

    out = find_candidates(["pirate cook"], limit=5)
    assert len(out) == 5
    assert ddg_called
    sources = {c["source"] for c in out}
    assert "playwright" in sources
    assert "duckduckgo" in sources
    assert web_search.last_search_meta()["search_state"] == "done"


def test_find_candidates_backend_ddg_skips_playwright(monkeypatch):
    monkeypatch.setenv("WAIFU_SEARCH_BACKEND", "ddg")
    called: list[str] = []
    monkeypatch.setattr(browser_search, "available", lambda: True)
    monkeypatch.setattr(
        browser_search,
        "search",
        lambda *_a, **_k: called.append("pw") or [_pw_hit("Sanji", 1887)],
    )
    monkeypatch.setattr("waifu_engine.sources.wikipedia.search_characters", lambda *_a, **_k: [])
    monkeypatch.setattr("waifu_engine.sources.anilist.search_characters", lambda *_a, **_k: [])
    monkeypatch.setattr(web_search, "_search_once", lambda *_a, **_k: [_pw_hit("Nami", 64)])

    out = find_candidates(["navigator"], limit=3)
    assert "pw" not in called
    assert out
    assert all(c.get("source") == "duckduckgo" for c in out)
