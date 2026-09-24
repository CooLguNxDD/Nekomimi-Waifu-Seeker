"""Gemini search grounding as a candidate source. Offline: the client is fake."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from waifu_engine import sources, web_search
from waifu_engine.sources import anilist, gemini, wikipedia

REPLY = """Here you go:
```json
[{"name": "Hatsune Miku", "series": "Vocaloid", "medium": "game",
  "description": "A virtual singer with long teal twintails."},
 {"name": "Rem", "series": "Re:Zero", "medium": "anime",
  "description": "A maid with blue hair and blue eyes."},
 {"series": "no name, dropped"}, "not an object"]
```"""


class FakeModels:
    def __init__(self, text=REPLY, fail=None, delay=0.0):
        self.text, self.fail, self.delay = text, fail, delay
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        web = SimpleNamespace(title="Hatsune Miku - Wikipedia", uri="https://example.org/miku")
        meta = SimpleNamespace(grounding_chunks=[SimpleNamespace(web=web)],
                               web_search_queries=["teal twintails virtual singer character"])
        return SimpleNamespace(text=self.text, candidates=[SimpleNamespace(grounding_metadata=meta)])


@pytest.fixture
def fake(monkeypatch):
    for var in ("WAIFU_GEMINI_MODEL", "WAIFU_GEMINI_SEARCH_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WAIFU_GEMINI_SEARCH", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("WAIFU_ONLINE_SEARCH", raising=False)
    gemini.clear()
    models = FakeModels()
    monkeypatch.setattr(gemini, "_client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(gemini, "_config", lambda: "grounded-config")
    web_search._begin_search()
    yield models
    gemini.clear()


def test_off_by_default_and_without_a_key(monkeypatch):
    monkeypatch.delenv("WAIFU_GEMINI_SEARCH", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert not gemini.enabled()
    monkeypatch.setenv("WAIFU_GEMINI_SEARCH", "1")
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert not gemini.enabled()
    assert gemini.search_characters(["silver hair"]) == []


def test_grounded_answer_becomes_candidates(fake):
    out = gemini.search_characters(["female", "twintails"], None, limit=5)
    assert [c["name"] for c in out] == ["Hatsune Miku", "Rem"]
    miku = out[0]
    assert miku["medium"] == "game" and miku["source"] == "gemini"
    assert miku["source_url"] == "https://example.org/miku"  # grounding source by title
    assert out[1]["source_url"] is None
    assert "idol" in miku["tags"]  # "virtual singer", mined like any snippet
    assert fake.calls[0]["config"] == "grounded-config"
    assert web_search.last_search_meta()["errors"] == []


def test_prompt_holds_player_facts_only(fake):
    gemini.search_characters(["silver hair", "from a video game"], "game", limit=3)
    prompt = fake.calls[0]["contents"]
    assert "- silver hair" in prompt and "- medium: game" in prompt
    assert "up to 3" in prompt


def test_medium_filter_and_cache(fake):
    games = gemini.search_characters(["singer"], "game", limit=5)
    assert [c["name"] for c in games] == ["Hatsune Miku"]
    gemini.search_characters(["singer"], "game", limit=5)
    assert len(fake.calls) == 1  # cached


@pytest.mark.parametrize("text", ["no json at all", "[1, 2]", '{"name": "x"}', "[broken"])
def test_bad_replies_yield_nothing(fake, text):
    fake.text = text
    assert gemini.search_characters(["x"]) == []


def test_prose_bracket_before_the_array_is_skipped():
    text = (
        'Here are [up to 5] matches: [{"name": "Ripley", "series": "Alien", '
        '"medium": "movie", "description": "Warrant officer."}]'
    )
    got = gemini.parse_characters(text, 5)
    assert [c["name"] for c in got] == ["Ripley"]


def test_failures_are_recorded_not_raised(fake):
    fake.fail = RuntimeError("429 RESOURCE_EXHAUSTED")
    errors = []
    assert gemini.search_characters(["x"], errors=errors) == []
    assert "RESOURCE_EXHAUSTED" in errors[0]
    assert any("gemini" in e for e in web_search.last_search_meta()["errors"])


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    monkeypatch.setattr(web_search, "_want_ddg_fill", lambda *a, **k: False)
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit: [])
    monkeypatch.setattr(anilist, "search_characters", lambda q, limit: [])
    monkeypatch.setattr(sources, "popular_characters", lambda medium, n: [])


def test_inline_when_the_pool_is_empty(fake, quiet):
    out = sources.find_candidates(["female", "twintails"], background_key="g1",
                                  ddg_gate=lambda: True, specific=False)
    assert [c["name"] for c in out] == ["Hatsune Miku", "Rem"]


def test_inline_when_wikipedia_only_returned_junk(fake, quiet, monkeypatch):
    """A page that shares no visual traits must not defer Gemini to the next turn."""
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit: [{
        "id": "saint",
        "name": "The Saint",
        "series": "The Saint",
        "medium": "tv",
        "blurb": "A mystery series about a detective known as The Saint.",
        "tags": [],
    }])
    out = sources.find_candidates(
        ["pink hair halo wings"], medium_hint="game", background_key="g-saint",
        ddg_gate=lambda: True, specific=True)
    assert fake.calls  # ran before this search returned
    assert not sources._GEMINI_BG.is_pending("g-saint")
    assert any(c["name"] == "Hatsune Miku" for c in out)


def test_trait_match_is_kept_ahead_of_an_earlier_junk_page(fake, quiet, monkeypatch):
    fake.text = """```json
[{"name": "Mika Misono", "series": "Blue Archive", "medium": "game",
  "description": "She has long pink hair, angel wings on her back, and a halo."}]
```"""
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit: [{
        "id": "saint",
        "name": "The Saint",
        "series": "The Saint",
        "medium": "tv",
        "blurb": "A mystery series about a detective known as The Saint.",
        "tags": [],
    }])
    out = sources.find_candidates(
        ["pink hair halo wings"], medium_hint="game", limit=1,
        background_key="g-cut", ddg_gate=lambda: True, specific=True)
    assert [c["name"] for c in out] == ["Mika Misono"]


def test_exclude_names_does_not_hide_a_trait_seed_from_gemini(fake, quiet, monkeypatch):
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit: [{
        "id": "saint",
        "name": "The Saint",
        "series": "The Saint",
        "medium": "tv",
        "blurb": "A mystery series about a detective known as The Saint.",
        "tags": [],
    }])
    sources.find_candidates(
        ["pink hair halo wings"], medium_hint="game", exclude_names={"The Saint"},
        background_key="g-excl", ddg_gate=lambda: True, specific=True)
    assert fake.calls
    assert not sources._GEMINI_BG.is_pending("g-excl")


def test_background_when_the_pool_has_candidates(fake, quiet):
    fake.delay = 0.2
    t0 = time.perf_counter()
    out = sources.find_candidates(["female", "twintails"], exclude_names={"Makima"},
                                  background_key="g2", ddg_gate=lambda: True, specific=False)
    assert out == [] and time.perf_counter() - t0 < 0.15
    for _ in range(100):
        if not sources._GEMINI_BG.is_pending("g2"):
            break
        time.sleep(0.02)
    later = sources.find_candidates(["female"], exclude_names={"Makima"},
                                    background_key="g2", ddg_gate=lambda: False)
    assert {c["name"] for c in later} == {"Hatsune Miku", "Rem"}


def test_skipped_when_laya_says_the_pool_fits_or_there_are_no_facts(fake, quiet):
    sources.find_candidates(["female"], background_key="g3", ddg_gate=lambda: False)
    sources.find_candidates(["fictional character"], background_key="g4")
    assert fake.calls == []


def test_background_gemini_queue_is_bounded(fake, quiet, monkeypatch):
    monkeypatch.setenv("WAIFU_GEMINI_BG_MAX_PENDING", "1")
    gate = threading.Event()
    monkeypatch.setattr(gemini, "search_characters", lambda *a, **k: gate.wait(5) and [])
    try:
        assert sources._GEMINI_BG.start("a", sources._gemini_run, ["x"], None, 5)
        assert not sources._GEMINI_BG.start("b", sources._gemini_run, ["x"], None, 5)
    finally:
        gate.set()


def test_healthz_reports_gemini(monkeypatch):
    from fastapi.testclient import TestClient

    from waifu_engine.web import app

    for var in ("WAIFU_GEMINI_SEARCH", "WAIFU_GEMINI_LLM", "WAIFU_GEMINI_MODEL",
                "WAIFU_GEMINI_SEARCH_MODEL", "WAIFU_GEMINI_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    health = TestClient(app).get("/healthz").json()
    assert health["gemini"]["search"] == {"enabled": False, "model": gemini.DEFAULT_MODEL}
    assert health["gemini"]["llm"]["enabled"] is False
