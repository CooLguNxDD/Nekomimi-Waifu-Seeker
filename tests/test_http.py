"""HTTP 429 retries for the character sources. No network."""

from __future__ import annotations

import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.message import Message
from email.utils import format_datetime

from waifu_engine import sources, web_search
from waifu_engine.sources import _http, anilist


def _headers(retry_after: str | None = None) -> Message:
    msg = Message()
    if retry_after is not None:
        msg["Retry-After"] = retry_after
    return msg


class _Body:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _patch_sleep(monkeypatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(_http.time, "sleep", lambda seconds: slept.append(seconds))
    monkeypatch.setattr(_http, "_throttle", lambda host: None)
    return slept


def test_get_json_retries_429_and_honours_retry_after(monkeypatch):
    _http.clear_cache()
    slept = _patch_sleep(monkeypatch)
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", _headers("1.5"), None)
        return _Body(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    data = _http.get_json("https://en.wikipedia.org/w/api.php", {"action": "query", "t": "retry"})
    assert data == {"ok": True}
    assert len(calls) == 2
    assert slept == [1.5]
    _http.clear_cache()


def test_retry_after_is_capped(monkeypatch):
    _http.clear_cache()
    slept = _patch_sleep(monkeypatch)
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", _headers("120"), None)
        return _Body(b'{"ok": 1}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _http.get_json("https://en.wikipedia.org/w/api.php", {"t": "cap"}) == {"ok": 1}
    assert slept == [_http._backoff_cap()]
    _http.clear_cache()


def test_retry_after_http_date(monkeypatch):
    frozen = 1_700_000_000.0
    monkeypatch.setattr(_http.time, "time", lambda: frozen)
    when = datetime.fromtimestamp(frozen + 5, tz=timezone.utc)
    exc = urllib.error.HTTPError(
        "https://en.wikipedia.org/w/api.php", 429, "Too Many Requests",
        _headers(format_datetime(when, usegmt=True)), None)
    assert _retry_close(exc) == 5


def _retry_close(exc: urllib.error.HTTPError) -> float:
    """Read Retry-After, then close the error the way ``_request`` does."""
    try:
        got = _http._retry_after_seconds(exc)
    finally:
        exc.close()
    assert got is not None
    return got


def test_spent_429_budget_skips_later_requests(monkeypatch):
    _http.clear_cache()
    monkeypatch.setenv("WAIFU_HTTP_429_RETRIES", "1")
    _patch_sleep(monkeypatch)
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", _headers("2"), None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _http.get_json("https://en.wikipedia.org/w/api.php", {"t": "first"}) is None
    assert len(calls) == 2  # first try plus one retry
    assert _http.host_blocked("https://en.wikipedia.org/w/api.php")
    assert _http.get_json("https://en.wikipedia.org/w/api.php", {"t": "second"}) is None
    assert len(calls) == 2
    _http.clear_cache()


def test_other_http_errors_are_not_retried(monkeypatch):
    _http.clear_cache()
    _patch_sleep(monkeypatch)
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(1)
        raise urllib.error.HTTPError(
            req.full_url, 503, "unavailable", _headers(), None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _http.get_json("https://en.wikipedia.org/w/api.php", {"t": "503"}) is None
    assert calls == [1]
    assert not _http.host_blocked("https://en.wikipedia.org/w/api.php")
    _http.clear_cache()


def test_find_candidates_stops_wikipedia_after_a_persistent_429(monkeypatch):
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    monkeypatch.setattr(anilist, "search_characters", lambda *a, **k: [])
    _http.clear_cache()
    _patch_sleep(monkeypatch)
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", _headers("0"), None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    sources.find_candidates(
        ["alpha clue", "beta clue", "gamma clue"],
        medium_hint="comic", limit=5, use_ddg=False)
    # Three template queries would each burn the retry budget. After the
    # first query the host is cooling, so the rest are not sent.
    assert len(calls) == 1 + _http._retry_budget()
    assert all("wikipedia.org" in url for url in calls)
    _http.clear_cache()
