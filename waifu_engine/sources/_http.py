"""Shared HTTP for the character sources.

Two things here are not optional, both learned the hard way:

* A real ``User-Agent``. AniList answers the default urllib UA with ``403``.
* ``certifi``'s CA bundle. The Windows system store produced
  ``CERTIFICATE_VERIFY_FAILED: certificate has expired`` on at least one host.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any

USER_AGENT = "waifu-determination-engine/0.3 (https://github.com/; local dev)"
TIMEOUT = 20
CACHE_TTL = 900
MIN_INTERVAL = 0.34  # per host, seconds

_ssl_ctx: ssl.SSLContext | None = None
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()
_last_call: dict[str, float] = {}
_cool_until: dict[str, float] = {}
_rate_lock = threading.Lock()


def _context() -> ssl.SSLContext:
    global _ssl_ctx
    if _ssl_ctx is None:
        try:
            import certifi

            _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001 - fall back to the system store
            _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _throttle(host: str) -> None:
    with _rate_lock:
        last = _last_call.get(host, 0.0)
        wait = MIN_INTERVAL - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.time()


def _cache_get(key: str) -> Any | None:
    with _cache_lock:
        hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    return None


def _cache_put(key: str, value: Any) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), value)


def _env_float(name: str, default: float) -> float:
    """``name`` as a float, or ``default`` when it is unset or not a number."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """``name`` as an int, or ``default`` when it is unset or not an integer."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _retry_budget() -> int:
    """Extra attempts after the first HTTP 429 (``WAIFU_HTTP_429_RETRIES``)."""
    return max(0, _env_int("WAIFU_HTTP_429_RETRIES", 2))


def _backoff_cap() -> float:
    """Longest a single 429 may stall a turn, in seconds.

    Wikipedia sometimes asks for minutes. Waiting that out inside one player
    turn is worse than skipping the rest of the host's queries.
    """
    return max(0.0, _env_float("WAIFU_HTTP_429_CAP", 8.0))


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    """Seconds the server asked us to wait, or None when it did not say.

    ``Retry-After`` is either a delta in seconds or an HTTP date. A missing
    or unreadable header falls through to exponential backoff.
    """
    headers = getattr(exc, "headers", None) or getattr(exc, "hdrs", None)
    raw = headers.get("Retry-After") if headers is not None else None
    if not raw:
        return None
    text = str(raw).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - time.time())


def _min_cooldown() -> float:
    """Shortest post-429 pause, so a ``Retry-After: 0`` does not resume immediately.

    The following queries in the same search would otherwise hit the limit
    again and return empty.
    """
    return min(_backoff_cap(), max(0.05, _env_float("WAIFU_HTTP_429_BACKOFF", 0.8)))


def _backoff(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait before the next try, capped.

    Retry-After wins when the server sends one. Otherwise the wait doubles
    from ``WAIFU_HTTP_429_BACKOFF`` so a burst of 429s is not a tight loop.
    """
    hinted = _retry_after_seconds(exc)
    if hinted is None:
        hinted = _env_float("WAIFU_HTTP_429_BACKOFF", 0.8) * (2 ** attempt)
    return min(_backoff_cap(), max(0.0, hinted))


def _cool_down(host: str, seconds: float) -> None:
    """Keep ``host`` from being called until ``seconds`` have passed."""
    with _rate_lock:
        until = time.time() + max(0.0, seconds)
        _cool_until[host] = max(_cool_until.get(host, 0.0), until)


def host_blocked(url: str) -> bool:
    """Whether ``url``'s host is cooling down after a spent 429 retry budget.

    Callers stop the rest of that host's queries: each one would return
    empty and count against the per-host gap.
    """
    host = urllib.parse.urlparse(url).netloc
    with _rate_lock:
        return time.time() < _cool_until.get(host, 0.0)


def _close_error(exc: BaseException) -> None:
    """Release an HTTP error body so a retried 429 does not leak the socket."""
    close = getattr(exc, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # noqa: BLE001 - closing must not hide the status
        pass


def _request(req: urllib.request.Request, key: str) -> Any | None:
    """Fetch and cache JSON, or None when the request fails.

    HTTP 429 used to return None on the first response, so Wikipedia looked
    empty and the next query in the same search hit the limit again. The
    status is retried with Retry-After (or a doubling wait). Once the budget
    is spent the host cools down and later calls skip the network.
    """
    cached = _cache_get(key)
    if cached is not None:
        return cached
    host = urllib.parse.urlparse(req.full_url).netloc
    if host_blocked(f"https://{host}/"):
        return None
    attempt = 0
    budget = _retry_budget()
    while True:
        _throttle(host)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT, context=_context()) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < budget:
                delay = _backoff(exc, attempt)
                attempt += 1
                _close_error(exc)
                time.sleep(delay)
                continue
            if exc.code == 429:
                # Retry-After: 0 means "try again now". The queries still
                # queued for this host would do exactly that and come back
                # empty, so the cooldown is at least one backoff step.
                _cool_down(host, max(_backoff(exc, attempt), _min_cooldown()))
            # A failed request must look different from "no results": record it
            # for the search meta and the per-turn timing line.
            _note_error(host, exc)
            _close_error(exc)
            return None
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            _note_error(host, exc)
            return None
        with _rate_lock:
            _cool_until.pop(host, None)
        _cache_put(key, data)
        return data


def _note_error(host: str, exc: BaseException) -> None:
    try:
        from .. import web_search

        web_search._note_error(f"{host}: {str(exc)[:120]}")
    except Exception:  # noqa: BLE001 - diagnostics must never break a search
        pass


def get_json(url: str, params: dict[str, Any], accept: str = "application/json") -> Any | None:
    full = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT, "Accept": accept})
    return _request(req, full)


def post_json(url: str, payload: dict[str, Any]) -> Any | None:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    # Hash the whole body: truncating it made every GraphQL page share one
    # cache key, so paging through AniList returned page 1 six times.
    return _request(req, f"{url}|{hashlib.sha1(body).hexdigest()}")


def clear_cache() -> None:
    """Drop cached responses and 429 cooldowns. Tests use this between cases."""
    with _cache_lock:
        _cache.clear()
    with _rate_lock:
        _cool_until.clear()
