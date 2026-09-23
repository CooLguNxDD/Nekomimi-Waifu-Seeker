"""Shared HTTP for the character sources.

Two things here are not optional, both learned the hard way:

* A real ``User-Agent``. AniList answers the default urllib UA with ``403``.
* ``certifi``'s CA bundle. The Windows system store produced
  ``CERTIFICATE_VERIFY_FAILED: certificate has expired`` on at least one host.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "waifu-determination-engine/0.3 (https://github.com/; local dev)"
TIMEOUT = 20
CACHE_TTL = 900
MIN_INTERVAL = 0.34  # per host, seconds

_ssl_ctx: ssl.SSLContext | None = None
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()
_last_call: dict[str, float] = {}
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


def _request(req: urllib.request.Request, key: str) -> Any | None:
    cached = _cache_get(key)
    if cached is not None:
        return cached
    _throttle(urllib.parse.urlparse(req.full_url).netloc)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=_context()) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    _cache_put(key, data)
    return data


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
    with _cache_lock:
        _cache.clear()
