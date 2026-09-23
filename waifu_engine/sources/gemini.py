"""Gemini with Google Search grounding, as a candidate source.

Name searches (AniList, Wikipedia) cannot match trait-only facts such as
"female, video game, silver hair"; Gemini can, because it runs Google searches
itself and reads the results. It is asked to *list candidate characters* for
the player's facts -- nothing else. It never writes question text and never
makes a decision: its answer is untrusted web content, parsed defensively and
scored by Laya like any other search hit.

Input is player facts only (search terms, medium). Scraped names and blurbs are
never put in the prompt, so a web page cannot steer the model.

Off unless ``WAIFU_GEMINI_SEARCH=1`` and an API key is set (``GEMINI_API_KEY``
or ``GOOGLE_API_KEY``); every call is billed. Needs the optional
``google-genai`` package (``pip install -e .[gemini]``). ``search_characters``
never raises: failures are recorded with ``web_search._note_error`` and return
``[]``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from typing import Any

DEFAULT_MODEL = "gemini-2.5-flash"
MEDIA = ("anime", "manga", "comic", "game")
_YES = {"1", "true", "yes"}

PROMPT = (
    "Use Google Search to find fictional characters from anime, manga, comics or "
    "video games (including visual novels and gacha games) who match ALL of these "
    "facts about one character:\n{facts}\n\n"
    "Reply with ONLY a JSON array of up to {n} objects, best match first, each "
    '{{"name": ..., "series": ..., "medium": "anime"|"manga"|"comic"|"game", '
    '"description": one sentence on appearance and role}}. No other text.'
)

_FENCE = re.compile(r"```(?:json)?", re.I)
_LOCK = threading.Lock()
_CLIENT: Any = None
_CACHE: dict[tuple, tuple[float, list[dict[str, Any]]]] = {}
CACHE_TTL = 900


def _api_key() -> str:
    """The Gemini key: ``GEMINI_API_KEY``, else ``GOOGLE_API_KEY`` (as the SDK reads them)."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def enabled() -> bool:
    """On only when explicitly switched on, online, and a key is configured."""
    if os.getenv("WAIFU_ONLINE_SEARCH", "1").strip().lower() not in _YES:
        return False
    return os.getenv("WAIFU_GEMINI_SEARCH", "0").strip().lower() in _YES and bool(_api_key())


def model() -> str:
    """``WAIFU_GEMINI_MODEL``: any Gemini model that supports the Google Search tool."""
    return os.getenv("WAIFU_GEMINI_MODEL", DEFAULT_MODEL)


def _timeout_ms() -> int:
    """``WAIFU_GEMINI_TIMEOUT`` seconds as the SDK's millisecond timeout."""
    try:
        return int(float(os.getenv("WAIFU_GEMINI_TIMEOUT", "15")) * 1000)
    except ValueError:
        return 15000


def _client() -> Any:
    """Process-wide ``genai.Client``; raises ImportError without google-genai."""
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from google import genai
            from google.genai import types

            _CLIENT = genai.Client(
                api_key=_api_key(), http_options=types.HttpOptions(timeout=_timeout_ms()))
        return _CLIENT


def _config() -> Any:
    """Request config with the Google Search tool on; no JSON mode (grounding forbids it)."""
    from google.genai import types

    return types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0,
    )


def _prompt(facts: list[str], medium_hint: str | None, n: int) -> str:
    """The fixed instruction filled with the player's facts, one per line."""
    lines = [f"- {' '.join(f.split())[:200]}" for f in facts if f and f.strip()]
    if medium_hint in MEDIA:
        lines.append(f"- medium: {medium_hint}")
    return PROMPT.format(facts="\n".join(lines) or "- (none)", n=n)


def parse_characters(text: str, n: int) -> list[dict[str, str]]:
    """Pull a JSON array of character objects out of a model reply.

    Grounded replies cannot use JSON mode, so the array is located in free
    text (code fences and prose around it are ignored). Anything that is not a
    well-formed object with a name is dropped; strings are length-capped.
    """
    text = _FENCE.sub("", text or "")
    start = text.find("[")
    if start < 0:
        return []
    try:
        items, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return []
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("name") or "").split())[:80]
        if not name:
            continue
        medium = str(item.get("medium") or "").strip().lower()
        out.append({
            "name": name,
            "series": " ".join(str(item.get("series") or "").split())[:120],
            "medium": medium if medium in MEDIA else "unknown",
            "description": " ".join(str(item.get("description") or "").split())[:320],
        })
        if len(out) >= n:
            break
    return out


def _grounding(response: Any) -> tuple[list[tuple[str, str]], list[str]]:
    """(title, uri) of the web sources and the Google queries Gemini ran."""
    try:
        meta = response.candidates[0].grounding_metadata
    except (AttributeError, IndexError, TypeError):
        return [], []
    if meta is None:
        return [], []
    sources = []
    for chunk in getattr(meta, "grounding_chunks", None) or []:
        web = getattr(chunk, "web", None)
        if web is not None and getattr(web, "uri", None):
            sources.append((getattr(web, "title", "") or "", web.uri))
    return sources, list(getattr(meta, "web_search_queries", None) or [])


def _to_candidate(item: dict[str, str], sources: list[tuple[str, str]]) -> dict[str, Any]:
    """A parsed character in the shape every source returns (``gem_`` id, mined tags)."""
    from ..web_search import mine_trait_slugs

    medium = item["medium"]
    # Grounding sources are per answer, not per character; link one whose
    # title names this character when there is one.
    low = item["name"].lower()
    url = next((uri for title, uri in sources if low in title.lower()), None)
    cid = "gem_" + hashlib.sha1(f"{item['name']}|{item['series']}".encode()).hexdigest()[:10]
    return {
        "id": cid,
        "name": item["name"],
        "series": item["series"],
        "medium": medium,
        "blurb": item["description"],
        "tags": ["online", "gemini"] + ([medium] if medium != "unknown" else [])
        + mine_trait_slugs(item["description"]),
        "source_url": url,
        "source": "gemini",
        "popularity": 0,
    }


def search_characters(
    facts: list[str],
    medium_hint: str | None = None,
    limit: int = 8,
    errors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Candidates matching the player's facts, via grounded Gemini. Never raises.

    Results are cached for ``CACHE_TTL`` seconds per (facts, medium, limit):
    a grounded call costs seconds and money, and the same facts recur across
    turns and sessions.
    """
    from .. import web_search

    if not enabled():
        return []
    key = (tuple(f.strip() for f in facts if f and f.strip()), medium_hint, limit, model())
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return [dict(c) for c in hit[1]]
    try:
        response = _client().models.generate_content(
            model=model(), contents=_prompt(facts, medium_hint, limit), config=_config())
        sources, searched = _grounding(response)
        items = parse_characters(getattr(response, "text", "") or "", limit)
    except Exception as exc:  # noqa: BLE001 - quota, timeout, missing SDK: degrade
        msg = f"gemini: {str(exc)[:160]}"
        web_search._note_error(msg)
        if errors is not None:
            errors.append(msg)
        return []
    if medium_hint in MEDIA:
        allowed = {"anime", "manga"} if medium_hint in ("anime", "manga") else {medium_hint}
        items = [i for i in items if i["medium"] in allowed | {"unknown"}]
    out = [_to_candidate(i, sources) for i in items]
    if searched:
        web_search._LAST_SEARCH.setdefault("gemini_queries", []).extend(searched[:5])
    if len(_CACHE) > 256:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (time.time(), out)
    return [dict(c) for c in out]


def status() -> dict[str, Any]:
    """For ``/healthz``: whether Gemini search is on and with which model."""
    try:
        import google.genai  # noqa: F401

        sdk = True
    except Exception:  # noqa: BLE001
        sdk = False
    return {"enabled": enabled(), "model": model(), "sdk": sdk, "key": bool(_api_key())}


def clear() -> None:
    """Forget the cached client and results (tests)."""
    global _CLIENT
    with _LOCK:
        _CLIENT = None
    _CACHE.clear()
