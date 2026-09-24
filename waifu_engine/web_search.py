"""Character search across Anime, Comics and Games (ACG).

Playwright (headless Chromium) is the primary HTML-index backend. DuckDuckGo
(``ddgs``) fills remaining slots when Playwright is missing, empty, or errors.

Two entry points (signatures unchanged):

* ``search_characters_multiround`` - the original free-text, seed-then-mine
  crawl used by the one-shot ``determine`` pipeline.
* ``search_by_constraints`` - used by the Nekomimi loop: builds queries out of
  the facts confirmed so far and returns fresh candidates for the pool.

State machine per call: idle → playwright_search → fill_ddg → enrich → done.
Any step can skip to the next; nothing here raises out of a request.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from functools import lru_cache
from typing import Any, Iterable
from urllib.parse import urlparse

from . import browser_search
from .names import already_seen, identity_id, name_keys, same_character, series_key
from .nekomimi.lexicon import (
    AGGREGATE_EXACT as _AGGREGATE_EXACT,
    CHARACTER_IDENTITIES as _CHARACTER_IDENTITIES,
    COLOR_WORDS as _COLOR_WORDS,
    HEADLINERS as _HEADLINERS,
    NAME_BLOCK as SERIES_BLOCK,
    NON_CHARACTERS as _NON_CHARACTERS,
    SERIES_CRUMBS as _SERIES_CRUMBS,
    SERIES_MARKERS as _SERIES_MARKERS,
    TRAIT_PATTERNS,
)

LISTICLE = re.compile(
    r"(top\s*\d+|\d+\s*best|best\s+\d+|ranked|list of|tier list|husbando material|certified|pinterest)",
    re.I,
)

# Name block: lexicon/franchises.yml ``name_block``. ``_ok_person_name`` and the
# plural-stem branch of ``is_aggregate_page`` both read ``SERIES_BLOCK``.

MEDIUM_DOMAIN_HINTS: tuple[tuple[str, str], ...] = (
    ("myanimelist.net", "anime"),
    ("anilist.co", "anime"),
    ("animenewsnetwork.com", "anime"),
    ("vndb.org", "game"),
    ("giantbomb.com", "game"),
    ("gamefaqs.gamespot.com", "game"),
    ("comicvine.gamespot.com", "comic"),
    ("marvel.fandom.com", "comic"),
    ("dc.fandom.com", "comic"),
    ("marvel.com", "comic"),
    ("dc.com", "comic"),
)

MEDIUM_TEXT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("game", ("video game", "playable character", "rpg", "visual novel", "gacha",
              "fighting game", "playstation", "nintendo", "steam", "mmorpg")),
    ("comic", ("comic book", "marvel", "dc comics", "superhero", "graphic novel",
               "webtoon", "issue #")),
    ("manga", ("manga", "light novel", "shonen", "shoujo", "seinen")),
    ("anime", ("anime", "seiyuu", "voice actor", "studio ghibli")),
    # After the ACG hints: an anime TV series stays "anime".
    ("movie", ("film", "movie", "box office", "pixar", "disney animated")),
    ("tv", ("tv series", "television series", "sitcom", "netflix series", "hbo")),
)

STOP = {
    "the","this","that","with","from","anime","manga","game","male","female","best","top",
    "list","character","characters","wiki","fandom","myanimelist","english","japanese",
    "voice","actor","blonde","blond","handsome","beautiful","popular","series","season",
    "these","his","her","him","she","they","them","but","and","looking","check","hair",
    "eyes","here","there","when","what","your","our","their","about","into","over",
    "after","before","also","only","just","like","than","then","some","many","most",
    "more","such","very","much","good","great","cute","cool","hot","new","old","free",
    "read","information","student","school","year","height","hobby","profile","material",
    "certified","husbando","waifu","crush","crushes","ranked","review","guide","watch",
    "online","download","episode","movie","film","official","discover","pinterest",
}


SEARCH_STATES = ("idle", "playwright_search", "fill_ddg", "enrich", "done")
_YES = {"1", "true", "yes"}
_TLS = threading.local()
_LAST_SEARCH: dict[str, Any] = {
    "backend": "auto",
    "search_state": "idle",
    "enriched": 0,
    "errors": [],
}


def search_backend() -> str:
    v = os.getenv("WAIFU_SEARCH_BACKEND", "auto").strip().lower()
    return v if v in {"auto", "playwright", "ddg"} else "auto"


def last_search_meta() -> dict[str, Any]:
    """Copy of the most recent search meta (backend, state, enriched, errors, queries)."""
    return {
        "backend": _LAST_SEARCH.get("backend"),
        "search_state": _LAST_SEARCH.get("search_state"),
        "enriched": _LAST_SEARCH.get("enriched", 0),
        "errors": list(_LAST_SEARCH.get("errors") or []),
        "queries": list(_LAST_SEARCH.get("queries") or []),
    }


def _begin_search() -> dict[str, Any]:
    _LAST_SEARCH["backend"] = search_backend()
    _LAST_SEARCH["search_state"] = "idle"
    _LAST_SEARCH["enriched"] = 0
    _LAST_SEARCH["errors"] = []
    _LAST_SEARCH["queries"] = []
    return _LAST_SEARCH


def _set_state(state: str) -> None:
    if state in SEARCH_STATES:
        _LAST_SEARCH["search_state"] = state


def _note_error(msg: str) -> None:
    errors = _LAST_SEARCH.setdefault("errors", [])
    if msg and msg not in errors:
        errors.append(msg)


def _want_playwright() -> bool:
    return search_backend() in {"auto", "playwright"}


def _enrich_on() -> bool:
    return os.getenv("WAIFU_PLAYWRIGHT_ENRICH", "1").strip().lower() in _YES


def _enrich_limit() -> int:
    try:
        return max(0, int(os.getenv("WAIFU_PLAYWRIGHT_ENRICH_LIMIT", "8")))
    except ValueError:
        return 8




def _want_ddg_fill(
    got: int,
    limit: int,
    *,
    pw_ok: bool,
    pw_empty: bool,
    pw_error: bool,
) -> bool:
    """Whether DuckDuckGo should fill remaining slots.

    * ``auto`` / ``ddg`` — fill until ``limit``.
    * ``playwright`` — skip unless Playwright was unavailable, empty, or errored.
    """
    if got >= limit:
        return False
    backend = search_backend()
    if backend == "playwright":
        return (not pw_ok) or pw_empty or pw_error
    return True


def _ddgs():
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore
        return DDGS


# One DuckDuckGo client for the process: opening a fresh one per request
# repeats the connection setup every time. The lock also keeps the request
# thread and the background filler from sharing it at once.
_DDG_CLIENT: Any = None
_DDG_LOCK = threading.Lock()


def _ddg_timeout() -> float:
    try:
        return float(os.getenv("WAIFU_DDG_TIMEOUT", "5"))
    except ValueError:
        return 5.0


def _search_once(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    global _DDG_CLIENT
    with _DDG_LOCK:
        if _DDG_CLIENT is None:
            DDGS = _ddgs()
            try:
                _DDG_CLIENT = DDGS(timeout=_ddg_timeout())
            except TypeError:  # older clients without a timeout argument
                _DDG_CLIENT = DDGS()
        try:
            return list(_DDG_CLIENT.text(query, max_results=max_results))
        except Exception as e:  # noqa: BLE001
            if "no results" in str(e).lower():
                return []
            _DDG_CLIENT = None  # a broken session is not reused
            raise


def ddg_max_requests() -> int:
    try:
        return max(0, int(os.getenv("WAIFU_DDG_MAX_REQUESTS", "3")))
    except ValueError:
        return 3


def ddg_budget() -> float:
    try:
        return max(0.0, float(os.getenv("WAIFU_DDG_BUDGET", "4")))
    except ValueError:
        return 4.0


def ddg_quick(
    queries: list[str],
    limit: int,
    max_requests: int | None = None,
    budget: float | None = None,
    per_query: int = 8,
    errors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Capped DuckDuckGo fill for the guessing loop. Never raises.

    One generic search per query (not the seven site-specific variants of
    ``_constraint_queries``), at most ``max_requests`` requests, and no new
    request once ``budget`` seconds have passed. Uncapped, one turn made ~35
    sequential requests and took ~40 s.
    """
    max_requests = ddg_max_requests() if max_requests is None else max_requests
    budget = ddg_budget() if budget is None else budget
    t0 = time.perf_counter()
    found: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for idx, q in enumerate(queries[:max_requests], start=1):
        if len(found) >= limit or (idx > 1 and time.perf_counter() - t0 >= budget):
            break
        try:
            raw = _search_once(f"{q} wiki", max_results=per_query)
        except Exception as exc:  # noqa: BLE001 - a dead request must not kill the turn
            _note_error(f"ddg: {exc}")
            if errors is not None:
                errors.append(f"ddg: {exc}")
            continue
        for item in raw:
            if _take_candidate(found, seen, _to_candidate(item, idx), set(), limit):
                break
    return list(found.values())


def _clean_name(title: str) -> str:
    name = title.split(" - ")[0].split(" | ")[0].split(" – ")[0].strip()
    name = re.sub(r"\s*\(.*?\)\s*$", "", name).strip() or title
    name = re.sub(
        r"\s+(Character|Wiki|Fandom|MyAnimeList|MAL|Overview|Profile|Anime)$",
        "",
        name,
        flags=re.I,
    ).strip()
    return name[:80]


def _guess_medium(title: str, href: str, body: str) -> str:
    """Medium implied by a search hit, or ``unknown``.

    A domain that hosts one medium wins. IMDb hosts film and television, so
    it is not in that map: labeling every IMDb hit a movie scored TV
    characters at 0.16 on a TV answer. Markers are whole words, the same
    rule as trait slugs — ``film`` matched ``filmed`` and ``hbo`` matched
    ``neighbor``, and a mislabeled candidate was then eliminated.
    """
    low_href = (href or "").lower()
    for domain, medium in MEDIUM_DOMAIN_HINTS:
        if domain in low_href:
            return medium
    low = f"{title} {body}".lower()
    for medium, markers in MEDIUM_TEXT_HINTS:
        if any(_marker_re(m).search(low) for m in markers):
            return medium
    return "unknown"


# Ordered franchise markers: lexicon/franchises.yml. ``_phrase_re`` is the
# matcher; longer Vocaloid aliases stay first in that file.


def _phrase_re(phrase: str) -> re.Pattern[str]:
    """Whole-phrase pattern for ``phrase``, allowing a plural ``s`` ("VOCALOIDs").

    Plain substring matching turned "personality" into Persona and
    "bleached" into Bleach.
    """
    return re.compile(r"(?<!\w)" + re.escape(phrase.lower()) + r"s?(?!\w)")


_SERIES_MARKER_RES = tuple((_phrase_re(m), label) for m, label in _SERIES_MARKERS)


def franchise_label(text: str) -> str:
    """Display name of a known franchise named in ``text``, or ``""``.

    Vocaloid aliases (Crypton, Project Diva, Sekai) share one label so a
    series question can offer "Vocaloid" instead of a category crumb.
    Markers match whole phrases only.
    """
    low = (text or "").lower()
    for marker_re, label in _SERIES_MARKER_RES:
        if marker_re.search(low):
            return label
    return ""


def franchise_mentioned(franchise: str, text: str) -> bool:
    """Whether ``text`` names ``franchise`` as a whole phrase."""
    return bool(franchise) and bool(_phrase_re(franchise).search((text or "").lower()))


def _headliner_hits() -> tuple[tuple[int, re.Pattern[str], str, tuple[str, ...]], ...]:
    """Phrase patterns sorted longest-first so "final fantasy vii" beats a shorter title."""
    flat: list[tuple[int, re.Pattern[str], str, tuple[str, ...]]] = []
    for row in _HEADLINERS:
        names = tuple(row["names"])
        for phrase in row["phrases"]:
            flat.append((len(phrase), _phrase_re(phrase), row["franchise"], names))
    flat.sort(key=lambda item: item[0], reverse=True)
    return tuple(flat)


_HEADLINER_HITS = _headliner_hits()


def headliner_from_texts(texts: Iterable[str]) -> tuple[str, tuple[str, ...]] | None:
    """Return ``(franchise, title names)`` for the longest headliner phrase in ``texts``.

    Player text only. A blurb that mentions the work must not select it: that
    is how a crossover used to join the wrong cast.
    """
    best: tuple[int, str, tuple[str, ...]] | None = None
    for text in texts:
        low = (text or "").lower()
        if not low:
            continue
        for length, pattern, franchise, names in _HEADLINER_HITS:
            if best is not None and length <= best[0]:
                break
            if pattern.search(low):
                best = (length, franchise, names)
                break
    if best is None:
        return None
    return best[1], best[2]


_NAME_WORD = re.compile(r"[a-z0-9]+")


def _name_tokens(name: str) -> set[str]:
    """Lower-case words of ``name``. Used to tell a spelling from a second name."""
    return set(_NAME_WORD.findall((name or "").lower()))


def disambiguating_alias(name: str, franchise: str) -> str:
    """Shortest lexicon alias of ``name`` whose words are not already in ``name``.

    A mantle title is also the character ("Wonder Woman"), so a search for
    that string ranks later holders. "Diana Prince" shares no words with the
    title. Romanisations ("Sohryu") still share words with the primary name
    and are not a second person, so they are not returned.
    """
    primary = _name_tokens(name)
    if len(primary) < 1:
        return ""
    slug = identity_id(name)
    if not slug:
        return ""
    aliases: tuple[str, ...] = ()
    for row in _CHARACTER_IDENTITIES:
        if row["id"] == slug:
            aliases = row["names"]
            break
    work = _name_tokens(franchise)
    best = ""
    best_key: tuple[int, int] | None = None
    for alias in aliases:
        tokens = _name_tokens(alias)
        if len(tokens) < 2 or tokens & primary:
            continue
        if work and work <= tokens:
            continue
        key = (len(tokens), len(alias))
        if best_key is None or key < best_key:
            best, best_key = alias, key
    return best


def title_character_queries(texts: Iterable[str]) -> list[str]:
    """Character names to send a name search, or ``[]`` when no headliner matches.

    AniList searches names and sorts by favourites. A trait sentence
    ("Neon Genesis Evangelion female character") does not name Asuka, and the
    favourites sort then fills the pool with unrelated leads.
    """
    hit = headliner_from_texts(texts)
    if not hit:
        return []
    franchise, names = hit
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
        alias = disambiguating_alias(name, franchise)
        if alias and alias not in out:
            out.append(alias)
    return out


def fulltext_pin_parts(texts: Iterable[str]) -> list[str]:
    """Work label, a sole title name, and any personal alias, for full-text search.

    Every later query has to keep this lead. The newest trait alone retrieves
    co-cast and other mantle pages. Several co-leads are not all AND-ed into
    the Wikipedia string (that misses a page about only one of them); each
    name is searched on its own by ``title_character_queries``.
    """
    hit = headliner_from_texts(texts)
    label = ""
    names: tuple[str, ...] = ()
    if hit:
        label, names = hit
    if not label:
        for text in texts:
            label = franchise_label(text or "")
            if label:
                break
    if not label:
        return []
    parts = [label]
    if len(names) == 1 and _name_tokens(names[0]) - _name_tokens(label):
        parts.append(names[0])
    for name in names:
        alias = disambiguating_alias(name, label)
        if alias and alias not in parts:
            parts.append(alias)
    return parts


def headliner_label(text: str) -> str:
    """Franchise display name named in ``text``, or ``""`` when none matches."""
    hit = headliner_from_texts([text])
    return hit[0] if hit else ""


# Page titles that are a work or a species, not a person. series_key folds case.
_NON_CHARACTER_KEYS = frozenset(key for name in _NON_CHARACTERS if (key := series_key(name)))
# First sentence of a franchise or species article. Bare "series" is not
# enough: "is a series regular" and "is the protagonist of the anime series"
# are still people. "series of" / "franchise" / "species" are the page.
_WORK_OR_SPECIES = re.compile(
    r"\b(?:is|are)\s+(?:a|an|the)\s+(?:[a-z0-9-]+\s+){0,4}"
    r"(?:species|franchise|series\s+of)\b",
    re.I,
)


# Category crumbs: lexicon/franchises.yml. ``series_is_crumb`` is an exact
# normalized string match; these are not regular expressions.


def series_is_crumb(series: str) -> bool:
    """Whether ``series`` is a category leftover rather than a work title."""
    low = " ".join((series or "").lower().split()).strip(".,;:")
    return low in _SERIES_CRUMBS


def _guess_series(text: str) -> str:
    """Series implied by a hit, or ``Web result`` when no franchise is known."""
    return franchise_label(text) or "Web result"


# Colour words: lexicon/colors.yml. ``is_color_phrase`` still requires every
# token to be in the set, so a colour-only detail is not searched as a name.


def is_color_phrase(text: str) -> bool:
    """Whether ``text`` is only hair-colour words, with no character or series name."""
    tokens = re.findall(r"[a-z]+", (text or "").lower())
    return bool(tokens) and all(token in _COLOR_WORDS for token in tokens)


_AGGREGATE_NAME = re.compile(
    r"(^|\b)(list of|category:|portal:|template:)|"
    r"\bdisambiguation\b|"
    r"^characters (of|in)\b|"
    r"(^|/)characters?$",
    re.I,
)
# A path that ends at the roster itself: /characters, /wiki/X/Characters.
_ROSTER_URL = re.compile(r"/characters?/?(?:[?#].*)?$")
# Exact roster titles: lexicon/franchises.yml ``aggregate_exact``. The rest of
# ``is_aggregate_page`` (regexes, plural stem against ``SERIES_BLOCK``) stays here.


def is_non_character(name: str, url: str = "", blurb: str = "") -> bool:
    """Whether this hit is a work, species, list, or category rather than one person.

    "Evangelion" and "Angels" soaked the Asuka posterior and were eligible to
    be guessed. List pages stay out through ``is_aggregate_page``. A title on
    the non-character list is out even when the blurb is empty. The opening
    sentence is checked too, because a franchise page can be titled with a
    name that is not on that list yet.
    """
    if is_aggregate_page(name, url, blurb):
        return True
    if series_key(name) in _NON_CHARACTER_KEYS:
        return True
    head = re.split(r"(?<=[.!?])\s", (blurb or "").strip(), maxsplit=1)[0][:240]
    return bool(head) and bool(_WORK_OR_SPECIES.search(head))


def is_aggregate_page(name: str, url: str = "", blurb: str = "") -> bool:
    """Whether this hit is a list, category, or disambiguation page.

    Fandom roster titles such as VOCALOIDs and Vocaloid/Characters pass a
    one-word name check and a ``/wiki/`` accept, then outrank the character.
    """
    raw = (name or "").strip()
    low = raw.lower()
    if not low:
        return False
    if low in _AGGREGATE_EXACT:
        return True
    stem = low.rstrip("s")
    if stem != low and stem in SERIES_BLOCK:
        return True
    if _AGGREGATE_NAME.search(raw):
        return True
    url_l = (url or "").lower()
    if any(token in url_l for token in ("list_of_", "category:", "disambiguation")):
        return True
    # A roster endpoint ends in /characters; /characters/hatsune-miku is a
    # profile and must survive.
    if _ROSTER_URL.search(url_l):
        return True
    head = (blurb or "")[:200].lower()
    if "list of characters" in head or head.startswith("this is a list") or "the following is a list" in head:
        return True
    return False


# Wiki scaffolding and trope pages that look like names but are not characters.
NON_NAME = re.compile(
    r"^(category|tag|trait|portal|template|file|help|special|list|index|glossary)[:\s]",
    re.I,
)
NON_NAME_WORDS = {
    "topics", "battles", "cast", "syndrome", "tropes", "trope", "characters",
    "protagonist", "deuteragonist", "antagonist", "navigation", "contents",
    "gallery", "appearances", "relationships", "trivia", "quotes",
}
JUNK_DOMAINS = ("tvtropes.org", "reddit.com", "quora.com", "youtube.com", "pinterest.")


def _ok_person_name(name: str) -> bool:
    """Whether ``name`` can be a character, not a franchise, list, or crumb page."""
    key = name.lower().strip()
    if not key or key in SERIES_BLOCK or is_aggregate_page(name):
        return False
    if ":" in name or NON_NAME.match(name):
        return False
    parts = name.split()
    if any(p.lower() in STOP for p in parts):
        return False
    if any(p.lower() in NON_NAME_WORDS for p in parts):
        return False
    # "FRIENDLY PLUMBER" and friends: headline text, not a character name.
    if name.isupper() and len(parts) > 1:
        return False
    if LISTICLE.search(name):
        return False
    if len(parts) == 1 and len(parts[0]) < 4:
        return False
    if len(parts) > 3:
        return False
    return True


# Neutral, medium-aware cues. (The old cue list weighted "blonde" and
# "male/husbando" heavily -- leftovers from an unrelated query that skewed
# every search toward blond men.)
CHARACTER_CUES = (
    "character", "protagonist", "antagonist", "playable", "voiced by",
    "voice actor", "seiyuu", "first appearance", "debut", "created by",
)
SOURCE_CUES = (
    "anime", "manga", "video game", "comic", "series", "franchise",
    "visual novel", "light novel", "webtoon",
)


def _extract_names(title: str, body: str) -> list[tuple[str, float]]:
    """Return (name, score); score reflects how character-page-like the context is."""
    text = f"{title} {body}"
    scored: dict[str, float] = {}
    for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})\b", text):
        name = m.group(1).strip()
        if not _ok_person_name(name):
            continue
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        window = text[start:end].lower()
        score = 1.0
        if any(w in window for w in CHARACTER_CUES):
            score += 1.5
        if any(w in window for w in SOURCE_CUES):
            score += 0.5
        if " is a " in window or " is an " in window:
            score += 0.5
        scored[name] = max(scored.get(name, 0), score)
    return sorted(scored.items(), key=lambda x: x[1], reverse=True)


def _is_character_page(title: str, href: str, body: str) -> bool:
    parsed = urlparse(href or "")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    character_paths = {
        "vndb.org": r"/c\d+/?$",
        "myanimelist.net": r"/character/\d+(?:/|$)",
        "anilist.co": r"/character/\d+(?:/|$)",
    }
    if host in character_paths and not re.match(character_paths[host], parsed.path):
        return False
    if LISTICLE.search(title or ""):
        return False
    if any(d in (href or "").lower() for d in JUNK_DOMAINS):
        return False
    if re.match(r"^\d+", title or ""):
        return False
    name = _clean_name(title)
    if not _ok_person_name(name):
        return False
    low_href = (href or "").lower()
    # Strong accept: MAL/AniList/Fandom character URLs
    if any(
        x in low_href
        for x in (
            "myanimelist.net/character",
            "anilist.co/character",
            "/wiki/",
            "fandom.com/wiki/",
            "vndb.org/c",
            "giantbomb.com/",
            "comicvine.gamespot.com/",
        )
    ):
        return True
    low = f"{title} {body}".lower()
    return any(
        x in low
        for x in (
            " is a ", " is an ", "protagonist", "voice actor", "seiyuu",
            "character from", "playable character", "first appearance", "created by",
        )
    )


def _to_candidate(
    item: dict[str, Any],
    round_idx: int,
    source: str = "duckduckgo",
) -> dict[str, Any] | None:
    title = (item.get("title") or "").strip()
    body = (item.get("body") or item.get("snippet") or "").strip()
    href = (item.get("href") or item.get("link") or "").strip()
    if not title or not _is_character_page(title, href, body):
        return None
    name = _clean_name(title)
    src = "playwright" if source == "playwright" else "duckduckgo"
    prefix = "pw_" if src == "playwright" else "ddg_"
    cid = prefix + hashlib.sha1((href or name).encode()).hexdigest()[:10]
    medium = _guess_medium(title, href, body)
    blurb = (body or title)[:320]
    return {
        "id": cid,
        "name": name,
        "series": _guess_series(f"{title} {body}"),
        "medium": medium,
        "tags": ["online", src, f"round{round_idx}"]
        + ([medium] if medium != "unknown" else [])
        + mine_trait_slugs(f"{title} {body}"),
        "blurb": blurb,
        "source_url": href,
        "source": src,
        "round": round_idx,
        "rounds_seen": [round_idx],
    }


def _playwright_hits(
    query: str,
    medium_hint: str | None,
    limit: int,
    round_idx: int = 1,
) -> list[dict[str, Any]]:
    """Run Playwright search and convert hits. Never raises."""
    try:
        if not browser_search.available():
            if search_backend() == "playwright":
                _note_error("playwright unavailable; filling with ddg")
            return []
        raw = browser_search.search(query, medium_hint=medium_hint, limit=limit)
    except Exception as exc:  # noqa: BLE001
        _note_error(f"playwright: {exc}")
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        cand = _to_candidate(item, round_idx, source="playwright")
        if cand:
            out.append(cand)
    return out


def enrich_candidates(candidates: list[dict[str, Any]], limit: int | None = None) -> int:
    """Enrich the top of a shortlist via Playwright. Never raises."""
    if not candidates or not _enrich_on():
        return 0
    try:
        if not browser_search.available():
            return 0
    except Exception as exc:  # noqa: BLE001
        _note_error(f"enrich: {exc}")
        return 0
    cap = _enrich_limit() if limit is None else max(0, int(limit))
    n = 0
    for cand in candidates[:cap]:
        try:
            if browser_search.enrich(cand) is not None:
                n += 1
        except Exception as exc:  # noqa: BLE001
            _note_error(f"enrich: {exc}")
    _LAST_SEARCH["enriched"] = n
    return n


# Trait markers: lexicon/traits.yml, compiled by ``_marker_re``. Wings stay
# reconciled in ``mine_trait_slugs`` via ``traits.mined_visual_slugs``.


@lru_cache(maxsize=512)
def _marker_re(marker: str) -> "re.Pattern[str]":
    # Whole-word match: plain substring matching made "he" fire on "the".
    return re.compile(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", re.I)


def mine_trait_slugs(text: str) -> list[str]:
    """Tag slugs described in ``text``.

    ``wings`` is reconciled with ``traits.has_body_wings``. The marker list
    includes "angel wings", and a halo described as "a heart with wings" must
    not receive the tag or every Trinity student looks winged.
    """
    low = text.lower()
    found = [
        slug
        for slug, markers in TRAIT_PATTERNS
        if any(_marker_re(m).search(low) for m in markers)
    ]
    # Lazy import: the nekomimi package init pulls the engine, which imports
    # this module, so a top-level import would cycle.
    from .nekomimi.traits import mined_visual_slugs

    visual = set(mined_visual_slugs(text))
    for slug in ("halo", "wings", "horns"):
        if slug in visual and slug not in found:
            found.append(slug)
        elif slug == "wings" and slug not in visual and slug in found:
            found.remove(slug)
    return found


def _feature_seed_queries(query: str) -> list[str]:
    """ACG-wide seed queries: anime/manga, games and comics all get a shot."""
    q = query.strip()
    return [
        f"site:myanimelist.net/character {q}",
        f"site:fandom.com/wiki {q} character",
        f"site:vndb.org {q} character",
        f"site:comicvine.gamespot.com {q} character",
        f"{q} video game character wiki",
        f"{q} anime character profile",
        f"{q} comic book character first appearance",
    ]


def _joined_constraints(constraints: list[str]) -> str:
    """Newest facts, with a resolved work put back in front of the window.

    The last-six slice dropped a series answered early, and full-text search
    then ran on the newest trait alone.
    """
    raw = [c.strip() for c in constraints if c and c.strip()]
    facts = raw[-6:]
    have = {fact.lower() for fact in facts}
    lead = [
        part for part in fulltext_pin_parts(raw)
        if part.lower() not in have and not any(part.lower() in fact.lower() for fact in facts)
    ]
    return " ".join([*lead, *facts])[:180].strip()


def _constraint_queries(constraints: list[str], medium_hint: str | None = None) -> list[str]:
    """Build searches from the facts confirmed so far in a guessing session."""
    base = _joined_constraints(constraints)
    if not base:
        base = "popular character"
    media = [medium_hint] if medium_hint and medium_hint != "unknown" else ["anime", "game", "comic"]
    queries: list[str] = []
    for medium in media:
        if medium == "game":
            queries += [
                f"site:fandom.com/wiki {base} video game character",
                f"site:vndb.org {base}",
            ]
        elif medium == "comic":
            queries += [
                f"site:comicvine.gamespot.com {base}",
                f"site:fandom.com/wiki {base} comic character",
            ]
        else:
            queries += [
                f"site:myanimelist.net/character {base}",
                f"site:fandom.com/wiki {base} anime character",
            ]
    queries.append(f"{base} character wiki")
    return queries


def _constraint_query_text(constraints: list[str]) -> str:
    """The same joined text ``_constraint_queries`` searches, without a site scope."""
    return _joined_constraints(constraints) or "popular character"


def _take_candidate(
    found: dict[str, dict[str, Any]],
    seen_names: set[str],
    cand: dict[str, Any] | None,
    exclude_ids: set[str],
    limit: int,
) -> bool:
    """Add ``cand`` unless its id or name is already taken; True when ``found`` is at ``limit``.

    Identity is ``names.same_character``: either word order, and a trailing
    series title ("Link" / "Link (The Legend of Zelda)"), are one person.
    "Young Link" is not. ``seen_names`` holds the raw names already taken.
    """
    if not cand or cand["id"] in exclude_ids or cand["id"] in found:
        return False
    name = cand.get("name", "")
    if not name_keys(name) or already_seen(name, seen_names):
        return False
    found[cand["id"]] = cand
    seen_names.add(name)
    return len(found) >= limit


def _ddg_constraint_hits(
    constraints: list[str],
    found: dict[str, dict[str, Any]],
    seen_names: set[str],
    exclude_ids: set[str],
    medium_hint: str | None,
    limit: int,
    per_query: int,
) -> None:
    """DuckDuckGo leaf used by the fill_ddg step. ``_search_once`` stays the DDG client."""
    for idx, q in enumerate(_constraint_queries(constraints, medium_hint), start=1):
        if len(found) >= limit:
            break
        try:
            raw = _search_once(q, max_results=per_query)
        except Exception as exc:  # noqa: BLE001 - a dead round must not kill the turn
            _note_error(f"ddg: {exc}")
            continue
        for item in raw:
            if _take_candidate(found, seen_names, _to_candidate(item, idx), exclude_ids, limit):
                break


def search_by_constraints(
    constraints: list[str],
    exclude_ids: set[str] | None = None,
    medium_hint: str | None = None,
    limit: int = 12,
    per_query: int = 6,
) -> list[dict[str, Any]]:
    """Fetch fresh candidates matching the facts a guessing session has confirmed."""
    exclude_ids = exclude_ids or set()
    found: dict[str, dict[str, Any]] = {}
    seen_names: set[str] = set()
    nested = bool(getattr(_TLS, "nested_ddg", False))
    if not nested:
        _begin_search()

    pw_ok = False
    pw_empty = True
    pw_error = False
    if not nested and _want_playwright():
        _set_state("playwright_search")
        before = len(found)
        try:
            if browser_search.available():
                for cand in _playwright_hits(
                    _constraint_query_text(constraints), medium_hint, limit
                ):
                    _take_candidate(found, seen_names, cand, exclude_ids, limit)
                pw_ok = True
                pw_empty = len(found) == before
            else:
                if search_backend() == "playwright":
                    _note_error("playwright unavailable; filling with ddg")
        except Exception as exc:  # noqa: BLE001
            pw_error = True
            _note_error(f"playwright: {exc}")

    do_ddg = True if nested else _want_ddg_fill(
        len(found), limit, pw_ok=pw_ok, pw_empty=pw_empty, pw_error=pw_error
    )
    if do_ddg and len(found) < limit:
        _set_state("fill_ddg")
        _ddg_constraint_hits(
            constraints, found, seen_names, exclude_ids, medium_hint, limit, per_query
        )

    out = list(found.values())[:limit]
    if not nested:
        if _enrich_on() and out:
            _set_state("enrich")
            try:
                enrich_candidates(out)
            except Exception as exc:  # noqa: BLE001
                _note_error(f"enrich: {exc}")
        _set_state("done")
    return out


def search_characters_multiround(
    query: str,
    rounds: int = 3,
    per_round: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One-shot mode's search: up to ``rounds`` widening rounds, merged by
    id and name. Returns (candidates, per-round logs)."""
    rounds = max(1, min(int(rounds), 5))
    merged: dict[str, dict[str, Any]] = {}
    logs: list[dict[str, Any]] = []
    mined: list[str] = []
    used: set[str] = set()
    seeds = _feature_seed_queries(query)
    _begin_search()

    def _stamp(log: dict[str, Any]) -> dict[str, Any]:
        log["search_state"] = _LAST_SEARCH.get("search_state")
        log["backend"] = _LAST_SEARCH.get("backend")
        log["enriched"] = _LAST_SEARCH.get("enriched", 0)
        log["errors"] = list(_LAST_SEARCH.get("errors") or [])
        return log

    pw_ok = False
    pw_empty = True
    pw_error = False
    if _want_playwright():
        _set_state("playwright_search")
        err = None
        raw: list[dict[str, Any]] = []
        added = 0
        try:
            if browser_search.available():
                raw = browser_search.search(query, limit=per_round)
                pw_ok = True
            elif search_backend() == "playwright":
                _note_error("playwright unavailable; filling with ddg")
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            pw_error = True
            _note_error(f"playwright: {exc}")
        for item in raw:
            scored = _extract_names(
                item.get("title") or "", item.get("body") or item.get("snippet") or ""
            )
            for name, _score in scored:
                if name.lower() in {m.lower() for m in mined}:
                    continue
                if name.lower() in {c["name"].lower() for c in merged.values()}:
                    continue
                mined.append(name)
            cand = _to_candidate(item, 1, source="playwright")
            if not cand:
                continue
            if cand["id"] not in merged:
                merged[cand["id"]] = cand
                added += 1
        pw_empty = not merged
        logs.append(
            _stamp(
                {
                    "round": 0,
                    "query": query,
                    "raw_hits": len(raw),
                    "new_candidates": added,
                    "mined_names": mined[:10],
                    "error": err,
                    "source": "playwright",
                }
            )
        )

    do_ddg = _want_ddg_fill(
        len(merged), per_round * rounds, pw_ok=pw_ok, pw_empty=pw_empty, pw_error=pw_error
    )
    if do_ddg:
        _set_state("fill_ddg")
        for ridx in range(1, rounds + 1):
            if mined and ridx >= 2:
                name = mined.pop(0)
                q = f'"{name}" site:myanimelist.net/character OR site:fandom.com/wiki'
            elif seeds:
                q = seeds.pop(0)
            else:
                q = f"{query.strip()} anime character wiki"
            if q.lower() in used:
                q = f"{query.strip()} character {ridx}"
            used.add(q.lower())

            try:
                raw = _search_once(q, max_results=per_round)
                err = None
            except Exception as e:  # noqa: BLE001
                raw, err = [], str(e)
                _note_error(f"ddg: {e}")

            scored_names: list[tuple[str, float]] = []
            for item in raw:
                scored_names.extend(
                    _extract_names(
                        item.get("title") or "", item.get("body") or item.get("snippet") or ""
                    )
                )
            scored_names.sort(key=lambda x: x[1], reverse=True)
            for name, _score in scored_names:
                if name.lower() in {m.lower() for m in mined}:
                    continue
                if name.lower() in {c["name"].lower() for c in merged.values()}:
                    continue
                mined.append(name)

            added = 0
            for item in raw:
                cand = _to_candidate(item, ridx)
                if not cand:
                    continue
                prev = merged.get(cand["id"])
                if prev is None:
                    # Name already taken by a Playwright hit — keep the first source.
                    if any(same_character(cand["name"], c["name"]) for c in merged.values()):
                        continue
                    merged[cand["id"]] = cand
                    added += 1
                else:
                    if len(cand.get("blurb") or "") > len(prev.get("blurb") or ""):
                        prev["blurb"] = cand["blurb"]
                    rs = prev.setdefault("rounds_seen", [prev.get("round", 1)])
                    if ridx not in rs:
                        rs.append(ridx)

            logs.append(
                _stamp(
                    {
                        "round": ridx,
                        "query": q,
                        "raw_hits": len(raw),
                        "new_candidates": added,
                        "mined_names": mined[:10],
                        "error": err,
                    }
                )
            )

    out = list(merged.values())
    if _enrich_on() and out:
        _set_state("enrich")
        try:
            enrich_candidates(out)
        except Exception as exc:  # noqa: BLE001
            _note_error(f"enrich: {exc}")
    _set_state("done")
    if logs:
        logs[-1] = _stamp(logs[-1])
    else:
        logs.append(
            _stamp(
                {
                    "round": 0,
                    "query": query,
                    "raw_hits": 0,
                    "new_candidates": 0,
                    "mined_names": [],
                    "error": None,
                }
            )
        )
    return out, logs


def fetch_image_url(query: str) -> str | None:
    try:
        DDGS = _ddgs()
        with DDGS() as ddgs:
            for item in ddgs.images(f"{query} anime character", max_results=6):
                url = (item.get("image") or item.get("url") or "").strip()
                if url and not url.lower().endswith(".svg"):
                    return url
    except Exception:
        return None
    return None


def attach_images(candidates: list[dict], user_query: str, limit: int = 8) -> list[dict]:
    for c in candidates[:limit]:
        if c.get("image_url"):
            continue
        name = (c.get("name") or "").strip()
        url = fetch_image_url(name or user_query)
        if url:
            c["image_url"] = url
    return candidates
