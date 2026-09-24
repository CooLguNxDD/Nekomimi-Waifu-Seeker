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
from typing import Any
from urllib.parse import urlparse

from . import browser_search
from .names import name_keys, same_character

LISTICLE = re.compile(
    r"(top\s*\d+|\d+\s*best|best\s+\d+|ranked|list of|tier list|husbando material|certified|pinterest)",
    re.I,
)

SERIES_BLOCK = {
    # anime / manga
    "naruto", "one piece", "black clover", "demon slayer", "grand blue", "mob psycho",
    "bungou stray dogs", "attack on titan", "my hero academia", "jujutsu kaisen",
    "chainsaw man", "fairy tail", "hunter x hunter", "tokyo revengers", "dr stone",
    "food wars", "vinland saga", "sword art online", "re zero", "re:zero",
    "spy x family", "frieren", "oshi no ko", "cowboy bebop", "neon genesis evangelion",
    "fullmetal alchemist", "death note", "dragon ball", "sailor moon", "bleach",
    # games
    "blue archive", "genshin impact", "honkai", "honkai star rail", "fate grand order",
    "final fantasy", "street fighter", "tekken", "guilty gear", "persona", "nier automata",
    "league of legends", "overwatch", "the legend of zelda", "super mario", "pokemon",
    "elden ring", "dark souls", "arknights", "azur lane", "umamusume", "vocaloid",
    # comics
    "marvel comics", "dc comics", "justice league", "the avengers", "x men", "teen titans",
    # noise
    "discover pinterest", "all time", "you will fall", "progression",
    "kokuhaku jikkou iinkai",
}

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


def _guess_series(text: str) -> str:
    low = text.lower()
    for marker, label in (
        ("blue archive", "Blue Archive"),
        ("honkai: star rail", "Honkai: Star Rail"),
        ("star rail", "Honkai: Star Rail"),
        ("fate/grand order", "Fate/Grand Order"),
        ("final fantasy", "Final Fantasy"),
        ("street fighter", "Street Fighter"),
        ("guilty gear", "Guilty Gear"),
        ("league of legends", "League of Legends"),
        ("elden ring", "Elden Ring"),
        ("dark souls", "Dark Souls"),
        ("the legend of zelda", "The Legend of Zelda"),
        ("super mario", "Super Mario"),
        ("pokemon", "Pokemon"),
        ("persona", "Persona"),
        ("nier", "NieR"),
        ("overwatch", "Overwatch"),
        ("arknights", "Arknights"),
        ("azur lane", "Azur Lane"),
        ("vocaloid", "Vocaloid"),
        ("marvel", "Marvel Comics"),
        ("dc comics", "DC Comics"),
        ("batman", "DC Comics"),
        ("spider-man", "Marvel Comics"),
        ("x-men", "Marvel Comics"),
        ("spy x family", "Spy x Family"),
        ("frieren", "Frieren"),
        ("oshi no ko", "Oshi no Ko"),
        ("evangelion", "Neon Genesis Evangelion"),
        ("cowboy bebop", "Cowboy Bebop"),
        ("death note", "Death Note"),
        ("dragon ball", "Dragon Ball"),
        ("sailor moon", "Sailor Moon"),
        ("bleach", "Bleach"),
        ("genshin impact", "Genshin Impact"),
        ("chainsaw man", "Chainsaw Man"),
        ("attack on titan", "Attack on Titan"),
        ("sword art online", "Sword Art Online"),
        ("re:zero", "Re:Zero"),
        ("one piece", "One Piece"),
        ("naruto", "Naruto"),
        ("jujutsu kaisen", "Jujutsu Kaisen"),
        ("demon slayer", "Demon Slayer"),
        ("tokyo revengers", "Tokyo Revengers"),
        ("black clover", "Black Clover"),
        ("fullmetal alchemist", "Fullmetal Alchemist"),
        ("my hero academia", "My Hero Academia"),
        ("assassination classroom", "Assassination Classroom"),
        ("food wars", "Food Wars"),
        ("dr. stone", "Dr. Stone"),
        ("seven deadly sins", "Seven Deadly Sins"),
    ):
        if marker in low:
            return label
    return "Web result"


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
    key = name.lower().strip()
    if not key or key in SERIES_BLOCK:
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


# Trait vocabulary mined out of snippets. Slugs line up with
# ``nekomimi.traits`` tag names so evidence and questions share a vocabulary.
TRAIT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("female", ("she", "her", "heroine", "woman", "girl", "female")),
    ("male", ("he", "his", "him", "hero", "man", "boy", "male")),
    ("protagonist", ("protagonist", "main character", "player character")),
    ("antagonist", ("antagonist", "villain", "main enemy")),
    ("playable", ("playable",)),
    ("student", ("student", "high school", "academy")),
    ("soldier", ("soldier", "knight", "mercenary", "warrior")),
    ("mage", ("mage", "wizard", "witch", "sorcerer", "spellcaster")),
    ("ninja", ("ninja", "shinobi", "assassin")),
    ("idol", ("idol", "singer", "virtual singer")),
    ("royalty", ("princess", "prince", "queen", "king", "noble")),
    ("robot", ("android", "robot", "cyborg", "artificial intelligence")),
    ("demon", ("demon", "devil", "oni", "vampire")),
    ("god", ("goddess", "god", "deity")),
    ("sword", ("sword", "katana", "blade")),
    ("gun", ("gun", "pistol", "rifle", "firearm")),
    ("magic", ("magic", "spell", "mana")),
    ("tsundere", ("tsundere",)),
    ("kuudere", ("kuudere", "stoic", "emotionless")),
    ("yandere", ("yandere",)),
    ("genki", ("cheerful", "energetic", "genki")),
    ("blonde", ("blonde", "blond", "golden hair")),
    ("black", ("black hair", "black-haired")),
    ("white", ("white hair", "silver hair", "grey hair", "gray hair")),
    ("red", ("red hair", "crimson hair", "redhead")),
    ("blue", ("blue hair",)),
    ("pink", ("pink hair",)),
    ("brown", ("brown hair", "brown-haired")),
    ("green", ("green hair", "green-haired")),
    ("purple", ("purple hair", "violet hair", "purple-haired")),
    # Eye colours are prefixed so they never collide with the hair slugs.
    ("eyes-blue", ("blue eyes", "blue-eyed")),
    ("eyes-red", ("red eyes", "crimson eyes", "red-eyed")),
    ("eyes-green", ("green eyes", "green-eyed")),
    ("eyes-gold", ("golden eyes", "gold eyes", "amber eyes", "yellow eyes")),
    ("eyes-brown", ("brown eyes", "brown-eyed", "dark eyes")),
    ("glasses", ("glasses", "spectacles")),
    ("fantasy", ("fantasy", "kingdom", "magic world")),
    ("scifi", ("sci-fi", "science fiction", "cyberpunk", "mecha", "spaceship")),
    ("school", ("school", "classroom", "academy")),
)


@lru_cache(maxsize=512)
def _marker_re(marker: str) -> "re.Pattern[str]":
    # Whole-word match: plain substring matching made "he" fire on "the".
    return re.compile(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", re.I)


def mine_trait_slugs(text: str) -> list[str]:
    low = text.lower()
    return [
        slug
        for slug, markers in TRAIT_PATTERNS
        if any(_marker_re(m).search(low) for m in markers)
    ]


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


def _constraint_queries(constraints: list[str], medium_hint: str | None = None) -> list[str]:
    """Build searches from the facts confirmed so far in a guessing session."""
    facts = [c.strip() for c in constraints if c and c.strip()][-6:]
    base = " ".join(facts)[:180].strip()
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
    facts = [c.strip() for c in constraints if c and c.strip()][-6:]
    return " ".join(facts)[:180].strip() or "popular character"


def _take_candidate(
    found: dict[str, dict[str, Any]],
    seen_names: set[str],
    cand: dict[str, Any] | None,
    exclude_ids: set[str],
    limit: int,
) -> bool:
    """Add ``cand`` unless its id or name (either word order) is taken; True at ``limit``."""
    if not cand or cand["id"] in exclude_ids or cand["id"] in found:
        return False
    keys = name_keys(cand.get("name", ""))
    if not keys or keys & seen_names:
        return False
    found[cand["id"]] = cand
    seen_names |= keys
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
