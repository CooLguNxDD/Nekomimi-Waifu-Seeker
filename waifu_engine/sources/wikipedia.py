"""Wikipedia (MediaWiki API) source: games, comics, and anything AniList misses.

One request returns search hits *with* their intro prose, image and categories,
so a candidate arrives with a real paragraph instead of a 320-character search
snippet. The categories are also what lets us tell a character page from a
trope page structurally, rather than by blocklisting names.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ._http import get_json

API = "https://en.wikipedia.org/w/api.php"

# "X is a fictional character", "X is a superhero appearing in..." etc.
IS_CHARACTER = re.compile(
    r"\bis (?:a|an|the)\s+(?:[a-z-]+\s+){0,3}"
    r"(character|superhero|supervillain|protagonist|antagonist|heroine|hero|villain|"
    r"anti-hero|mascot|fighter|deuteragonist)\b",
    re.I,
)
CATEGORY_HINT = re.compile(
    r"\bcharacters?\b|\bsuperheroes\b|\bsupervillains\b|\bmascots\b", re.I
)

# Franchise, game and list pages whose intro still trips IS_CHARACTER -- e.g.
# "Marvel's Spider-Man is a series of superhero action-adventure video games".
NOT_CHARACTER = re.compile(
    r"\bis (?:a|an|the)\s+(?:[a-z-]+\s+){0,2}"
    r"(series|franchise|video game|game|film|movie|television|manga|anime|list|"
    r"company|studio|album|song|genre|term|trope|alignment)\b",
    re.I,
)
DISAMBIGUATION = re.compile(r"disambiguation", re.I)

MEDIUM_CATEGORY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("comic", ("marvel comics", "dc comics", "comics characters", "superheroes",
               "supervillains", "image comics", "webtoon")),
    ("game", ("video game characters", "video game", "nintendo", "sega", "capcom",
              "square enix", "playstation", "visual novel", "fighting game")),
    ("anime", ("anime and manga characters", "anime", "shōnen", "shonen")),
    ("manga", ("manga characters", "manga")),
    # After the ACG hints, so an anime TV adaptation stays "anime".
    ("movie", ("film characters", "characters in film", "film series characters")),
    ("tv", ("television characters", "characters in television", "television series characters",
            "sitcom characters", "animated television")),
)

MEDIUM_TEXT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("game", ("video game", "video game series", "playable character", "visual novel",
              "role-playing game", "fighting game", "nintendo", "playstation")),
    ("comic", ("comic book", "comic books", "american comic", "marvel comics",
               "dc comics", "graphic novel", "webtoon")),
    ("manga", ("manga series", "manga", "light novel")),
    ("anime", ("anime series", "anime", "japanese animated")),
    ("movie", ("film", "movie")),
    ("tv", ("television series", "tv series", "sitcom")),
)

# Disambiguation, list and franchise pages that survive the category check.
BAD_TITLE = re.compile(
    r"\((disambiguation|franchise|video game series|film series|TV series)\)|"
    r"^(List of|Category:|Portal:|Template:)|"
    # Roster articles: "Characters of the Mortal Kombat series".
    r"^Characters (of|in) \b|\bcast of\b",
    re.I,
)


def _medium_of(categories: list[str], extract: str) -> str:
    joined = " ".join(categories).lower()
    for medium, markers in MEDIUM_CATEGORY_HINTS:
        if any(m in joined for m in markers):
            return medium
    low = extract.lower()
    for medium, markers in MEDIUM_TEXT_HINTS:
        if any(m in low for m in markers):
            return medium
    return "unknown"


def _is_character(title: str, categories: list[str], extract: str) -> bool:
    """Whether this Wikipedia page is one character, not a list or franchise."""
    from ..web_search import is_aggregate_page

    if not title or BAD_TITLE.search(title) or is_aggregate_page(title, "", extract):
        return False
    if any(DISAMBIGUATION.search(c) for c in categories):
        return False
    head = extract[:400]
    if any(CATEGORY_HINT.search(c) for c in categories):
        return True
    # The negative check has to come first: "is a series of superhero ... games"
    # satisfies IS_CHARACTER on its own.
    if NOT_CHARACTER.search(head):
        return False
    return bool(IS_CHARACTER.search(head))


GENERIC_SERIES = {
    "male", "female", "video game", "comics", "anime and manga", "animated",
    "animated film", "animated television", "teenage", "adoption in", "orphan",
}


def _series_of(title: str, categories: list[str], extract: str) -> str:
    """Series for this page, preferring a known franchise over category crumbs.

    ``Category:Internet meme characters`` and a ``(software)`` title used to
    beat the Vocaloid marker on Hatsune Miku and Kaito. Crypton, Project Diva
    and Sekai map to Vocaloid. Sapporo, Japanese, and other leftovers do not.
    """
    from ..web_search import franchise_label, series_is_crumb

    context = " ".join([title, *categories, (extract or "")[:400]])
    hit = franchise_label(context)
    if hit:
        return hit
    named = []
    for c in categories:
        m = re.match(r"Category:(.+?) (?:characters|superheroes|supervillains)$", c, re.I)
        if not m:
            continue
        # "Marvel Comics American superheroes" -> "Marvel Comics"
        series = re.sub(
            r"\s+(American|British|Japanese|Canadian|female|male|teenage|fictional)$",
            "",
            m.group(1).strip(),
            flags=re.I,
        ).strip()
        low = series.lower()
        if low.startswith("fictional") or low in GENERIC_SERIES or series_is_crumb(series):
            continue
        named.append(series)
    if named:
        # The most specific category is usually the longest.
        return max(named, key=len)
    m = re.search(r"\(([^)]+)\)$", title)
    if m and "character" not in m.group(1).lower() and not series_is_crumb(m.group(1)):
        return m.group(1)
    m = re.search(
        r"\b(?:from|in) (?:the )?(?:video game |comic book |manga |anime )?"
        r"(?:series |franchise )?([A-Z][\w'&:.-]*(?:\s+[A-Z][\w'&:.-]*){0,3})",
        extract[:300],
    )
    if m and not series_is_crumb(m.group(1)):
        return m.group(1)
    return "Unknown"


def _clean_title(title: str) -> str:
    """Drop Wikipedia's disambiguation parenthetical: "Black Cat (Marvel Comics)"."""
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return cleaned or title.strip()


def _pageviews(page: dict[str, Any]) -> int:
    """Sum of the last ~60 days of views -- the fame signal Nekomimi needs."""
    views = page.get("pageviews") or {}
    return sum(v for v in views.values() if isinstance(v, int))


def _to_candidate(page: dict[str, Any]) -> dict[str, Any] | None:
    title = (page.get("title") or "").strip()
    extract = re.sub(r"\s+", " ", (page.get("extract") or "")).strip()
    categories = [c.get("title", "") for c in (page.get("categories") or [])]
    if not _is_character(title, categories, extract):
        return None
    name = _clean_title(title)
    medium = _medium_of(categories, extract)
    from ..web_search import mine_trait_slugs

    tags = ["wikipedia"] + ([medium] if medium != "unknown" else [])
    tags += [t for t in mine_trait_slugs(extract[:900]) if t not in tags]
    return {
        "id": "wp_" + hashlib.sha1(title.encode()).hexdigest()[:10],
        "name": name,
        "series": _series_of(title, categories, extract),
        "medium": medium,
        "blurb": extract[:900] or f"{name}.",
        "tags": tags,
        "image_url": (page.get("thumbnail") or {}).get("source"),
        "source_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "popularity": _pageviews(page),
        "source": "wikipedia",
    }


def _query(params: dict[str, Any]) -> list[dict[str, Any]]:
    data = get_json(API, {**params, "format": "json", "formatversion": "2"})
    if not isinstance(data, dict):
        return []
    return data.get("query", {}).get("pages", []) or []


def search_characters(query: str, limit: int = 8) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    pages = _query(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query[:250],
            "gsrlimit": str(max(1, min(limit, 20))),
            "gsrnamespace": "0",
            "prop": "extracts|pageimages|categories|pageviews",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": "max",
            "piprop": "thumbnail",
            "pithumbsize": "400",
            "clshow": "!hidden",
            "cllimit": "50",
        }
    )
    out = []
    for p in sorted(pages, key=lambda p: p.get("index", 0)):
        cand = _to_candidate(p)
        if cand:
            out.append(cand)
    return out


def category_members(category: str, limit: int = 100) -> list[str]:
    """Page titles in a category -- used to seed the prebuilt catalog."""
    data = get_json(
        API,
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category if category.startswith("Category:") else f"Category:{category}",
            "cmlimit": str(max(1, min(limit, 500))),
            "cmtype": "page",
            "cmnamespace": "0",
            "format": "json",
            "formatversion": "2",
        },
    )
    if not isinstance(data, dict):
        return []
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def pages_by_title(titles: list[str]) -> list[dict[str, Any]]:
    """Fetch extracts/images/categories for up to 20 titles at a time."""
    out: list[dict[str, Any]] = []
    for i in range(0, len(titles), 20):
        chunk = titles[i : i + 20]
        pages = _query(
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "extracts|pageimages|categories|pageviews",
                "exintro": "1",
                "explaintext": "1",
                # Without exlimit the extracts API returns prose for the FIRST
                # title only, and the rest come back bare -- which silently
                # dropped Hatsune Miku and Pikachu from the catalog.
                "exlimit": "max",
                "piprop": "thumbnail",
                "pithumbsize": "400",
                "clshow": "!hidden",
                "cllimit": "max",
            }
        )
        for p in pages:
            cand = _to_candidate(p)
            if cand:
                out.append(cand)
    return out
