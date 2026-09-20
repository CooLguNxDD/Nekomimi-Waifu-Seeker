"""DuckDuckGo character search across Anime, Comics and Games (ACG).

Two entry points:

* ``search_characters_multiround`` - the original free-text, seed-then-mine
  crawl used by the one-shot ``determine`` pipeline.
* ``search_by_constraints`` - used by the Nekomimi loop: builds queries out of
  the facts confirmed so far and returns fresh candidates for the pool.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Any

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
    ("anime", ("anime", "seiyuu", "voice actor", "studio ghibli", "tv series")),
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


def _ddgs():
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore
        return DDGS


def _search_once(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    DDGS = _ddgs()
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:  # noqa: BLE001
        if "no results" in str(e).lower():
            return []
        raise


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
    low_href = (href or "").lower()
    for domain, medium in MEDIUM_DOMAIN_HINTS:
        if domain in low_href:
            return medium
    low = f"{title} {body}".lower()
    for medium, markers in MEDIUM_TEXT_HINTS:
        if any(m in low for m in markers):
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


def _to_candidate(item: dict[str, Any], round_idx: int) -> dict[str, Any] | None:
    title = (item.get("title") or "").strip()
    body = (item.get("body") or item.get("snippet") or "").strip()
    href = (item.get("href") or item.get("link") or "").strip()
    if not title or not _is_character_page(title, href, body):
        return None
    name = _clean_name(title)
    cid = "ddg_" + hashlib.sha1((href or name).encode()).hexdigest()[:10]
    medium = _guess_medium(title, href, body)
    blurb = (body or title)[:320]
    return {
        "id": cid,
        "name": name,
        "series": _guess_series(f"{title} {body}"),
        "medium": medium,
        "tags": ["online", "duckduckgo", f"round{round_idx}"]
        + ([medium] if medium != "unknown" else [])
        + mine_trait_slugs(f"{title} {body}"),
        "blurb": blurb,
        "source_url": href,
        "source": "duckduckgo",
        "round": round_idx,
        "rounds_seen": [round_idx],
    }


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
    for idx, q in enumerate(_constraint_queries(constraints, medium_hint), start=1):
        if len(found) >= limit:
            break
        try:
            raw = _search_once(q, max_results=per_query)
        except Exception:  # noqa: BLE001 - a dead round must not kill the turn
            continue
        for item in raw:
            cand = _to_candidate(item, idx)
            if not cand or cand["id"] in exclude_ids or cand["id"] in found:
                continue
            found[cand["id"]] = cand
            if len(found) >= limit:
                break
    return list(found.values())


def search_characters_multiround(
    query: str,
    rounds: int = 3,
    per_round: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rounds = max(1, min(int(rounds), 5))
    merged: dict[str, dict[str, Any]] = {}
    logs: list[dict[str, Any]] = []
    mined: list[str] = []
    used: set[str] = set()
    seeds = _feature_seed_queries(query)

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

        # Mine names with scores
        scored_names: list[tuple[str, float]] = []
        for item in raw:
            scored_names.extend(
                _extract_names(item.get("title") or "", item.get("body") or item.get("snippet") or "")
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
                merged[cand["id"]] = cand
                added += 1
            else:
                if len(cand.get("blurb") or "") > len(prev.get("blurb") or ""):
                    prev["blurb"] = cand["blurb"]
                rs = prev.setdefault("rounds_seen", [prev.get("round", 1)])
                if ridx not in rs:
                    rs.append(ridx)

        logs.append(
            {
                "round": ridx,
                "query": q,
                "raw_hits": len(raw),
                "new_candidates": added,
                "mined_names": mined[:10],
                "error": err,
            }
        )

    return list(merged.values()), logs


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
