"""AniList GraphQL source: anime, manga and light-novel characters.

Gives what DuckDuckGo snippets never could -- a real description, a declared
gender, the source work's type, and ``favourites``, which is the popularity
signal a yes/no guesser lives on.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ._http import post_json

ENDPOINT = "https://graphql.anilist.co"

_SEARCH = """
query ($search: String, $perPage: Int) {
  Page(page: 1, perPage: $perPage) {
    characters(search: $search, sort: FAVOURITES_DESC) {
      id name { full native alternative }
      image { large }
      description(asHtml: false)
      gender age favourites siteUrl
      media(perPage: 1, sort: POPULARITY_DESC) {
        nodes { title { romaji english } type format }
      }
    }
  }
}
"""

_TOP = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    characters(sort: FAVOURITES_DESC) {
      id name { full native alternative }
      image { large }
      description(asHtml: false)
      gender age favourites siteUrl
      media(perPage: 1, sort: POPULARITY_DESC) {
        nodes { title { romaji english } type format }
      }
    }
  }
}
"""

# AniList descriptions carry markdown and spoiler markup.
_MARKUP = re.compile(r"~!.*?!~|<[^>]+>|[*_~`]|\[([^\]]*)\]\([^)]*\)", re.S)


def _clean(text: str | None, limit: int = 700) -> str:
    if not text:
        return ""
    out = _MARKUP.sub(r"\1", text)
    out = re.sub(r"\s+", " ", out).strip()
    return out[:limit]


def _medium_of(node: dict[str, Any]) -> str:
    fmt = (node.get("format") or "").upper()
    kind = (node.get("type") or "").upper()
    if fmt in {"MANGA", "ONE_SHOT"}:
        return "manga"
    if fmt in {"NOVEL", "LIGHT_NOVEL"}:
        return "manga"
    if kind == "MANGA":
        return "manga"
    return "anime"


def _tags(char: dict[str, Any], medium: str, blurb: str) -> list[str]:
    tags = ["anilist", medium]
    gender = (char.get("gender") or "").lower()
    if gender.startswith("f"):
        tags.append("female")
    elif gender.startswith("m"):
        tags.append("male")
    elif gender:
        tags.append("androgynous")
    if (char.get("favourites") or 0) >= 5000:
        tags.append("famous")
    # Reuse the shared trait vocabulary so questions and evidence line up.
    from ..web_search import mine_trait_slugs

    tags.extend(t for t in mine_trait_slugs(blurb) if t not in tags)
    return tags


def _to_candidate(char: dict[str, Any]) -> dict[str, Any] | None:
    name = ((char.get("name") or {}).get("full") or "").strip()
    if not name:
        return None
    nodes = ((char.get("media") or {}).get("nodes") or [])
    node = nodes[0] if nodes else {}
    title = (node.get("title") or {})
    series = title.get("english") or title.get("romaji") or "Unknown"
    medium = _medium_of(node)
    blurb = _clean(char.get("description"))
    return {
        "id": "al_" + hashlib.sha1(str(char.get("id") or name).encode()).hexdigest()[:10],
        "name": name,
        "series": series,
        "medium": medium,
        "blurb": blurb or f"{name} from {series}.",
        "tags": _tags(char, medium, blurb),
        "image_url": (char.get("image") or {}).get("large"),
        "source_url": char.get("siteUrl"),
        "popularity": int(char.get("favourites") or 0),
        "source": "anilist",
    }


def _page(data: Any) -> list[dict[str, Any]]:
    try:
        chars = data["data"]["Page"]["characters"]
    except (TypeError, KeyError):
        return []
    out = []
    for c in chars or []:
        cand = _to_candidate(c)
        if cand:
            out.append(cand)
    return out


def search_characters(query: str, limit: int = 10) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    data = post_json(ENDPOINT, {"query": _SEARCH, "variables": {"search": query[:120], "perPage": limit}})
    return _page(data)


def top_characters(page: int = 1, per_page: int = 50) -> list[dict[str, Any]]:
    data = post_json(ENDPOINT, {"query": _TOP, "variables": {"page": page, "perPage": per_page}})
    return _page(data)
