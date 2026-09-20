"""Character sources, best first.

AniList and Wikipedia are structured APIs that return real descriptions,
images and a popularity number. DuckDuckGo scraping stays only as a last
resort for the long tail -- it was the primary in v1 and it is why the pool
filled up with trope pages.
"""

from __future__ import annotations

from typing import Any

from . import anilist, wikipedia

__all__ = ["anilist", "wikipedia", "find_candidates", "normalize_name"]


def normalize_name(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


_MEDIUM_SUFFIX = {
    "anime": "anime character",
    "manga": "manga character",
    "game": "video game character",
    "comic": "comic book character",
}


def _queries(constraints: list[str], medium_hint: str | None) -> list[str]:
    facts = [c.strip() for c in constraints if c and c.strip()]
    base = " ".join(facts[-5:])[:200].strip() or "popular character"
    suffix = _MEDIUM_SUFFIX.get(medium_hint or "", "character")
    return [f"{base} {suffix}", base]


def find_candidates(
    constraints: list[str],
    medium_hint: str | None = None,
    limit: int = 12,
    exclude_names: set[str] | None = None,
    use_ddg: bool = True,
) -> list[dict[str, Any]]:
    """Merge candidates from every source, best source first, deduped by name."""
    exclude_names = {normalize_name(n) for n in (exclude_names or set())}
    found: dict[str, dict[str, Any]] = {}

    def take(items: list[dict[str, Any]]) -> None:
        for cand in items:
            key = normalize_name(cand.get("name", ""))
            if not key or key in exclude_names or key in found:
                continue
            found[key] = cand

    queries = _queries(constraints, medium_hint)

    # Wikipedia first: it covers all four media and returns real prose.
    for q in queries:
        if len(found) >= limit:
            break
        take(wikipedia.search_characters(q, limit=limit))

    # AniList adds gender, popularity and better anime/manga coverage.
    if medium_hint in (None, "anime", "manga") and len(found) < limit:
        for q in queries:
            if len(found) >= limit:
                break
            take(anilist.search_characters(q, limit=limit))

    if use_ddg and len(found) < max(3, limit // 3):
        from ..web_search import search_by_constraints

        take(search_by_constraints(constraints, medium_hint=medium_hint, limit=limit))

    ranked = sorted(found.values(), key=lambda c: c.get("popularity", 0), reverse=True)
    return ranked[:limit]
