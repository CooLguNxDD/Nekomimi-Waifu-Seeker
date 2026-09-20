"""Character sources, best first.

Order: Playwright HTML indexes → Wikipedia + AniList (structured APIs) →
DuckDuckGo fill → Playwright enrich of the top N. DuckDuckGo stays the
long-tail fallback -- it was the primary in v1 and it is why the pool
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
    from .. import web_search

    exclude_names = {normalize_name(n) for n in (exclude_names or set())}
    found: dict[str, dict[str, Any]] = {}
    web_search._begin_search()

    def take(items: list[dict[str, Any]]) -> None:
        for cand in items:
            key = normalize_name(cand.get("name", ""))
            if not key or key in exclude_names or key in found:
                continue
            found[key] = cand

    queries = _queries(constraints, medium_hint)
    pw_ok = False
    pw_empty = True
    pw_error = False

    if web_search._want_playwright():
        web_search._set_state("playwright_search")
        before = len(found)
        try:
            from .. import browser_search

            if browser_search.available():
                q = queries[0] if queries else "popular character"
                take(web_search._playwright_hits(q, medium_hint, limit))
                pw_ok = True
                pw_empty = len(found) == before
            elif web_search.search_backend() == "playwright":
                web_search._note_error("playwright unavailable; filling with ddg")
        except Exception as exc:  # noqa: BLE001
            pw_error = True
            web_search._note_error(f"playwright: {exc}")

    # Wikipedia: covers all four media and returns real prose.
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

    need_ddg = use_ddg and web_search._want_ddg_fill(
        len(found), limit, pw_ok=pw_ok, pw_empty=pw_empty, pw_error=pw_error
    )
    if need_ddg:
        web_search._set_state("fill_ddg")
        web_search._TLS.nested_ddg = True
        try:
            take(web_search.search_by_constraints(
                constraints, medium_hint=medium_hint, limit=limit
            ))
        except Exception as exc:  # noqa: BLE001
            web_search._note_error(f"ddg: {exc}")
        finally:
            web_search._TLS.nested_ddg = False

    ranked = sorted(found.values(), key=lambda c: c.get("popularity", 0), reverse=True)
    out = ranked[:limit]
    if web_search._enrich_on() and out:
        web_search._set_state("enrich")
        try:
            web_search.enrich_candidates(out)
        except Exception as exc:  # noqa: BLE001
            web_search._note_error(f"enrich: {exc}")
    web_search._set_state("done")
    return out
