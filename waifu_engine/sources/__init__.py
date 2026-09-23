"""Character sources, best first.

Order: Playwright HTML indexes → Wikipedia + AniList (structured APIs) →
DuckDuckGo fill → Playwright enrich of the top N. DuckDuckGo stays the
long-tail fallback -- it was the primary in v1 and it is why the pool
filled up with trope pages.
"""

from __future__ import annotations

import re
from typing import Any

from .. import timing
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


MAX_QUERIES = 5


def _queries(
    constraints: list[str],
    medium_hint: str | None,
    focus: list[str] | None = None,
    rewritten: list[str] | None = None,
) -> list[str]:
    """Search strings, best first.

    ``rewritten`` (from the optional query LLM) goes first verbatim, then the
    ``focus`` facts Laya ranked most distinctive, then the template groups.
    Without either, this is exactly the template list.
    """
    templates = _template_queries(constraints, medium_hint)
    if not focus and not rewritten:
        return templates
    suffix = _MEDIUM_SUFFIX.get(medium_hint or "", "character")
    lead = [" ".join(q.split())[:200] for q in (rewritten or []) if q and q.strip()]
    if focus:
        lead.append(f"{' '.join(dict.fromkeys(f[:80] for f in focus))[:200]} {suffix}")
    return list(dict.fromkeys([*lead, *templates]))[:MAX_QUERIES]


def _template_queries(constraints: list[str], medium_hint: str | None) -> list[str]:
    facts = []
    for c in constraints:
        c = (c or "").strip()
        if not c or re.match(r"(?:not true:|the character is not\b)", c, re.I):
            continue
        c = re.sub(r"^(?:The character (?:is|does|has)|(?:Is|Does|Has|Did) your character)\s+",
                   "", c, flags=re.I).rstrip("?")
        if c and c not in facts:
            facts.append(c[:80])
    suffix = _MEDIUM_SUFFIX.get(medium_hint or "", "character")
    if not facts:
        return [f"fictional {suffix}"]
    # Retry narrower combinations when an overconstrained conjunction is empty.
    # Keep the seed/first clue as an anchor and the newest clue in every query.
    groups = [[facts[0], *facts[-3:]], [facts[0], facts[-1]], [facts[-1]]]
    return list(dict.fromkeys(
        f"{' '.join(dict.fromkeys(group))[:200]} {suffix}" for group in groups
    ))


def find_candidates(
    constraints: list[str],
    medium_hint: str | None = None,
    limit: int = 12,
    exclude_names: set[str] | None = None,
    use_ddg: bool = True,
    focus: list[str] | None = None,
    rewritten: list[str] | None = None,
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

    queries = _queries(constraints, medium_hint, focus=focus, rewritten=rewritten)
    web_search._LAST_SEARCH["queries"] = list(queries)
    pw_ok = False
    pw_empty = True
    pw_error = False

    with timing.span("fetch.playwright"):
        if web_search._want_playwright():
            web_search._set_state("playwright_search")
            before = len(found)
            try:
                from .. import browser_search

                if browser_search.available():
                    for q in queries:
                        if len(found) >= limit:
                            break
                        take(web_search._playwright_hits(q, medium_hint, min(50, limit + len(exclude_names))))
                    pw_ok = True
                    pw_empty = len(found) == before
                elif web_search.search_backend() == "playwright":
                    web_search._note_error("playwright unavailable; filling with ddg")
            except Exception as exc:  # noqa: BLE001
                pw_error = True
                web_search._note_error(f"playwright: {exc}")

    # Wikipedia: covers all four media and returns real prose.
    with timing.span("fetch.wikipedia"):
        for q in queries:
            if len(found) >= limit:
                break
            try:
                take(wikipedia.search_characters(q, limit=min(50, limit + len(exclude_names))))
            except Exception as exc:  # noqa: BLE001
                web_search._note_error(f"wikipedia: {exc}")

    # AniList adds gender, popularity and better anime/manga coverage.
    with timing.span("fetch.anilist"):
        if medium_hint in (None, "anime", "manga") and len(found) < limit:
            for q in queries:
                if len(found) >= limit:
                    break
                try:
                    take(anilist.search_characters(q, limit=limit))
                except Exception as exc:  # noqa: BLE001
                    web_search._note_error(f"anilist: {exc}")

    need_ddg = use_ddg and web_search._want_ddg_fill(
        len(found), limit, pw_ok=pw_ok, pw_empty=pw_empty, pw_error=pw_error
    )
    with timing.span("fetch.ddg"):
        if need_ddg:
            web_search._set_state("fill_ddg")
            web_search._TLS.nested_ddg = True
            try:
                for q in queries:
                    if len(found) >= limit:
                        break
                    take(web_search.search_by_constraints(
                        [q], medium_hint=medium_hint,
                        limit=min(50, limit + len(exclude_names)),
                    ))
            except Exception as exc:  # noqa: BLE001
                web_search._note_error(f"ddg: {exc}")
            finally:
                web_search._TLS.nested_ddg = False

    # Preserve provider relevance. Fame is only a capped prior after retrieval.
    out = list(found.values())[:limit]
    with timing.span("fetch.enrich"):
        if web_search._enrich_on() and out:
            web_search._set_state("enrich")
            try:
                web_search.enrich_candidates(out)
            except Exception as exc:  # noqa: BLE001
                web_search._note_error(f"enrich: {exc}")
    web_search._set_state("done")
    timing.note(found=len(out))
    return out
