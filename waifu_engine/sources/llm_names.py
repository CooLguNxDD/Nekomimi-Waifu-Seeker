"""Local-LLM hypotheses, resolved to real candidates.

Broad button facts ("female, demon, red hair, adult") cannot be name-searched,
and the popular pool only holds the famous. BED-LLM's loop has the model
*propose* hypotheses and a separate likelihood model judge them; here the
query LLM (``query_llm.propose_characters``) proposes names and Laya judges
them like any other candidate.

A proposed name is only a lead. Each one is looked up on AniList (then
Wikipedia), and only a page whose name or series matches the proposal is
kept, so a hallucinated name never enters the pool, and every profile, tag
and popularity number comes from that page -- never from the model.

Input to the model is player facts only (the same rule as the query rewrite).
``resolve`` and ``search`` never raise: failures go to
``web_search._note_error`` and yield ``[]``.
"""

from __future__ import annotations

from typing import Any

from .. import query_llm
from ..names import same_character, same_series, series_key
from . import anilist, wikipedia

# Names the model may propose per call, and how many of them are looked up.
# Each lookup is one AniList request (plus Wikipedia on a miss), all on a
# background worker; the cap bounds that worker's time.
PROPOSE = 12
RESOLVE_MAX = 10
_MEDIA_WIKI_FIRST = {"movie", "tv", "comic"}


def _matches(hit: dict[str, Any], name: str, series: str) -> bool:
    """Whether a source page is the proposed character.

    The name matching (word order, alias titles) is ``same_character``. A
    proposal whose page uses another spelling ("2B" vs "YoRHa No.2 Type B")
    still counts when both name the same work.
    """
    if same_character(hit.get("name") or "", name):
        return True
    return bool(series and series_key(hit.get("series") or "")
                and same_series(hit.get("series") or "", series))


def _lookup(name: str, series: str, medium_hint: str | None) -> dict[str, Any] | None:
    """The first real page matching one proposal, or None.

    AniList covers anime, manga and games; film, TV and Western comics are
    tried on Wikipedia first, where AniList has nothing.
    """
    order = ((wikipedia, anilist) if medium_hint in _MEDIA_WIKI_FIRST
             else (anilist, wikipedia))
    for source in order:
        query = name if source is anilist else f"{name} {series}".strip()
        for hit in source.search_characters(query, limit=3):
            if _matches(hit, name, series):
                return hit
    return None


def resolve(rows: list[dict[str, str]], medium_hint: str | None = None,
            errors: list[str] | None = None) -> list[dict[str, Any]]:
    """Real candidates for proposed ``{"name", "series"}`` rows, in proposal order."""
    from .. import web_search

    out: list[dict[str, Any]] = []
    for row in rows[:RESOLVE_MAX]:
        name = row.get("name") or ""
        if not name:
            continue
        try:
            hit = _lookup(name, row.get("series") or "", medium_hint)
        except Exception as exc:  # noqa: BLE001 - one bad lookup must not end the batch
            note = f"llm_names: {exc}"
            web_search._note_error(note)
            if errors is not None:
                errors.append(note)
            continue
        if hit is not None and not any(same_character(hit["name"], o["name"]) for o in out):
            out.append(dict(hit, via="llm_names"))
    return out


def search(facts: list[str], medium_hint: str | None = None,
           errors: list[str] | None = None) -> list[dict[str, Any]]:
    """Propose names for ``facts`` with the local LLM, then resolve them. Blocking."""
    rows = query_llm.propose_characters(facts, medium_hint, PROPOSE)
    if not rows:
        return []
    return resolve(rows, medium_hint, errors)
