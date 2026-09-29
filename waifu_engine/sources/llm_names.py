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
from ..names import identity_id, same_character, same_series, series_key
from . import anilist, wikipedia
from .cache import get_entity, save_entity


# Names the model may propose per call, and how many of them are looked up.
# Each lookup is one AniList request (plus Wikipedia on a miss), all on a
# background worker; the cap bounds that worker's time.
PROPOSE = 12
RESOLVE_MAX = 10
_MEDIA_WIKI_FIRST = {"movie", "tv", "comic"}


def _matches(hit: dict[str, Any], name: str, series: str) -> bool:
    """Whether a source page matches both the proposed identity and known work.

    The name matching (word order, alias titles) is ``same_character``. A
    proposal whose page uses another spelling ("2B" vs "YoRHa No.2 Type B")
    still counts when both name the same work. A bare namesake such as Aqua
    cannot make a page from another known series satisfy the proposal.
    """
    hit_name = str(hit.get("name") or "")
    hit_series = str(hit.get("series") or "")
    requested_series = series_key(series)
    cached_series = series_key(hit_series)
    identity_match = bool(identity_id(hit_name)
                          and identity_id(hit_name) == identity_id(name))
    name_match = same_character(hit_name, name)
    if requested_series and cached_series and not same_series(hit_series, series):
        return False
    if name_match:
        if not requested_series:
            return True
        if cached_series:
            return True
        if identity_match:
            return True
        from ..web_search import franchise_mentioned

        return franchise_mentioned(series, f"{hit_name} {hit.get('blurb') or ''}")
    return identity_match or bool(
        requested_series and cached_series and same_series(hit_series, series)
    )


def _lookup(name: str, series: str, medium_hint: str | None) -> dict[str, Any] | None:
    """The first real page matching one proposal, or None."""
    
    cached = get_entity(name, series)
    if cached and _matches(cached, name, series):
        return cached
        
    order = ((wikipedia, anilist) if medium_hint in _MEDIA_WIKI_FIRST
             else (anilist, wikipedia))
    for source in order:
        query = name if source is anilist else f"{name} {series}".strip()
        for hit in source.search_characters(query, limit=3):
            if _matches(hit, name, series):
                save_entity(hit)
                return hit
    return None


def resolve(rows: list[dict[str, str]], medium_hint: str | None = None,
            errors: list[str] | None = None,
            stats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Real candidates for proposed ``{"name", "series"}`` rows, in proposal order.

    ``stats`` (when given) gains ``resolved`` and ``unresolved`` name lists.
    24 Gemma calls once added 0 candidates and the log could not say whether
    the model missed or the lookup dropped its names.
    """
    from .. import web_search

    out: list[dict[str, Any]] = []
    resolved: list[str] = []
    unresolved: list[str] = []
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
            unresolved.append(name)
            continue
        if hit is None:
            unresolved.append(name)
            continue
        resolved.append(name)
        if not any(same_character(hit["name"], o["name"]) for o in out):
            out.append(dict(hit, via="llm_names"))
    if stats is not None:
        stats["resolved"] = resolved
        stats["unresolved"] = unresolved
    return out


def search(facts: list[str], medium_hint: str | None = None,
           errors: list[str] | None = None,
           stats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Propose names for ``facts`` with the local LLM, then resolve them. Blocking.

    ``stats`` gains ``proposed`` (the model's names, player-fact output only)
    plus ``resolve``'s lists, so a zero-hit run says where the names went.
    """
    rows = query_llm.propose_characters(facts, medium_hint, PROPOSE)
    if stats is not None:
        stats["proposed"] = [row.get("name") or "" for row in rows or []]
    if not rows:
        return []
    return resolve(rows, medium_hint, errors, stats)
