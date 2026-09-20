from __future__ import annotations

import re
from typing import Any

from .catalog import character_text

STOP = {
    "a", "an", "the", "and", "or", "with", "who", "has", "have", "is", "are",
    "of", "to", "for", "in", "on", "my", "me", "i", "want", "looking", "like",
    "prefer", "girl", "waifu", "character", "anime",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOP and len(t) > 1]


def keyword_score(query: str, character: dict[str, Any]) -> float:
    tokens = tokenize(query)
    if not tokens:
        return 0.0
    hay = character_text(character).lower()

    def whole(tok: str) -> bool:
        return re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", hay) is not None

    hits = sum(1 for t in tokens if whole(t))
    name = (character.get("name") or "").lower()
    q = query.lower().strip()
    if q and (q == name or name.startswith(q) or q.startswith(name) or q in name):
        hits += max(2, len(tokens))
    phrase_bonus = 0.4 if q and q in name else 0.0
    # prefer candidates seen in more rounds
    rounds_seen = character.get("rounds_seen") or [character.get("round", 1)]
    phrase_bonus += 0.05 * max(0, len(rounds_seen) - 1)
    return hits / max(len(tokens), 1) + phrase_bonus


def shortlist(query: str, top_k: int = 8) -> list[tuple[dict[str, Any], float]]:
    """Search-backed convenience API; never reads a local character catalog."""
    return shortlist_with_online(query, top_k=top_k)[0]


def shortlist_with_online(
    query: str,
    top_k: int = 8,
    online: bool = True,
    rounds: int = 3,
) -> tuple[list[tuple[dict[str, Any], float]], dict[str, Any]]:
    meta: dict[str, Any] = {
        "online_used": False,
        "online_count": 0,
        "online_error": None,
        "rounds": [],
        "catalog_size": 0,
    }
    if not online:
        meta["online_error"] = "online search disabled"
        return [], meta

    try:
        from .web_search import attach_images, last_search_meta, search_characters_multiround

        remote, logs = search_characters_multiround(query, rounds=rounds, per_round=5)
        remote = attach_images(remote, query, limit=min(top_k, 8))
    except Exception as e:  # noqa: BLE001
        meta["online_error"] = str(e)
        return [], meta

    meta["online_used"] = True
    meta["online_count"] = len(remote)
    meta["rounds"] = logs
    got = last_search_meta()
    meta["backend"] = got.get("backend")
    meta["enriched"] = got.get("enriched", 0)
    meta["search_state"] = got.get("search_state")
    meta["errors"] = got.get("errors") or []

    merged: dict[str, tuple[dict[str, Any], float]] = {}
    for c in remote:
        s = keyword_score(query, c) + 0.35
        toks = tokenize(query)
        name = (c.get("name") or "").lower()
        if toks and all(t in name for t in toks[:3] if len(t) > 2):
            s += 0.6
        prev = merged.get(c["id"])
        if prev is None or s > prev[1]:
            merged[c["id"]] = (c, s)

    ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k], meta
