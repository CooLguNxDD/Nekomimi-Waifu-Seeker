from __future__ import annotations

import os
from typing import Any

from .nekomimi import laya_client
from .catalog import character_text
from .search import keyword_score, shortlist_with_online


def _fallback_decide(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        ((c, keyword_score(query, c)) for c in candidates),
        key=lambda x: x[1],
        reverse=True,
    )
    winner, raw = ranked[0]
    total = sum(max(s, 0.01) for _, s in ranked) or 1.0
    runners = []
    for c, s in ranked[1:6]:
        runners.append(
            {
                "id": c["id"],
                "name": c["name"],
                "series": c["series"],
                "confidence": round(max(s, 0.01) / total, 4),
                "blurb": c.get("blurb", ""),
                "image_url": c.get("image_url"),
                "source_url": c.get("source_url"),
            }
        )
    return {
        "mode": "fallback_keyword",
        "query": query,
        "winner": {
            "id": winner["id"],
            "name": winner["name"],
            "series": winner["series"],
            "confidence": round(max(raw, 0.01) / total, 4),
            "blurb": winner.get("blurb", ""),
            "tags": winner.get("tags", []),
            "image_url": winner.get("image_url"),
            "source_url": winner.get("source_url"),
        },
        "runners_up": runners,
        "notes": "Used keyword ranking over DuckDuckGo candidates (Laya skipped or unavailable).",
    }


def _laya_decide(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Build a compact state + choice question over shortlisted names
    state = {
        "user_preferences": query,
        "candidates": [
            {
                "id": c["id"],
                "name": c["name"],
                "series": c["series"],
                "profile": character_text(c),
            }
            for c in candidates
        ],
    }
    criteria = {
        c["id"]: f"{c['name']} from {c['series']}: {c.get('blurb', '')}; tags={', '.join(c.get('tags', [])[:8])}"
        for c in candidates
    }
    questions = {
        "best_match": {
            "type": "choice",
            "instructions": (
                "Which candidate best matches the user's preferred character features? "
                "Prefer tag and vibe alignment over fame."
            ),
            "criteria": criteria,
        },
        "fit_score": {
            "type": "score",
            "instructions": "How strongly does the best candidate match the preferences?",
            "criteria": ["weak match", "decent match", "excellent match"],
        },
    }

    # Shared, process-wide agent: loading the checkpoint costs ~7-10s, so it is
    # never rebuilt per request (the old code did exactly that).
    answers = laya_client.ask(state, questions)
    if not answers:
        return None
    best = answers.get("best_match") or {}
    choice = best.get("choice") or best.get("label")
    probs = best.get("probabilities") or {}
    conf = best.get("confidence")
    if conf is None and isinstance(probs, dict) and choice in probs:
        conf = probs[choice]
    if choice is None:
        return None

    by_id = {c["id"]: c for c in candidates}
    # map choice which might be id or name
    winner = by_id.get(choice)
    if winner is None:
        for c in candidates:
            if c["name"].lower() == str(choice).lower() or c["id"] == str(choice):
                winner = c
                break
    if winner is None:
        return None

    runners = []
    for cid, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        if cid == winner["id"]:
            continue
        c = by_id.get(cid)
        if not c:
            continue
        runners.append(
            {
                "id": c["id"],
                "name": c["name"],
                "series": c["series"],
                "confidence": round(float(p), 4),
                "blurb": c.get("blurb", ""),
                "image_url": c.get("image_url"),
                "source_url": c.get("source_url"),
            }
        )
        if len(runners) >= 5:
            break

    fit = answers.get("fit_score") or {}
    fit_val = fit.get("score")
    return {
        "mode": "laya",
        "query": query,
        "winner": {
            "id": winner["id"],
            "name": winner["name"],
            "series": winner["series"],
            "confidence": round(float(conf) if conf is not None else 0.0, 4),
            "blurb": winner.get("blurb", ""),
            "tags": winner.get("tags", []),
            "image_url": winner.get("image_url"),
            "source_url": winner.get("source_url"),
            "fit_score": fit_val,
        },
        "runners_up": runners,
        "notes": "Decision from Laya with calibrated choice probabilities.",
    }


def determine(
    query: str,
    top_k: int = 8,
    force_fallback: bool = False,
    online: bool | None = None,
    rounds: int = 3,
) -> dict[str, Any]:
    if os.getenv("WAIFU_FORCE_FALLBACK", "").lower() in {"1", "true", "yes"}:
        force_fallback = True
    if online is None:
        online = os.getenv("WAIFU_ONLINE_SEARCH", "1").lower() in {"1", "true", "yes"}
    # Empty catalog => always search online
    from .catalog import load_catalog
    if len(load_catalog()) == 0:
        online = True
    rounds = int(os.getenv("WAIFU_SEARCH_ROUNDS", str(rounds)) or rounds)
    pairs, search_meta = shortlist_with_online(query, top_k=top_k, online=online, rounds=rounds)
    candidates = [c for c, _ in pairs]
    if not candidates:
        return {
            "mode": "empty",
            "query": query,
            "winner": None,
            "runners_up": [],
            "notes": "No candidates.",
            "search": search_meta,
        }
    if not force_fallback:
        decided = _laya_decide(query, candidates)
        if decided is not None:
            decided["search"] = search_meta
            if search_meta.get("online_used"):
                decided["notes"] = ((decided.get("notes") or "") + " | DuckDuckGo multi-round shortlist.").strip(" |")
            return decided
        laya_miss = True
    else:
        laya_miss = False
    out = _fallback_decide(query, candidates)
    out["search"] = search_meta
    reasons = []
    if force_fallback:
        reasons.append("Force keyword fallback was on")
    elif laya_miss:
        reasons.append("Laya could not load/run in this container")
    if search_meta.get("online_used"):
        reasons.append("DuckDuckGo multi-round shortlist")
    if search_meta.get("online_error"):
        reasons.append(f"DDG error: {search_meta['online_error']}")
    if reasons:
        out["notes"] = " | ".join(reasons)
    return out
