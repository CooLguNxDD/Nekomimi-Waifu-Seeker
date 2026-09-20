from __future__ import annotations

import argparse
import json
import sys

from .decide import determine


def format_result(result: dict) -> str:
    lines = []
    mode = result.get("mode")
    lines.append(f"Mode: {mode}")
    lines.append(f"Query: {result.get('query')}")
    w = result.get("winner")
    if not w:
        lines.append("No winner.")
        return "\n".join(lines)
    lines.append("")
    lines.append(f"Winner: {w['name']} ({w['series']})")
    lines.append(f"Confidence: {w.get('confidence')}")
    if w.get("fit_score") is not None:
        lines.append(f"Fit score: {w.get('fit_score')}")
    lines.append(f"Why: {w.get('blurb')}")
    if w.get("image_url"):
        lines.append(f"Image: {w.get('image_url')}")
    if w.get("tags"):
        lines.append("Tags: " + ", ".join(w["tags"][:12]))
    runners = result.get("runners_up") or []
    if runners:
        lines.append("")
        lines.append("Runners-up:")
        for i, r in enumerate(runners, 1):
            lines.append(f"  {i}. {r['name']} ({r['series']}) — {r.get('confidence')}")
    if result.get("notes"):
        lines.append("")
        lines.append(result["notes"])
    search = result.get("search") or {}
    if search:
        lines.append("")
        lines.append(
            f"Search: online={search.get('online_used')} "
            f"backend={search.get('backend')} "
            f"state={search.get('search_state')} "
            f"hits={search.get('online_count')} "
            f"err={search.get('online_error')} "
            f"rounds={len(search.get('rounds') or [])}"
        )
        for log in (search.get("rounds") or []):
            lines.append(
                f"  r{log.get('round')}: q={log.get('query')!r} "
                f"hits={log.get('raw_hits')} new={log.get('new_candidates')} err={log.get('error')}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Best Waifu Determination Engine")
    p.add_argument("query", nargs="*", help="Desired character features")
    p.add_argument("--json", action="store_true", help="Print raw JSON")
    p.add_argument("--fallback", action="store_true", help="Force keyword fallback (skip Laya)")
    p.add_argument("--top-k", type=int, default=8, help="Shortlist size before decision")
    p.add_argument("--online", dest="online", action="store_true", default=None, help="Force online search")
    p.add_argument("--no-online", dest="online", action="store_false", help="Disable online search")
    p.add_argument("--rounds", type=int, default=3, help="Online search rounds (1-5)")
    args = p.parse_args(argv)
    query = " ".join(args.query).strip()
    if not query:
        query = input("Describe your ideal waifu features: ").strip()
    if not query:
        print("Empty query.", file=sys.stderr)
        return 2
    result = determine(query, top_k=args.top_k, force_fallback=args.fallback, online=args.online, rounds=args.rounds)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
