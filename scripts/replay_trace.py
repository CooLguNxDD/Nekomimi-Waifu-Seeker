"""Replay a bench trace's answers over real profiles and compare scoring settings.

A trace (``GET /api/nekomimi/trace/{sid}?target=...``) names the answers and
the wrong guesses, but not the profiles. This fetches each named character
from AniList, then Wikipedia, replays the bank questions in order with the
real scoring code, and prints the target's rank and its evidence gap to each
rival under two settings: the raw Laya noul (``SILENT_WEIGHT=1``, no choice
fallback) and the current calibration. Laya judgments are cached on the
session, so the second setting costs no model calls.

    python scripts/replay_trace.py trace.json --target "Tifa Lockhart"
    python scripts/replay_trace.py trace.json --target 2B --rivals "Firefly,Cate Archer"

Dynamic ``series*`` questions are skipped (the trace holds no options), and
so is ``medium``: it is a hard filter on source metadata, which differs
between this lookup and the live pool. Needs network; Laya is optional
(``--fallback`` uses the heuristics).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waifu_engine.names import same_character  # noqa: E402
from waifu_engine.nekomimi import engine  # noqa: E402
from waifu_engine.nekomimi import session as sess_mod  # noqa: E402
from waifu_engine.nekomimi.traits import QUESTIONS_BY_ID  # noqa: E402
from waifu_engine.sources import anilist, wikipedia  # noqa: E402

_SKIP = {"medium"}


def fetch(name: str) -> dict[str, Any] | None:
    """The first AniList, then Wikipedia, page whose name is ``name``."""
    for source in (anilist, wikipedia):
        try:
            hits = source.search_characters(name, limit=3)
        except Exception:  # noqa: BLE001 - a dead source just falls through
            continue
        for hit in hits:
            if same_character(hit.get("name") or "", name):
                return hit
    return None


def replay(rows: list[dict[str, Any]], answers: list[tuple[str, str]]) -> sess_mod.GuessSession:
    """A session holding ``rows`` with every answer scored in order."""
    sess = sess_mod.new_session()
    sess.add_candidates(rows)
    for qid, answer in answers:
        engine.score_candidates(sess, QUESTIONS_BY_ID[qid], answer)
    return sess


def report(sess: sess_mod.GuessSession, target_id: str, label: str) -> None:
    """Print the target's rank and its per-answer gap to every rival."""
    engine._rescore_candidates(sess)
    ranked = sess.posterior()
    print(f"\n[{label}] ranking: "
          + ", ".join(f"{c.name} {p:.3f}" for c, p in ranked))
    for cand, _p in ranked:
        if cand.id == target_id:
            continue
        gap = engine._evidence_gap(sess, sess.contrib, target_id, cand.id)
        total = sum(row["delta"] for row in gap)
        worst = ", ".join(f"{r['qid']}={r['answer']} {r['delta']:+.2f}" for r in gap[:3])
        print(f"  target vs {cand.name}: {total:+.2f}  (worst: {worst})")


def main() -> None:
    """Replay one trace under the raw and the calibrated settings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace")
    parser.add_argument("--target", required=True)
    parser.add_argument("--rivals", default="", help="comma list; default: the trace's guesses")
    parser.add_argument("--fallback", action="store_true", help="heuristics only, no Laya")
    args = parser.parse_args()
    if args.fallback:
        engine.laya_client.ask = lambda state, questions: None

    with open(args.trace, encoding="utf-8") as fh:
        trace = json.load(fh)
    rivals = [n.strip() for n in args.rivals.split(",") if n.strip()] or [
        g["name"] for g in trace.get("guesses") or [] if not same_character(g["name"], args.target)]
    answers = [(t["qid"], t["answer"]) for t in trace.get("turns") or []
               if t["qid"] in QUESTIONS_BY_ID and t["qid"] not in _SKIP]
    rows = []
    for name in [args.target, *rivals]:
        hit = fetch(name)
        if hit is None:
            print(f"(no page for {name}; skipped)")
            continue
        print(f"{hit['name']} [{hit.get('source')}] {hit.get('series')} "
              f"tags={hit.get('tags')} blurb={len(hit.get('blurb') or '')} chars")
        rows.append(hit)
    if not rows or not same_character(rows[0]["name"], args.target):
        sys.exit("target page not found")
    print(f"replaying {len(answers)} answers: " + " ".join(f"{q}={a}" for q, a in answers))

    sess = replay(rows, answers)
    target_id = next(c.id for c in sess.candidates if same_character(c.name, args.target))
    weight, choice_max = engine.SILENT_WEIGHT, engine.CHOICE_SILENT_MAX
    engine.SILENT_WEIGHT, engine.CHOICE_SILENT_MAX = 1.0, 0.0
    report(sess, target_id, "raw Laya")
    engine.SILENT_WEIGHT, engine.CHOICE_SILENT_MAX = weight, choice_max
    report(sess, target_id, f"calibrated (weight {weight}, choice max {choice_max})")


if __name__ == "__main__":
    main()
