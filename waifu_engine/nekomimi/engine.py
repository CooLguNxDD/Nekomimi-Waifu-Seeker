"""Nekomimi turn loop.

Laya never writes text; it decides. Per turn:

* **which question to ask is ours**, by expected information gain over the
  candidate posterior -- Laya's ``choice`` over question ids came back
  near-uniform and was dropped;
* one call for ``ready_to_guess`` (noul, is the evidence enough to name a
  character);
* one call **per candidate** for ``match`` (noul, does this candidate satisfy
  the question just answered) -- see ``_match_probabilities`` for why these are
  not batched into a shared state.

When Laya is unavailable the same decisions fall back to tag overlap and
entropy heuristics, so the loop still plays (less sharply) with no model.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any

from .. import sources, web_search
from ..catalog import load_catalog
from . import laya_client
from .session import (
    MAX_GUESSES,
    MAX_TURNS,
    Candidate,
    GuessSession,
    logit,
    new_session,
)
from .traits import (
    ANSWER_WEIGHT,
    QUESTION_BANK,
    QUESTIONS_BY_ID,
    make_dynamic,
    noul_criteria,
)

# Which candidate media each medium question accepts. Used to eliminate
# outright rather than down-weight once the user settles the medium.
_MEDIUM_QUESTIONS: dict[str, frozenset[str]] = {
    "medium_game": frozenset({"game"}),
    "medium_anime": frozenset({"anime", "manga"}),
    "medium_comic": frozenset({"comic"}),
}

# How many bank questions Laya chooses between each turn. Laya is weak on
# wide choice sets, so the bank is pre-filtered by entropy first.
CHOICE_WIDTH = int(os.getenv("WAIFU_NEKOMINI_CHOICE_WIDTH", "8"))
GUESS_CONFIDENCE = float(os.getenv("WAIFU_NEKOMINI_GUESS_CONFIDENCE", "0.80"))
MIN_QUESTIONS_BEFORE_GUESS = int(os.getenv("WAIFU_NEKOMINI_MIN_QUESTIONS", "5"))
# Laya saying "ready" is not enough on its own -- with a flat posterior it
# would guess a random candidate. Require a leader as well.
READY_MIN_POSTERIOR = float(os.getenv("WAIFU_NEKOMINI_READY_POSTERIOR", "0.45"))
REFRESH_EVERY = 3
REFRESH_WHEN_BELOW = 4
ONLINE = os.getenv("WAIFU_ONLINE_SEARCH", "1").lower() in {"1", "true", "yes"}

_SESSION_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _session_lock(sess: GuessSession) -> threading.Lock:
    with _LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(sess.id, threading.Lock())


# --- candidate sourcing ---------------------------------------------


def _medium_hint(sess: GuessSession) -> str | None:
    for a in sess.asked:
        if a.get("answer") != "yes":
            continue
        if a["qid"] == "medium_game":
            return "game"
        if a["qid"] == "medium_anime":
            return "anime"
        if a["qid"] == "medium_comic":
            return "comic"
    return None


def _search_terms(sess: GuessSession) -> list[str]:
    terms = list(sess.constraints)
    if sess.seed:
        terms.insert(0, sess.seed)
    return terms or ["popular anime game comic character"]


def seed_from_catalog(sess: GuessSession) -> int:
    """Start warm from the prebuilt catalog instead of a cold web search."""
    try:
        entries = load_catalog()
    except Exception as exc:  # noqa: BLE001 - a broken catalog must not end the round
        sess.notes.append(f"catalog unavailable: {exc}")
        return 0
    if not entries:
        return 0
    with _session_lock(sess):
        return sess.add_candidates(entries)


def refresh_candidates(sess: GuessSession, limit: int = 12, initial: bool = False) -> int:
    """Pull fresh candidates for the current constraints.

    Playwright HTML indexes first, then AniList and Wikipedia (structured, with
    real prose and a popularity number); DuckDuckGo fills remaining slots
    inside ``find_candidates``.
    """
    if not ONLINE:
        return 0
    exclude_names = {c.name for c in sess.candidates}
    raws: list[dict[str, Any]] = []
    try:
        raws = sources.find_candidates(
            _search_terms(sess),
            medium_hint=_medium_hint(sess),
            limit=limit * 2 if initial else limit,
            exclude_names=exclude_names,
        )
    except Exception as exc:  # noqa: BLE001 - search is best effort
        sess.notes.append(f"search failed: {exc}")
        if not raws:
            return 0
    with _session_lock(sess):
        added = sess.add_candidates(raws)
    return added


def _refresh_async(sess: GuessSession) -> None:
    threading.Thread(target=refresh_candidates, args=(sess,), daemon=True).start()


# --- question selection ---------------------------------------------


def _weighted(sess: GuessSession) -> list[tuple[Candidate, float]]:
    """Live candidates paired with their posterior probability."""
    return sess.posterior()


def _p_yes(question: dict[str, Any], candidates: Any) -> float:
    """Probability the answer is yes, weighted by how likely each candidate is.

    Weighting by posterior rather than counting heads is what makes this a real
    information-gain estimate: splitting off candidates nobody believes in
    teaches us nothing.
    """
    if not candidates:
        return float(question["prior"])
    pairs = (
        [(c, 1.0) for c in candidates]
        if isinstance(candidates[0], Candidate)
        else list(candidates)
    )
    true_tags = set(question["tags_true"])
    false_tags = set(question["tags_false"])
    yes = known = 0.0
    for cand, weight in pairs:
        tags = set(cand.tags)
        if true_tags & tags:
            known += weight
            yes += weight
        elif false_tags & tags:
            known += weight
    if known <= 0.0:
        return float(question["prior"])
    # Shrink toward the prior when the pool barely knows anything about this
    # trait, so a single tagged candidate cannot look like a perfect splitter.
    total = sum(w for _, w in pairs) or 1.0
    coverage = min(1.0, known / total)
    return coverage * (yes / known) + (1.0 - coverage) * float(question["prior"])


def _split_quality(question: dict[str, Any], candidates: Any) -> float:
    """Expected information gain in bits: 1.0 halves the pool, 0.0 says nothing."""
    p = _p_yes(question, candidates)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _dynamic_questions(sess: GuessSession) -> list[dict[str, Any]]:
    """Questions mined from the blurbs of the current candidate pool."""
    asked = sess.asked_ids()
    counts: dict[str, int] = {}
    for c in sess.alive_candidates():
        for slug in web_search.mine_trait_slugs(c.blurb):
            counts[slug] = counts.get(slug, 0) + 1
    out = []
    for slug, n in counts.items():
        if n < 2 or f"dyn_{slug}" in asked or slug in QUESTIONS_BY_ID:
            continue
        if any(slug in q["tags_true"] for q in QUESTION_BANK):
            continue
        out.append(make_dynamic(slug, slug.replace("-", " ")))
    return out


def candidate_questions(sess: GuessSession) -> list[dict[str, Any]]:
    """Top-N questions worth asking, highest expected information gain first."""
    asked = sess.asked_ids()
    settled = sess.settled_categories()
    weighted = _weighted(sess)
    pool = [
        q
        for q in QUESTION_BANK + _dynamic_questions(sess)
        if q["id"] not in asked and q["category"] not in settled
    ]
    if not pool:
        pool = [q for q in QUESTION_BANK if q["id"] not in asked]
    pool.sort(key=lambda q: _split_quality(q, weighted), reverse=True)
    return pool[:CHOICE_WIDTH]


def _laya_state(sess: GuessSession, candidates: list[Candidate]) -> dict[str, Any]:
    return {
        "goal": "Identify one anime, comic or video game character.",
        "confirmed_facts": sess.constraints or ["nothing confirmed yet"],
        "answer_history": sess.history() or [{"question": "none yet", "answer": "", "detail": ""}],
        "characters": {c.name: c.profile() for c in candidates},
        "questions_asked": sess.turn,
    }


def _pick_question(sess: GuessSession) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return (question, laya answers).

    Question choice is *ours*, not Laya's. Asking Laya to pick from a list of
    question ids came back near-uniform (measured ``confidence: 0.0003``) -- it
    was not really choosing, and the entropy ordering underneath was doing all
    the work. Expected information gain decides; Laya is spent where it is
    actually good, on judging candidates and on readiness.
    """
    options = candidate_questions(sess)
    if not options:
        return {}, None
    questions = {
        "ready_to_guess": {
            "type": "noul",
            "instructions": (
                "Given `confirmed_facts` and `answer_history`, is there now enough "
                "evidence to name one specific character with confidence?"
            ),
            "criteria": {
                "true": "yes, the facts so far point at one specific character",
                "false": "no, several different characters still fit the facts",
            },
        },
    }
    answers = laya_client.ask(_laya_state(sess, sess.scoring_pool()), questions)
    if answers:
        sess.laya_used = True
    return options[0], answers


# --- evidence ---------------------------------------------------------


def _tag_match(question: dict[str, Any], cand: Candidate) -> float:
    tags = set(cand.tags)
    if set(question["tags_true"]) & tags:
        return 0.85
    if set(question["tags_false"]) & tags:
        return 0.15
    blurb = f"{cand.name} {cand.series} {cand.blurb}".lower()
    if any(t.replace("-", " ") in blurb for t in question["tags_true"]):
        return 0.7
    return 0.5


def _match_probabilities(
    pool: list[Candidate], question: dict[str, Any]
) -> dict[str, float]:
    """P(question is true) for each candidate, one Laya call per candidate.

    One shared state with every candidate in it does not work: Laya attends to
    the whole state, so a batch of ``match_<i>`` questions comes back with
    near-identical probabilities (measured: 1.0 for all ten, anime and games
    alike). Scoring each candidate against its own state separates them cleanly
    -- and is faster, because each sequence is short.
    """
    probs: dict[str, float] = {}
    # Explicit true/false option text beats Laya's generic "yes, the statement
    # holds" -- the model is scoring against a restatement of the real claim.
    criteria = question.get("criteria") or noul_criteria(question["instructions"])
    for c in pool:
        answers = laya_client.ask(
            {"candidate": c.profile()},
            {
                "match": {
                    "type": "noul",
                    "instructions": question["instructions"],
                    "criteria": criteria,
                }
            },
        )
        got = (answers or {}).get("match") or {}
        if "noul" in got:
            probs[c.id] = float(got["noul"])
    return probs


def _eliminate_by_medium(sess: GuessSession, question: dict[str, Any], answer: str) -> int:
    """A settled medium is a hard fact, not a nudge.

    Once the user says "yes, it is a video game character", an anime candidate
    is not merely less likely -- it is wrong, and leaving it in the pool lets it
    soak up evidence from later questions.
    """
    if question["category"] != "medium" or question["id"] not in _MEDIUM_QUESTIONS:
        return 0
    medium = _MEDIUM_QUESTIONS[question["id"]]
    dropped = 0
    for cand in sess.alive_candidates():
        if cand.medium in ("", "unknown"):
            continue
        matches = cand.medium in medium
        if (answer == "yes" and not matches) or (answer == "no" and matches):
            cand.alive = False
            dropped += 1
    return dropped


def score_candidates(sess: GuessSession, question: dict[str, Any], answer: str) -> None:
    """Fold the answer into every live candidate's log-odds."""
    weight = ANSWER_WEIGHT.get(answer, 0.0)
    if weight == 0.0:
        return
    live = sess.alive_candidates()
    if not live:
        return

    # Laya judges the leaders; the rest are scored from their tags, so a
    # several-hundred-entry catalog costs the same per turn as ten candidates.
    pool = sess.scoring_pool()
    probs = _match_probabilities(pool, question)
    if probs:
        sess.laya_used = True
    for c in live:
        p = probs.get(c.id)
        if p is None:
            p = _tag_match(question, c)
        c.logodds += weight * logit(p)
        # Record what the pool now looks like so mined tags stay useful.
        if weight > 0 and p > 0.6:
            for tag in question["tags_true"]:
                if tag not in c.tags:
                    c.tags.append(tag)
    _eliminate_by_medium(sess, question, answer)
    sess.prune()


# --- public API -------------------------------------------------------


def _question_payload(sess: GuessSession, question: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": question["id"],
        "text": question["text"],
        "category": question["category"],
        "turn": sess.turn,
        "max_turns": MAX_TURNS,
    }


def _top_payload(sess: GuessSession, n: int = 3) -> list[dict[str, Any]]:
    return [c.public(p) for c, p in sess.posterior()[:n]]


def _guess_payload(sess: GuessSession) -> dict[str, Any]:
    ranked = sess.posterior()
    if not ranked:
        sess.stage = "done"
        return {
            "session_id": sess.id,
            "stage": "done",
            "guess": None,
            "message": "Out of candidates -- no character matches those answers.",
            "top": [],
        }
    cand, prob = ranked[0]
    sess.stage = "guessing"
    sess.pending_guess = cand.id
    sess.guesses_made += 1
    return {
        "session_id": sess.id,
        "stage": "guessing",
        "guess": cand.public(prob),
        "guess_number": sess.guesses_made,
        "turn": sess.turn,
        "top": _top_payload(sess),
        "laya": sess.laya_used,
    }


def _should_guess(sess: GuessSession, laya_answers: dict[str, Any] | None) -> bool:
    ranked = sess.posterior()
    if not ranked:
        return True
    if len(ranked) == 1 and sess.turn >= 3:
        return True
    top_p = ranked[0][1]
    if top_p >= GUESS_CONFIDENCE and sess.turn >= MIN_QUESTIONS_BEFORE_GUESS:
        return True
    if sess.turn >= MAX_TURNS:
        return True
    if laya_answers and sess.turn >= MIN_QUESTIONS_BEFORE_GUESS and top_p >= READY_MIN_POSTERIOR:
        ready = laya_answers.get("ready_to_guess") or {}
        act = (ready.get("action") or {}).get("act_probability", 0.0)
        if float(ready.get("noul", 0.0)) >= 0.75 and float(act) >= 0.6:
            return True
    return False


def _advance(sess: GuessSession) -> dict[str, Any]:
    """Emit the next question, or a guess when the evidence is strong enough."""
    if len(sess.alive_candidates()) < REFRESH_WHEN_BELOW:
        refresh_candidates(sess)
    question, laya_answers = _pick_question(sess)
    if _should_guess(sess, laya_answers) or not question:
        return _guess_payload(sess)
    sess.turn += 1
    sess.asked.append(
        {
            "qid": question["id"],
            "text": question["text"],
            "category": question["category"],
            "instructions": question["instructions"],
            "tags_true": question["tags_true"],
            "tags_false": question["tags_false"],
            "prior": question["prior"],
            "answer": None,
            "detail": None,
        }
    )
    sess.touch()
    return {
        "session_id": sess.id,
        "stage": "asking",
        "question": _question_payload(sess, question),
        "top": _top_payload(sess),
        "candidates_alive": len(sess.alive_candidates()),
        "laya": sess.laya_used,
    }


def start(seed: str = "") -> dict[str, Any]:
    sess = new_session(seed)
    if seed:
        sess.constraints.append(seed)
    seeded = seed_from_catalog(sess)
    if seeded:
        sess.notes.append(f"catalog seeded {seeded} candidates")
    refresh_candidates(sess, limit=16, initial=True)
    payload = _advance(sess)
    payload["seed"] = sess.seed
    return payload


def submit_answer(sess: GuessSession, answer: str, detail: str = "") -> dict[str, Any]:
    if sess.stage != "asking" or not sess.asked:
        return {"error": "no question is pending", "session_id": sess.id, "stage": sess.stage}
    answer = (answer or "").strip().lower()
    if answer not in ANSWER_WEIGHT:
        return {"error": f"answer must be one of {sorted(ANSWER_WEIGHT)}", "session_id": sess.id}

    current = sess.asked[-1]
    current["answer"] = answer
    current["detail"] = detail.strip() or None

    question = {
        "id": current["qid"],
        "text": current["text"],
        "category": current["category"],
        "instructions": current["instructions"],
        "tags_true": current["tags_true"],
        "tags_false": current["tags_false"],
        "prior": current["prior"],
    }

    if answer == "yes":
        sess.constraints.append(current["text"].replace("Is your character", "The character is")
                                .replace("Does your character", "The character does")
                                .replace("Has your character", "The character has")
                                .replace("Did your character", "The character did")
                                .rstrip("?"))
    elif answer == "no":
        sess.constraints.append("Not true: " + current["text"].rstrip("?"))
    if current["detail"]:
        sess.constraints.append(current["detail"])

    score_candidates(sess, question, answer)

    answered = sum(1 for a in sess.asked if a.get("answer"))
    if current["detail"] or answered % REFRESH_EVERY == 0:
        _refresh_async(sess)

    sess.touch()
    return _advance(sess)


def submit_guess_result(sess: GuessSession, correct: bool) -> dict[str, Any]:
    if sess.stage != "guessing" or not sess.pending_guess:
        return {"error": "no guess is pending", "session_id": sess.id, "stage": sess.stage}
    guessed = sess.by_id(sess.pending_guess)
    sess.pending_guess = None

    if correct:
        sess.stage = "done"
        sess.winner = guessed.id if guessed else None
        sess.touch()
        return {
            "session_id": sess.id,
            "stage": "done",
            "correct": True,
            "winner": guessed.public(1.0) if guessed else None,
            "turns": sess.turn,
            "laya": sess.laya_used,
        }

    if guessed:
        guessed.alive = False
        sess.rejected.add(guessed.id)
        sess.constraints.append(f"The character is not {guessed.name}")

    if sess.guesses_made >= MAX_GUESSES:
        sess.stage = "done"
        sess.touch()
        return {
            "session_id": sess.id,
            "stage": "done",
            "correct": False,
            "message": "Out of guesses. Start a new round with more detail.",
            "top": _top_payload(sess, 5),
            "turns": sess.turn,
        }

    sess.stage = "asking"
    refresh_candidates(sess)
    sess.touch()
    return _advance(sess)


def state_payload(sess: GuessSession) -> dict[str, Any]:
    return {
        "session_id": sess.id,
        "stage": sess.stage,
        "turn": sess.turn,
        "seed": sess.seed,
        "laya": sess.laya_used,
        "laya_available": laya_client.available(),
        "constraints": sess.constraints,
        "asked": [
            {"qid": a["qid"], "text": a["text"], "answer": a["answer"], "detail": a["detail"]}
            for a in sess.asked
        ],
        "candidates_alive": len(sess.alive_candidates()),
        "candidates_total": len(sess.candidates),
        "top": _top_payload(sess, 8),
        "notes": sess.notes,
    }
