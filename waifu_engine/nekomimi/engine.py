"""Nekomimi turn loop.

Laya never writes text; it decides. Per turn:

* **which question to ask is ours**, by expected information gain over the
  candidate posterior -- Laya's ``choice`` over question ids came back
  near-uniform and was dropped. "Where is your character from?" and
  "Which series?" compete in that ranking; neither is forced first;
* one call for ``ready_to_guess`` (noul, is the evidence enough to name a
  character);
* one call **per candidate** for ``match`` (noul, does this candidate satisfy
  the question just answered; or ``choice`` over the options of a
  multiple-choice question) -- see ``_match_probabilities`` for why these are
  not batched into a shared state;
* one ``choice`` call ranking which confirmed facts are most distinctive, so
  search queries lead with them rather than with "female" or "anime".

Search query *text* is never Laya's: templates build it, optionally with an
OpenAI-compatible LLM rewriting the player's own typed text (``query_llm``).
The LLM is slow and Laya is fast, so Laya gates it: the LLM is only asked when
there is free text it has not seen and a ``pool_fits`` noul says the current
candidates do not fit the facts. Even then it runs in the background and the
queries are used from a later search; a turn never waits for it.

When Laya is unavailable the same decisions fall back to tag overlap and
entropy heuristics, so the loop still plays (less sharply) with no model.
"""

from __future__ import annotations

import math
import os
import re
import threading
from typing import Any

from .. import query_llm, sources, timing, web_search
from ..names import (
    is_publisher_series,
    longer_namesake,
    same_series,
    same_series_key,
    series_key,
)
from . import laya_client
from .session import (
    MAX_GUESSES,
    MAX_TURNS,
    Candidate,
    GuessSession,
    new_session,
    popularity_prior,
)
from .traits import (
    ANSWER_WEIGHT,
    MAX_SERIES_OPTIONS,
    MEDIUM_ACCEPTS,
    MEDIUM_VALUES,
    QUESTION_BANK,
    QUESTIONS_BY_ID,
    appearance_question_ids,
    clue_likelihood,
    clue_question,
    is_choice,
    make_dynamic,
    noul_criteria,
    series_question,
    valid_answers,
    visual_search_phrases,
    yesno_visual_likelihood,
)

# The medium question's id. A known medium is a hard fact (``MEDIUM_ACCEPTS``):
# candidates of a known, non-accepted medium are removed outright. "other"
# is not: an empty accept-set means no hard filter.
MEDIUM_QID = "medium"
# Laya's framing of the task, shared by every state it sees.
GOAL = "Identify one anime, manga, comic, video game, movie or TV character."

# How many bank questions Laya chooses between each turn. Laya is weak on
# wide choice sets, so the bank is pre-filtered by entropy first.
CHOICE_WIDTH = int(os.getenv("WAIFU_NEKOMINI_CHOICE_WIDTH", "8"))
GUESS_CONFIDENCE = float(os.getenv("WAIFU_NEKOMINI_GUESS_CONFIDENCE", "0.80"))
MIN_QUESTIONS_BEFORE_GUESS = int(os.getenv("WAIFU_NEKOMINI_MIN_QUESTIONS", "5"))
# Laya saying "ready" is not enough on its own -- with a flat posterior it
# would guess a random candidate. Require a leader as well.
READY_MIN_POSTERIOR = float(os.getenv("WAIFU_NEKOMINI_READY_POSTERIOR", "0.45"))
# An empty seed fills the pool with famous characters, and a correct leader
# then sits around 0.50–0.70 for the whole round — under 0.80, so the turn
# cap arrived first (Mikasa). Guess earlier when that lead is stable and
# clearly ahead of the runner-up. 0.80 remains the single-check bar.
LEADER_POSTERIOR = float(os.getenv("WAIFU_NEKOMINI_LEADER_POSTERIOR", "0.50"))
LEADER_MARGIN = float(os.getenv("WAIFU_NEKOMINI_LEADER_MARGIN", "0.15"))
LEADER_STREAK = int(os.getenv("WAIFU_NEKOMINI_LEADER_STREAK", "2"))
# Log-odds added to the face of a shared series. One noisy trait used to tie
# Kylo with Vader and Gil with Homer; this stays smaller than a clear miss
# (about log(0.15/0.85) ≈ -1.7) so evidence can still overrule fame.
_SERIES_LEAD_BONUS = 0.9
_PROTAGONIST_LEAD_BONUS = 0.8
_SIDE_CHARACTER_PENALTY = 0.7
# A typed name ("mario") must not let "Mario Rossi" outrank the exact name
# once a medium or franchise is known.
_NAMESAKE_PENALTY = 2.2
# Whole-phrase aliases the player types on "Another series". Longer phrases
# are listed first so "super mario" wins over "mario".
_FRANCHISE_ALIASES: tuple[tuple[str, str], ...] = (
    ("cowboy bebop", "Cowboy Bebop"),
    ("star wars", "Star Wars"),
    ("super mario", "Super Mario"),
    ("spider-man", "Spider-Man"),
    ("spiderman", "Spider-Man"),
    ("spider man", "Spider-Man"),
    ("the simpsons", "The Simpsons"),
    ("simpsons", "The Simpsons"),
    ("mario", "Super Mario"),
)
# Aliases that are also a character's given name: they count only as the
# whole typed clue.
_EXACT_ALIASES = frozenset({"mario"})
ONLINE = os.getenv("WAIFU_ONLINE_SEARCH", "1").lower() in {"1", "true", "yes"}
# Facts offered to Laya when ranking which ones lead the search query, and how
# many of the winners go into the focused query.
FOCUS_OPTIONS = 8
FOCUS_TAKE = 3
# Broad facts that match thousands of characters; they go last in the fallback
# focus ranking.
_BROAD_CATEGORIES = frozenset({"medium", "gender", "meta"})
# Questions whose typed detail may be only colour words.
_HAIR_CATEGORIES = frozenset({"hair_color", "hair"})

_SESSION_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _session_lock(sess: GuessSession) -> threading.Lock:
    with _LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(sess.id, threading.Lock())


# --- candidate sourcing ---------------------------------------------


def _medium_hint(sess: GuessSession) -> str | None:
    """The medium the player picked (``anime``/``game``/``comic``/``movie``/``tv``),
    or None before the medium question is answered or after "Something else"."""
    for a in sess.asked:
        if a["qid"] == MEDIUM_QID and a.get("answer") in MEDIUM_ACCEPTS:
            return None if a["answer"] == "other" else a["answer"]
    return None


def _fact_entries(sess: GuessSession) -> list[tuple[str, float]]:
    """Positive search facts with a distinctiveness rank (lower = rarer).

    Search engines often ignore negation: "not female" retrieves female
    characters. Negative answers remain in the model evidence, not keywords.
    Free text from the player ranks first; broad categories rank last.
    """
    entries: dict[str, float] = {}

    def add(text: str | None, rank: float) -> None:
        """Record one search fact. ``+`` is flattened so it is not a query operator."""
        text = " ".join((text or "").replace("+", " ").split())
        if text and text not in entries:
            entries[text] = rank

    # The seed is always searched: "Aqua" or "Silver" can be a name.
    add(sess.seed, 0.0)
    add(_typed_franchise(sess), 0.0)
    for a in sess.asked:
        # A colour phrase typed on a hair question is a trait, not a name.
        # Searching "teal aqua turquoise" retrieved a page titled Turquoise.
        detail = a.get("detail")
        if not (a.get("category") in _HAIR_CATEGORIES
                and web_search.is_color_phrase(detail or "")):
            add(detail, 0.0)
        answer = a.get("answer")
        if a.get("kind") == "choice":
            option = (a.get("options") or {}).get(answer) or {}
            add(option.get("fact"), float(option.get("prior", 0.5)))
        elif answer == "yes":
            text = re.sub(r"^(?:Is|Does|Has|Did) your character\s+", "", a["text"])
            broad = a.get("category") in _BROAD_CATEGORIES
            add(text.rstrip("?"), 1.0 + float(a.get("prior", 0.5)) if broad
                else float(a.get("prior", 0.5)))
    return list(entries.items())


def _search_terms(sess: GuessSession) -> list[str]:
    return [text for text, _ in _fact_entries(sess)] or ["fictional character"]


def _focus_offer(entries: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Facts Laya may rank as the lead of the search query.

    The window used to be the first fact plus the newest ones. A series
    answered early then fell out, and the query became the hair colour
    without "Blue Archive". Rank 0 is the seed, a typed detail, or a series.
    """
    if len(entries) <= FOCUS_OPTIONS:
        return entries
    pinned = [entry for entry in entries if entry[1] <= 0.0]
    rest = [entry for entry in entries if entry[1] > 0.0]
    room = FOCUS_OPTIONS - len(pinned)
    if room <= 0:
        return pinned[:FOCUS_OPTIONS]
    rarest = sorted(rest, key=lambda entry: entry[1])
    chosen = list(dict.fromkeys([*pinned, *rarest[:room]]))
    for entry in rest[-room:]:
        if len(chosen) >= FOCUS_OPTIONS:
            break
        if entry not in chosen:
            chosen.append(entry)
    return chosen[:FOCUS_OPTIONS]


def _focus_facts(sess: GuessSession, entries: list[tuple[str, float]]) -> list[str]:
    """The few most distinctive facts, to lead the search query.

    Laya cannot write the query, but it can pick among the facts: one ``choice``
    call over up to eight of them. Its choice over question ids once came back
    near-uniform, so a flat answer is ignored in favour of rarity order.
    """
    if len(entries) <= FOCUS_TAKE:
        return []  # the template queries already carry every fact
    offered = _focus_offer(entries)
    keys = {f"fact_{i}": text for i, (text, _) in enumerate(offered, start=1)}
    answers = laya_client.ask(
        {"goal": GOAL,
         "confirmed_facts": list(keys.values())},
        {"focus": {
            "type": "choice",
            "instructions": "Which confirmed fact most narrows down which specific character this is?",
            "criteria": {k: text[:80] for k, text in keys.items()},
        }},
    )
    probs = ((answers or {}).get("focus") or {}).get("probabilities")
    if isinstance(probs, dict):
        clean = {}
        for k, v in probs.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if k in keys and math.isfinite(v) and 0.0 <= v <= 1.0:
                clean[k] = v
        if clean and max(clean.values()) >= 1.5 / len(keys):
            sess.laya_used = True
            ranked = sorted(clean, key=clean.get, reverse=True)
            return [keys[k] for k in ranked[:FOCUS_TAKE]]
    ranked_text = sorted(offered, key=lambda e: e[1])  # stable: ties keep answer order
    return [text for text, _ in ranked_text[:FOCUS_TAKE]]


def _free_text(sess: GuessSession) -> list[str]:
    """What the player typed: the seed and every detail. The only LLM input.

    Bank answers already have fixed search wording, so they never need the LLM.
    """
    texts = [sess.seed, *(a.get("detail") for a in sess.asked)]
    return [t.strip() for t in dict.fromkeys(texts) if t and t.strip()]


# Laya decides whether search needs help. Asked over the current leaders only.
POOL_FITS_CANDIDATES = 5
_POOL_FITS = {
    "pool_fits": {
        "type": "noul",
        "instructions": "Does any character in `characters` fit every one of `confirmed_facts`?",
        "criteria": {
            "true": "yes, at least one listed character fits all the confirmed facts",
            "false": "no, none of the listed characters fits all the confirmed facts",
        },
    }
}


def _search_stuck(sess: GuessSession) -> bool:
    """Does the pool look like it is missing the answer? One fast Laya call.

    Without Laya: stuck when nothing is found, or when no candidate leads
    after a few questions.
    """
    ranked = sess.posterior()
    if not ranked:
        return True
    answers = laya_client.ask(
        _laya_state(sess, sess.scoring_pool()[:POOL_FITS_CANDIDATES]), _POOL_FITS)
    got = (answers or {}).get("pool_fits") or {}
    try:
        p = float(got["noul"])
    except (KeyError, TypeError, ValueError):
        p = math.nan
    if math.isfinite(p) and 0.0 <= p <= 1.0:
        sess.laya_used = True
        return p < 0.5
    support = _leader_support(sess, ranked[0][0])
    if len(support) >= 2:
        return sum(support) / len(support) < 0.6
    return sess.turn >= 3 and ranked[0][1] < 0.3


def _llm_queries(sess: GuessSession, medium: str | None,
                 stuck: Any = None) -> list[str] | None:
    """Rewritten queries for this search, without ever waiting by default.

    Calls happen at most once per distinct set of typed text, and only when
    Laya says the pool is stuck. Until a result lands, the last good queries
    (one detail stale) are reused.
    """
    free = _free_text(sess)
    if not free or not query_llm.enabled():
        return sess.llm_queries or None
    ready = query_llm.peek(free, medium)
    if ready:
        sess.llm_queries = ready
        return ready
    stuck = stuck or (lambda: _search_stuck(sess))
    if not query_llm.known(free, medium) and stuck():
        query_llm.prefetch(free, medium)
        budget = query_llm.wait_seconds()
        if budget > 0:
            got = query_llm.wait(free, medium, timeout=budget)
            if got:
                sess.llm_queries = got
                return got
    return sess.llm_queries or None


def refresh_candidates(sess: GuessSession, limit: int = 12, initial: bool = False) -> int:
    with timing.span("search"):
        return _refresh_candidates(sess, limit, initial)


def _refresh_candidates(sess: GuessSession, limit: int, initial: bool) -> int:
    """Pull fresh candidates for the current constraints.

    Playwright HTML indexes first, then AniList and Wikipedia (structured, with
    real prose and a popularity number); DuckDuckGo fills remaining slots
    inside ``find_candidates``. Once a series is known, the lead query keeps
    that series next to the rare visual traits from the seed.
    """
    if not ONLINE:
        return 0
    exclude_names = {c.name for c in sess.candidates}
    raws: list[dict[str, Any]] = []
    memo: dict[str, bool] = {}

    def stuck() -> bool:
        # One Laya pool_fits call per search, shared by the LLM and DDG gates,
        # and only made if one of them actually needs the answer.
        if "v" not in memo:
            with timing.span("search.stuck"):
                memo["v"] = _search_stuck(sess)
        return memo["v"]

    try:
        entries = _fact_entries(sess)
        medium = _medium_hint(sess)
        with timing.span("search.focus"):
            focus = _focus_facts(sess, entries)
        anchor = _series_trait_anchor(sess)
        terms = _search_terms(sess)
        if anchor:
            # Lead with the series and the rare traits together. Focus is
            # empty when there are only a few facts, which is exactly when
            # the template would otherwise drop the series from later groups.
            terms = [anchor, *[term for term in terms if term != anchor]]
            focus = [anchor, *[fact for fact in focus if fact != anchor]][:FOCUS_TAKE]
        with timing.span("search.llm_gate"):
            # The first search still uses templates. Starting the prefetch
            # here means a trait seed is rewritten before the next turn
            # instead of waiting until the pool is already full of junk.
            rewritten = _llm_queries(sess, medium, stuck)
            if initial:
                rewritten = None
        with timing.span("search.fetch"):
            raws = sources.find_candidates(
                terms,
                medium_hint=medium,
                limit=limit * 2 if initial else limit,
                exclude_names=exclude_names,
                focus=focus or None,
                rewritten=rewritten or None,
                background_key=sess.id,
                ddg_gate=None if initial else stuck,
                # Only typed text (or an LLM rewrite of it) can match names;
                # broad button facts fall back to the popular pool.
                specific=bool(_free_text(sess) or rewritten or _confirmed_series(sess)),
                pool_size=len(sess.alive_candidates()),
                gemini_inline=_gemini_inline(sess),
            )
    except Exception as exc:  # noqa: BLE001 - search is best effort
        sess.notes.append(f"search failed: {exc}")
        if not raws:
            return 0
    with _session_lock(sess), timing.span("search.rescore_new"):
        added = sess.add_candidates(raws)
        if added and sess.evidence:
            _rescore_candidates(sess)
    timing.note(new=added)
    return added


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
    """Mutual information: answer entropy minus within-candidate uncertainty.

    A trait unknown for every candidate is a coin flip, not a useful split.
    """
    if not candidates:
        return 0.0
    pairs = ([(c, 1.0) for c in candidates]
             if isinstance(candidates[0], Candidate) else list(candidates))
    total = sum(w for _, w in pairs) or 1.0

    if is_choice(question):
        # Same quantity over n outcomes; a multiple-choice question can be
        # worth more than one bit, which is why it replaces a run of yes/nos.
        dists = [(_known_choice(question, c) or _tag_choice(question, c, strength=6.0), w / total)
                 for c, w in pairs]
        mix = {k: sum(d[k] * w for d, w in dists) for k in question["options"]}
        return max(0.0, _entropy_n(mix) - sum(w * _entropy_n(d) for d, w in dists))

    def entropy(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    predictions = [(_tag_match(question, c), w / total) for c, w in pairs]
    p_yes = sum(p * w for p, w in predictions)
    return max(0.0, entropy(p_yes) - sum(w * entropy(p) for p, w in predictions))


def _entropy_n(dist: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0.0)


def _bank_tags() -> set[str]:
    tags = set()
    for q in QUESTION_BANK:
        tags.update(q["tags_true"])
        for option in (q.get("options") or {}).values():
            tags.update(option["tags"])
    return tags


_BANK_TAGS = _bank_tags()


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
        if slug in _BANK_TAGS:
            continue
        out.append(make_dynamic(slug, slug.replace("-", " ")))
    return out


# Asked ahead of school/uniform/teen once the player has named them, or once
# the series is known and those shared answers no longer separate anyone.
_SERIES_APPEARANCE = ("hair_color", "look_halo", "look_wings", "look_horns")


def _pinned_question_ids(sess: GuessSession) -> list[str]:
    """Appearance questions the seed named, plus the rest after a series lock.

    Information gain never asked hair colour or halo: every Trinity student
    shares school, uniform and teen, and halo was buried in one horns/wings
    question. The traits the player already typed have to be offered early.
    """
    ids: list[str] = []
    for text in _free_text(sess):
        for qid in appearance_question_ids(text):
            if qid not in ids:
                ids.append(qid)
    if _confirmed_series(sess):
        for qid in _SERIES_APPEARANCE:
            if qid not in ids:
                ids.append(qid)
    return ids


def _gemini_inline(sess: GuessSession) -> bool:
    """Whether this search should wait for Gemini.

    Inline while nobody alive shows the visual combination in the seed.
    A junk page stored from Wikipedia must not count as that somebody, or
    Mika only arrives on the next turn as a background hit.
    """
    clue = " ".join(dict.fromkeys(
        phrase for text in _free_text(sess) for phrase in visual_search_phrases(text)))
    if len(visual_search_phrases(clue)) < 2:
        return False
    return not any(
        (score := clue_likelihood(clue, c.blurb, c.tags)) is not None and score >= 0.9
        for c in sess.alive_candidates()
    )


def candidate_questions(sess: GuessSession) -> list[dict[str, Any]]:
    """Top-N questions worth asking, highest expected information gain first.

    Medium ("Where is your character from?") and series ("Which series?")
    compete in that ranking like any other question. Forcing medium first
    burned a turn when the pool already shared one medium. Traits the seed
    already named are pulled in front of that ranking: otherwise hair colour,
    halo and wings lose to questions the whole school answers the same way.
    """
    asked = sess.asked_ids()
    settled = sess.settled_categories()
    weighted = _weighted(sess)
    series = _series_question(sess)
    pool = [
        q
        for q in QUESTION_BANK + _dynamic_questions(sess) + ([series] if series else [])
        if q["id"] not in asked and q["category"] not in settled
    ]
    if not pool:
        pool = [q for q in QUESTION_BANK if q["id"] not in asked]
    pool.sort(key=lambda q: _split_quality(q, weighted), reverse=True)
    order = _pinned_question_ids(sess)
    pin_ids = set(order)
    pins = [q for q in pool if q["id"] in pin_ids]
    # Keep the seed's own order (hair, halo, wings), not the info-gain order.
    pins.sort(key=lambda q: order.index(q["id"]))
    rest = [q for q in pool if q["id"] not in pin_ids]
    return (pins + rest)[:CHOICE_WIDTH]


def _laya_state(sess: GuessSession, candidates: list[Candidate]) -> dict[str, Any]:
    """Round-level state for Laya: goal, confirmed facts, history, given candidates."""
    return {
        "goal": GOAL,
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
    with timing.span("pick.rank"):
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


def _profile_likelihood(question: dict[str, Any], cand: Candidate) -> float | None:
    """P(the question holds) from appearance text, or None when it is silent.

    Flat noul scores left Mika tied with other Trinity students: their blurbs
    share "pink", "halo" and, for a wing-shaped halo, "wings". A clear hit or
    miss on that combination has to outrank the fame prior.
    """
    if question.get("clues"):
        return clue_likelihood(question["clues"], cand.blurb, cand.tags)
    if question.get("kind", "yesno") == "yesno":
        return yesno_visual_likelihood(question.get("tags_true"), cand.blurb, cand.tags)
    return None


def _tag_match(question: dict[str, Any], cand: Candidate) -> float:
    """Heuristic P(yes). Halo, wings and horns use the appearance text.

    Information gain reads this. The capped fallback in rescoring does not,
    or a sharp visual split would be flattened back to [0.4, 0.6].
    """
    visual = _profile_likelihood(question, cand)
    if visual is not None and not question.get("clues"):
        return visual
    tags = set(cand.tags)
    if set(question["tags_true"]) & tags:
        return 0.85
    if set(question["tags_false"]) & tags:
        return 0.15
    blurb = f"{cand.name} {cand.series} {cand.blurb}".lower()
    if any(re.search(r"\b" + re.escape(t.replace("-", " ")) + r"\b", blurb)
           for t in question["tags_true"]):
        return 0.7
    return 0.5


def _is_series_question(question: dict[str, Any]) -> bool:
    """Whether ``question`` is a "Which series?" question (its options carry keys)."""
    return any("series_key" in o for o in (question.get("options") or {}).values())


def _known_choice(question: dict[str, Any], cand: Candidate) -> dict[str, float] | None:
    """Option distribution fixed by data the candidate already carries, or None.

    Medium question: a known medium puts most of the mass on its own option,
    a little on crossover options (movie/TV). Series question: a known series puts 0.9 on its option, or on
    "Another series" when it is not listed. Either way no model call is
    needed; candidates without the data go to Laya instead.
    """
    options = list(question.get("options") or {})
    if not options:
        return None
    if question.get("id") == MEDIUM_QID:
        if cand.medium not in MEDIUM_VALUES:
            return None
        # The candidate's own medium gets most of the mass; an option that only
        # accepts it as a crossover (movie <-> TV) gets a smaller share, so
        # "Movie" still ranks film characters above TV ones.
        own = "anime" if cand.medium == "manga" else cand.medium
        cross = [k for k in options
                 if k != own and cand.medium in MEDIUM_ACCEPTS.get(k, ())]
        rest = [k for k in options if k != own and k not in cross]
        dist = {own: 0.8 if cross else 0.96}
        dist.update({k: 0.16 / len(cross) for k in cross})
        dist.update({k: 0.04 / len(rest) for k in rest})
        return dist
    if _is_series_question(question):
        key = series_key(cand.series)
        if not key:
            return None
        hits = [k for k, o in question["options"].items()
                if same_series_key(key, o.get("series_key", ""))][:1] or ["other"]
    else:
        return None
    rest = [k for k in options if k not in hits]
    on, off = 0.9, 0.1
    if not rest:
        return {k: 1.0 / len(hits) for k in hits}
    dist = {k: on / len(hits) for k in hits}
    dist.update({k: off / len(rest) for k in rest})
    return dist


_SERIES_LABEL = re.compile(r"[^\w\s:'&.!/-]")


def _series_label(series: str) -> str:
    """A series name safe to show as a button: scraped text, so only letters,
    digits, spaces and ``:'&.!-/`` survive, capped at 40 characters."""
    label = " ".join(_SERIES_LABEL.sub("", series or "").split())
    return label[:40].rstrip()


def _typed_franchise(sess: GuessSession) -> str:
    """Canonical series named in the player's own text, or "".

    "Another series" plus a detail never sets ``_confirmed_series``, so
    "cowboy bebop" / "star wars" / "simpsons" used to fall out of the lead
    query. A whole-phrase alias keeps that work in every later search.
    """
    texts = [" ".join(t.lower().split()) for t in _free_text(sess)]
    for marker, label in _FRANCHISE_ALIASES:
        for text in texts:
            # "mario" is also the first word of "Mario Rossi"; only the whole
            # clue names the franchise, or Rossi's own search is penalised.
            if marker in _EXACT_ALIASES:
                if text == marker:
                    return label
            elif f" {marker} " in f" {text} ":
                return label
    return ""


def _series_trait_anchor(sess: GuessSession) -> str:
    """Series plus the rare visual traits the player already stated, or "".

    Template groups keep the first fact and the newest three. After several
    answers the series sits in the middle and drops out, so a later search
    for "pink hair" returns every pink-haired student. Pinning both in one
    lead string is what brings Mika's page back instead of the whole school.
    A typed franchise counts: the player never clicked a series chip.
    """
    series = _confirmed_series(sess) or _typed_franchise(sess)
    if not series:
        return ""
    phrases: list[str] = []
    for text in (sess.seed, *(a.get("detail") for a in sess.asked)):
        phrases.extend(visual_search_phrases(text or ""))
    for text, _rank in _fact_entries(sess):
        phrases.extend(visual_search_phrases(text))
    phrases = list(dict.fromkeys(phrases))
    if not phrases:
        # A typed or picked series with no visual combo still has to lead.
        # Otherwise "cowboy bebop" is just another fact and later drops out.
        return series[:180]
    return f"{series} {' '.join(phrases)}"[:180]


def _confirmed_series(sess: GuessSession) -> str:
    """The series the player picked, or a known franchise they typed instead.

    "Another series" plus the detail ``vocaloid`` is a confirmation. Leaving
    it as free text asked ``series_2`` over junk labels and never offered
    Vocaloid.
    """
    for a in sess.asked:
        option = (a.get("options") or {}).get(a.get("answer") or "")
        if option and option.get("series_key"):
            return option.get("fact", "")
        if str(a.get("qid") or "").startswith("series") and a.get("answer") == "other":
            label = web_search.franchise_label(a.get("detail") or "")
            if label:
                return label
    return ""


def _series_question(sess: GuessSession) -> dict[str, Any] | None:
    """A "Which series?" question over the leading candidates' series, or None.

    Live candidates are grouped by series (``names.series_key``), weighted by
    posterior, and the top groups become options. Asked at most twice: the
    second time only after "Another series", without the series already
    offered. Needs at least two series to be worth asking.
    """
    series_asked = [a for a in sess.asked if a["qid"].startswith("series")]
    if _confirmed_series(sess) or len(series_asked) >= 2:
        return None
    if series_asked and series_asked[-1].get("answer") != "other":
        return None
    offered = [o["series_key"] for a in series_asked
               for o in (a.get("options") or {}).values() if o.get("series_key")]
    groups: list[dict[str, Any]] = []
    for cand, p in sess.posterior():
        key = series_key(cand.series)
        label = _series_label(cand.series)
        # A publisher bucket is not a series chip. Offering "Marvel Comics"
        # locked Abomination to every other Marvel page.
        if (not key or not label or is_publisher_series(cand.series)
                or any(same_series_key(key, k) for k in offered)):
            continue
        group = next((g for g in groups if same_series_key(key, g["key"])), None)
        if group is None:
            group = {"key": key, "weight": 0.0, "labels": {}}
            groups.append(group)
        group["weight"] += p
        group["labels"][label] = group["labels"].get(label, 0) + 1
    typed = web_search.franchise_label(" ".join(_free_text(sess)))
    if typed:
        key = series_key(typed)
        if key and not any(same_series_key(key, g["key"]) for g in groups) \
                and not any(same_series_key(key, k) for k in offered):
            groups.insert(0, {"key": key, "weight": 0.0, "labels": {typed: 1}})
    if len(groups) < 2:
        return None
    groups.sort(key=lambda g: g["weight"], reverse=True)
    picked = [(g["key"], max(g["labels"], key=g["labels"].get))
              for g in groups[:MAX_SERIES_OPTIONS]]
    if typed:
        key = series_key(typed)
        if key and not any(same_series_key(key, k) for k, _label in picked):
            picked = [(key, typed), *picked[: MAX_SERIES_OPTIONS - 1]]
    return series_question("series" if not series_asked else "series_2", picked)


def _tag_choice(question: dict[str, Any], cand: Candidate,
                strength: float = 1.5) -> dict[str, float]:
    """Heuristic option distribution for a multiple-choice question.

    Uniform, with options the candidate's tags (or a whole-phrase blurb hit)
    point at weighted ``strength`` times the rest. The default 1.5 mirrors the
    [0.4, 0.6] cap on yes/no fallbacks: noisy tags stay weak evidence.
    """
    tags = set(cand.tags)
    blurb = f"{cand.name} {cand.series} {cand.blurb}".lower()
    weights = {}
    for key, option in question["options"].items():
        hit = bool(set(option["tags"]) & tags) or bool(
            option["fact"] and re.search(r"\b" + re.escape(option["fact"].lower()) + r"\b", blurb))
        weights[key] = strength if hit else 1.0
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def _choice_probabilities(
    pool: list[Candidate], question: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Option probabilities per candidate, one Laya ``choice`` call each.

    Same reasoning as ``_match_probabilities``: a shared state blurs the
    candidates together. Answers that do not cover every option, or carry
    non-probabilities, are dropped so the heuristic takes over.
    """
    out: dict[str, dict[str, float]] = {}
    keys = list(question["options"])
    for c in pool:
        answers = laya_client.ask(
            {"candidate": c.profile()},
            {
                "match": {
                    "type": "choice",
                    "instructions": question["instructions"],
                    "criteria": question["criteria"],
                }
            },
        )
        got = ((answers or {}).get("match") or {}).get("probabilities")
        if not isinstance(got, dict):
            continue
        try:
            dist = {k: float(got[k]) for k in keys}
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in dist.values()):
            continue
        total = sum(dist.values())
        if total <= 0.0:
            continue
        out[c.id] = {k: p / total for k, p in dist.items()}
    return out


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
        state = {"candidate": c.profile()}
        if question.get("clues"):
            state["clues"] = question["clues"]
        answers = laya_client.ask(
            state,
            {
                "match": {
                    "type": "noul",
                    "instructions": question["instructions"],
                    "criteria": criteria,
                }
            },
        )
        got = (answers or {}).get("match") or {}
        try:
            p = float(got["noul"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(p) and 0.0 <= p <= 1.0:
            probs[c.id] = p
    return probs


def _eliminate_by_medium(sess: GuessSession, question: dict[str, Any], answer: str) -> int:
    """Drop candidates whose known medium contradicts a settled pick.

    Once the player says "it's from a video game", an anime candidate
    is not merely less likely -- it is wrong, and leaving it in the pool lets it
    soak up evidence from later questions. "Something else" is not that kind
    of fact: its accept-set is empty, and treating empty as "accept nothing"
    deleted every anime, game, comic, movie and TV row. Search then ran with
    no medium hint and the pool drifted. Unknown media are never removed.
    """
    if question["id"] != MEDIUM_QID or answer == "other":
        return 0
    accepts = MEDIUM_ACCEPTS.get(answer)
    if not accepts:
        return 0
    dropped = 0
    for cand in sess.alive_candidates():
        if cand.medium not in MEDIUM_VALUES:
            continue  # unknown medium: never eliminated
        if cand.medium not in accepts:
            cand.alive = False
            cand.logodds = -math.inf
            dropped += 1
    return dropped


def score_candidates(sess: GuessSession, question: dict[str, Any], answer: str) -> None:
    """Evaluate every eligible identity; posterior rank never gates model access."""
    if is_choice(question):
        if answer not in question["options"]:
            return
    elif not ANSWER_WEIGHT.get(answer, 0.0):
        return
    with _session_lock(sess), timing.span("score"):
        sess.evidence[question["id"]] = (dict(question), answer)
        _rescore_candidates(sess)


def _rescore_candidates(sess: GuessSession) -> None:
    """Replay evidence for new arrivals, reusing successful model judgments.

    The caller holds the session lock. Only known medium contradictions and
    rejected guesses remove identities; uncertain evidence must be recoverable.
    A visual clue the profile clearly misses is a strong down-rank, not a
    removal, so a thin blurb can still recover on a later answer.
    """
    for question, answer in sess.evidence.values():
        _eliminate_by_medium(sess, question, answer)
    live = sess.alive_candidates()
    if not live:
        return
    for c in live:
        c.logodds = popularity_prior(c.popularity)
    for qid, (question, answer) in sess.evidence.items():
        if is_choice(question):
            # Known medium/series is data, not a judgment: no model call.
            known = {c.id: d for c in live if (d := _known_choice(question, c))}
            missing = [c for c in live
                       if c.id not in known and (c.id, qid) not in sess.choice_cache]
            dists = _choice_probabilities(missing, question)
            if dists:
                sess.laya_used = True
                sess.choice_cache.update({(cid, qid): d for cid, d in dists.items()})
            for c in live:
                dist = (known.get(c.id) or sess.choice_cache.get((c.id, qid))
                        or _tag_choice(question, c))
                c.logodds += math.log(min(0.98, max(0.02, dist.get(answer, 0.0))))
            continue
        missing = [c for c in live if (c.id, qid) not in sess.match_cache]
        probs = _match_probabilities(missing, question)
        if probs:
            sess.laya_used = True
            sess.match_cache.update({(cid, qid): p for cid, p in probs.items()})
        for c in live:
            # Appearance text wins over a mushy noul for a stated visual
            # combination. The cache still holds the model score so the
            # guess gate can see that a judgment happened.
            visual = _profile_likelihood(question, c)
            p = sess.match_cache.get((c.id, qid))
            if visual is not None:
                p = visual
            elif p is None:
                # A failed model call must not make noisy tags stronger evidence
                # than the model's typically modest confidence.
                p = min(0.6, max(0.4, _tag_match(question, c)))
            likelihood = p if answer == "yes" else 1.0 - p
            c.logodds += math.log(min(0.98, max(0.02, likelihood)))
    _apply_identity_priors(sess, live)


def _protagonist_answer(sess: GuessSession) -> str:
    """The player's answer to the protagonist question, or "" if unasked."""
    for question, answer in sess.evidence.values():
        if question.get("id") == "role_protagonist":
            return answer or ""
    return ""


def _is_series_lead(group: list[Candidate]) -> list[Candidate]:
    """The face of one series: a protagonist tag, else a much more famous row.

    Side characters share the series string with the lead. Without this, one
    mushy trait lets Gil or Kylo tie Homer or Vader.
    """
    tagged = [c for c in group if {"protagonist", "mascot"} & set(c.tags)]
    if tagged:
        return tagged
    ranked = sorted(group, key=lambda c: c.popularity, reverse=True)
    top = ranked[0].popularity
    nxt = ranked[1].popularity if len(ranked) > 1 else 0
    if top >= max(1000, 3 * max(nxt, 1)):
        return [ranked[0]]
    return []


def _apply_identity_priors(sess: GuessSession, live: list[Candidate]) -> None:
    """Prefer a franchise lead, and an exact short name over a longer namesake.

    Called at the end of a rescore, after trait evidence. A clear trait miss
    is still larger than these bonuses. Publisher buckets are not a series,
    so Marvel characters are not all treated as one cast.
    """
    franchise = _confirmed_series(sess) or _typed_franchise(sess)
    if franchise:
        for cand in live:
            if is_publisher_series(cand.series):
                continue
            blob = f"{cand.name} {cand.series} {cand.blurb}".lower()
            if same_series(cand.series, franchise) or franchise.lower() in blob:
                cand.logodds += _SERIES_LEAD_BONUS
            elif series_key(cand.series):
                cand.logodds -= _SIDE_CHARACTER_PENALTY
    focus = series_key(franchise)
    groups: dict[str, list[Candidate]] = {}
    for cand in live:
        if not focus or is_publisher_series(cand.series):
            continue
        key = series_key(cand.series)
        blob = f"{cand.name} {cand.series} {cand.blurb}".lower()
        # A missing series field still counts when the page names the work.
        # Spike's row was "Unknown" while Ed's said Cowboy Bebop, so the
        # series key alone never put them in one cast.
        if not ((key and same_series_key(key, focus)) or franchise.lower() in blob):
            continue
        groups.setdefault(focus, []).append(cand)
    protag = _protagonist_answer(sess)
    for group in groups.values():
        # "Not the protagonist" must not hand the face of the series a bonus
        # that outweighs the player's own answer.
        if len(group) < 2 or protag == "no":
            continue
        leads = set(id(c) for c in _is_series_lead(group))
        if not leads:
            continue
        for cand in group:
            if id(cand) in leads:
                cand.logodds += _SERIES_LEAD_BONUS
                if protag == "yes":
                    cand.logodds += _PROTAGONIST_LEAD_BONUS
            elif protag == "yes":
                cand.logodds -= _SIDE_CHARACTER_PENALTY
    for cand in live:
        if "/" not in cand.name:
            continue
        low = cand.name.lower()
        for other in live:
            if other is cand:
                continue
            other_name = other.name.lower().strip()
            if len(other_name) < 4:
                continue
            if re.search(r"\b" + re.escape(other_name) + r"\b", low):
                cand.logodds -= _NAMESAKE_PENALTY
                break
    typed = [t.strip().lower() for t in _free_text(sess)]
    if not typed or not (franchise or _medium_hint(sess)):
        return
    exact = [c for c in live if c.name.strip().lower() in typed]
    if not exact:
        return
    for cand in live:
        if any(longer_namesake(short.name, cand.name) for short in exact):
            if any(same_series(short.series, cand.series) for short in exact):
                continue
            cand.logodds -= _NAMESAKE_PENALTY


# --- public API -------------------------------------------------------


def _question_payload(sess: GuessSession, question: dict[str, Any]) -> dict[str, Any]:
    out = {
        "qid": question["id"],
        "text": question["text"],
        "category": question["category"],
        "kind": question.get("kind", "yesno"),
        "turn": sess.turn,
        "max_turns": MAX_TURNS,
    }
    if is_choice(question):
        out["options"] = [{"key": k, "label": o["label"]} for k, o in question["options"].items()]
    return out


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
            "message": "Search did not find a matching character. Try another round with a specific clue.",
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


def _leader_support(sess: GuessSession, leader: Candidate) -> list[float]:
    """How well each model judgment of ``leader`` agrees with the answers given."""
    support = []
    for qid, (question, answer) in sess.evidence.items():
        if is_choice(question):
            # Scale-free support: 0.5 when the pick ties the candidate's best
            # other option, so the same 0.6 bar applies as for yes/no.
            dist = sess.choice_cache.get((leader.id, qid))
            if dist:
                p = dist.get(answer, 0.0)
                other = max((v for k, v in dist.items() if k != answer), default=0.0)
                support.append(p / (p + other) if p + other > 0.0 else 0.5)
            continue
        p = sess.match_cache.get((leader.id, qid))
        if p is not None:
            support.append(p if answer == "yes" else 1.0 - p)
    return support


def _note_leader(sess: GuessSession, leader_id: str, qualifies: bool) -> None:
    """Count consecutive checks where ``leader_id`` clears the stable-guess bar.

    The streak used to grow whenever someone merely topped the pool. Weak
    checks then filled ``LEADER_STREAK``, and the first later spike above
    ``LEADER_POSTERIOR`` and ``LEADER_MARGIN`` guessed immediately. It grows
    only while both thresholds hold, and resets when either fails.
    """
    if qualifies and sess.leader_id == leader_id:
        sess.leader_streak += 1
        return
    sess.leader_id = leader_id
    sess.leader_streak = 1 if qualifies else 0


def _should_guess(sess: GuessSession, laya_answers: dict[str, Any] | None) -> bool:
    """Whether the evidence is strong enough to name a candidate.

    Early guesses wait for ``MIN_QUESTIONS_BEFORE_GUESS`` and for at least
    two model judgments averaging 0.6 — a lone search hit is posterior 1 and
    must not guess. Past that bar, guess when the posterior clears
    ``GUESS_CONFIDENCE``, or when the same candidate has led for
    ``LEADER_STREAK`` checks at ``LEADER_POSTERIOR`` and ``LEADER_MARGIN``
    ahead of the runner-up, or when Laya's ready_to_guess is confident and
    the posterior is at least ``READY_MIN_POSTERIOR``. The turn cap guesses
    even without that support.
    """
    ranked = sess.posterior()
    if not ranked:
        return sess.turn >= MAX_TURNS
    leader, top_p = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    # Count this check toward the stable-leader streak only when the lead is
    # already wide enough to guess. A string of narrow tops must not pre-fill it.
    _note_leader(
        sess, leader.id,
        top_p >= LEADER_POSTERIOR and top_p - second >= LEADER_MARGIN,
    )
    if sess.turn >= MAX_TURNS:
        return True
    # The posterior is conditional on what search happened to find. A lone hit
    # has probability 1 even when it contradicts every clue. Require independent
    # model support before turning relative rank into an early guess.
    support = _leader_support(sess, leader)
    if len(support) < 2 or sum(support) / len(support) < 0.6:
        return False
    if sess.turn < MIN_QUESTIONS_BEFORE_GUESS:
        return False
    if top_p >= GUESS_CONFIDENCE:
        return True
    if (top_p >= LEADER_POSTERIOR and top_p - second >= LEADER_MARGIN
            and sess.leader_streak >= max(1, LEADER_STREAK)):
        return True
    if laya_answers and top_p >= READY_MIN_POSTERIOR:
        ready = laya_answers.get("ready_to_guess") or {}
        act = (ready.get("action") or {}).get("act_probability", 0.0)
        if float(ready.get("noul", 0.0)) >= 0.75 and float(act) >= 0.6:
            return True
    return False


def _advance(sess: GuessSession) -> dict[str, Any]:
    """Emit the next question, or a guess when the evidence is strong enough."""
    with timing.span("pick"):
        question, laya_answers = _pick_question(sess)
    alive = len(sess.alive_candidates())
    timing.note(turn=sess.turn, cands=f"{alive}/{len(sess.candidates)}")
    if _should_guess(sess, laya_answers) or not question:
        timing.note(guess=True)
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
            "kind": question.get("kind", "yesno"),
            "criteria": question.get("criteria"),
            "options": question.get("options"),
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


@timing.traced("start")
def start(seed: str = "") -> dict[str, Any]:
    sess = new_session(seed)
    if seed:
        sess.constraints.append(seed)
        score_candidates(sess, clue_question("clue_seed", seed), "yes")
    refresh_candidates(sess, limit=16, initial=True)
    payload = _advance(sess)
    payload["seed"] = sess.seed
    return payload


@timing.traced("answer")
def submit_answer(sess: GuessSession, answer: str, detail: str = "") -> dict[str, Any]:
    if sess.stage != "asking" or not sess.asked:
        return {"error": "no question is pending", "session_id": sess.id, "stage": sess.stage}
    current = sess.asked[-1]
    answer = (answer or "").strip().lower()
    allowed = valid_answers(current)
    if answer not in allowed:
        return {"error": f"answer must be one of {sorted(allowed)}", "session_id": sess.id}

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
        "kind": current.get("kind", "yesno"),
    }
    if current.get("criteria"):
        question["criteria"] = current["criteria"]
    if current.get("options"):
        question["options"] = current["options"]

    if is_choice(question) and answer in question["options"]:
        label = question["options"][answer]["label"]
        sess.constraints.append(f"{current['text'].rstrip('?')}: {label}")
    elif answer == "yes":
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
    if current["detail"]:
        score_candidates(sess, clue_question(f"clue_{sess.turn}", current["detail"]), "yes")

    # Every answer opens a new search branch. Discover and replay evidence
    # before choosing the next question or declaring a winner.
    refresh_candidates(sess)

    sess.touch()
    return _advance(sess)


@timing.traced("guess_result")
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
    """Return a round a refreshed page can render.

    The turn responses already include the pending question or guess. This
    snapshot used to omit them, so a follow-up GET replaced that payload and
    the UI lost the question it was showing.
    """
    payload: dict[str, Any] = {
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
    if sess.stage == "asking" and sess.asked:
        current = sess.asked[-1]
        question = {
            "id": current["qid"],
            "text": current["text"],
            "category": current["category"],
            "kind": current.get("kind", "yesno"),
            "options": current.get("options"),
        }
        payload["question"] = _question_payload(sess, question)
    elif sess.stage == "guessing" and sess.pending_guess:
        cand = sess.by_id(sess.pending_guess)
        if cand is not None:
            prob = next((p for c, p in sess.posterior() if c.id == cand.id), 0.0)
            payload["guess"] = cand.public(prob)
            payload["guess_number"] = sess.guesses_made
    elif sess.stage == "done" and sess.winner:
        cand = sess.by_id(sess.winner)
        if cand is not None:
            payload["winner"] = cand.public(1.0)
            payload["correct"] = True
            payload["turns"] = sess.turn
    return payload
