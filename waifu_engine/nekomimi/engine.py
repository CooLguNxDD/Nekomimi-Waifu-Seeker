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

import json
import math
import os
import re
import threading
from typing import Any

from .. import query_llm, sources, timing, web_search
from ..names import (
    canonical_name,
    identity_id,
    is_publisher_series,
    longer_namesake,
    same_character,
    same_series,
    same_series_key,
    series_key,
)
from . import laya_client
from .lexicon import BROAD_CATEGORIES as _BROAD_CATEGORIES
from .lexicon import EXACT_ALIASES as _EXACT_ALIASES
from .lexicon import HAIR_CATEGORIES as _HAIR_CATEGORIES
from .lexicon import SERIES_APPEARANCE as _SERIES_APPEARANCE
from .lexicon import SERIES_SPLIT as _SERIES_SPLIT
from .lexicon import TYPED_ALIASES as _FRANCHISE_ALIASES
from .session import (
    MAX_GUESSES,
    MAX_TURNS,
    Candidate,
    GuessSession,
    answer_label,
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
    chip_likelihood,
    clue_likelihood,
    clue_overlap_likelihood,
    clue_question,
    is_choice,
    make_dynamic,
    noul_criteria,
    series_question,
    valid_answers,
    visual_residual,
    visual_search_phrases,
    yesno_visual_likelihood,
)
from .chips import chip_residual, free_text_trait_hits, free_text_trait_ids
from .traits import _clue_requirements

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
# Extra log-odds for a title character once the player named that work.
# One capped hair-choice gap is about log(1.5/8.5) - log(1/8) ≈ 0.35, and a
# royalty tag under the heuristic cap is about log(0.6/0.5) ≈ 0.18. 1.05
# beats both together. A clear model miss (log(0.15/0.85) ≈ -1.7) still wins.
_HEADLINER_BONUS = 1.05
# Two alias rows each at least this probable mean the identity is still split.
# Guessing either fragment commits before the mass is one person.
_ALIAS_SPLIT_MASS = 0.15
# Asked once after a same-work wrong guess, first trait the two answer differently.
_RECOVERY_QIDS = (
    "hair_color",
    "hair_long",
    "hair_short",
    "hair_twintails",
    "power_sword",
    "gender_female",
    "gender_male",
)
# Typed aliases: lexicon/franchises.yml. ``_typed_franchise`` applies the
# ``exact`` flag (whole clue only). Broad and hair sets: lexicon/categories.yml.
# Skip-on-yes and the focus ranking that uses those sets stay in this module.
ONLINE = os.getenv("WAIFU_ONLINE_SEARCH", "1").lower() in {"1", "true", "yes"}
# Facts offered to Laya when ranking which ones lead the search query, and how
# many of the winners go into the focused query.
FOCUS_OPTIONS = 8
FOCUS_TAKE = 3

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
            "instructions": "Which fact most narrows down the character?",
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
        "instructions": "Does one `characters` entry fit `confirmed_facts`?",
        "criteria": {
            "true": "yes, at least one listed character fits all the confirmed facts",
            "false": "no, none of the listed characters fits all the confirmed facts",
        },
    }
}
# Hoisted so the state-window budget and ``_pick_question`` share one head.
_READY_TO_GUESS = {
    "ready_to_guess": {
        "type": "noul",
        "instructions": "Do `confirmed_facts` name one character?",
        "criteria": {
            "true": "yes, the facts so far point at one specific character",
            "false": "no, several different characters still fit the facts",
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
    inside ``find_candidates``. Once a series is known, ``pin`` keeps that
    work in every template group, next to the rare visual traits from the seed.
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
            # The pin is also a search term, so the same work was often listed
            # three times (canonical label, typed detail, pin). Those copies
            # filled the focus window and the lead query never mentioned the
            # hair colour. Drop phrases the pin already says.
            terms = [anchor, *[term for term in terms if not sources._fact_in_pin(term, anchor)]]
            focus = [anchor, *[
                fact for fact in focus if not sources._fact_in_pin(fact, anchor)
            ]][:FOCUS_TAKE]
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
                pin=anchor or None,
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


# Appearance pin order: lexicon/categories.yml. hair_color plus the
# series_appearance flags (the angel kit). ``_pinned_question_ids`` still
# decides when. series_split categories become question ids below.
_ANGEL_KIT = frozenset(qid for qid in _SERIES_APPEARANCE if qid != "hair_color")
# A same-work runner below this mass is an extra, not a near-twin. Deferring
# a settled guess for that extra spent turns the leader had already earned.
_NEAR_TWIN_MASS = 0.10


def _series_split_qids() -> tuple[str, ...]:
    """Bank question ids for ``series_split`` categories, in lexicon order.

    ``look_ears`` is the category and ``look_animal_ears`` is the question.
    Pinning the category name would never match the bank.
    """
    found: list[str] = []
    for cat in _SERIES_SPLIT:
        for question in QUESTION_BANK:
            if question["id"] in found:
                continue
            if question["category"] == cat or question["id"] == cat:
                found.append(question["id"])
                break
    return tuple(found)


_SERIES_SPLIT_QIDS = _series_split_qids()


def _candidate_has_trait(question: dict[str, Any], cand: Candidate) -> bool:
    """Whether ``cand`` shows one of ``question``'s yes-tags.

    Tags are checked first. A blurb that only says "automail" or "wolf ears"
    still counts: those rows often never received the mined slug as a tag,
    and a silent side is not a confirmed "no" until the question is asked.
    """
    true = set(question.get("tags_true") or [])
    if not true:
        return False
    if true & set(cand.tags):
        return True
    blob = f"{cand.name} {cand.blurb}"
    return bool(true & set(web_search.mine_trait_slugs(blob)))


def _split_look_ids(sess: GuessSession) -> list[str]:
    """Rare-look questions that divide the locked franchise's living cast.

    Halo, wings and horns used to be pinned on every series lock and burned
    the turns that would have asked a prosthetic or animal ears. A look
    nobody in that cast differs on is left to information gain.
    """
    franchise = _confirmed_series(sess)
    if not franchise:
        return []
    cast = [c for c in sess.alive_candidates() if _in_franchise(c, franchise)]
    if len(cast) < 2:
        return []
    out: list[str] = []
    for qid in _SERIES_SPLIT_QIDS:
        question = QUESTIONS_BY_ID.get(qid)
        if question is None:
            continue
        present = sum(1 for cand in cast if _candidate_has_trait(question, cand))
        if 0 < present < len(cast):
            out.append(qid)
    return out


def _pinned_question_ids(sess: GuessSession) -> list[str]:
    """Appearance questions the seed named, plus cast-splitting looks after a series lock.

    Information gain never asked hair colour or halo: every Trinity student
    shares school, uniform and teen, and halo was buried in one horns/wings
    question. The traits the player already typed have to be offered early.
    The angel kit is not forced again once the series is known unless that
    text already named it. A prosthetic or animal ears that splits the cast
    is pinned instead, or it loses to the kit and is never asked.
    """
    ids: list[str] = []
    for text in _free_text(sess):
        for qid in appearance_question_ids(text):
            if qid not in ids:
                ids.append(qid)
    if _confirmed_series(sess):
        for qid in _SERIES_APPEARANCE:
            if qid in _ANGEL_KIT:
                continue
            if qid not in ids:
                ids.append(qid)
        for qid in _split_look_ids(sess):
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


# Round-level calls share one 512-token sequence. ``laya.common.build_sequence``
# serializes the state with ``json.dumps`` and keeps only the first state
# tokens (right truncation). Profiles used to lead that JSON, so answered
# traits were the first thing cut. Trait rows lead now; identities, prose
# facts and history are the padding the cut may remove.
_LAYA_ROUND_PROFILE = 160
_LAYA_ROUND_FACTS = 8
_LAYA_ROUND_HISTORY = 6
# Column order for every durable row. A button answer is 1.0 (yes) or 0.0
# (no). A typed chip stays at the top of the mild band so it cannot read as
# that hard yes — the same reason a raw "angel" noul is not applied.
_TRAIT_FIELDS = ("id", "answer", "score")
_SOFT_YES_SCORE = 0.6
_SOFT_NO_SCORE = 0.4
# Wordpiece stand-in: letters, numbers and punctuation are separate pieces,
# so this cuts at least as much as the English window. No tokenizer in the
# offline suite, and the real one must not be downloaded from a test.
_TOKEN_PIECE = re.compile(r"[A-Za-z]+|[0-9]+|[^\s]")


def _piece_count(text: str) -> int:
    """How many pessimistic wordpieces ``text`` is."""
    return sum(1 for _ in _TOKEN_PIECE.finditer(text or ""))


def _noul_head_tokens(instructions: str, criteria: dict[str, str]) -> int:
    """Tokens ``build_sequence`` spends before the state on one noul question.

    CLS, ``noul question:`` plus the instruction, SEP, a masked false option,
    a masked true option, SEP. Each option is capped at 48 pieces, matching
    the library. The state then receives ``max_len - head - 1`` tokens.
    """
    ins = "noul question: " + instructions
    options = (
        " false: " + (criteria.get("false") or "no, the statement does not hold"),
        " true: " + (criteria.get("true") or "yes, the statement holds"),
    )
    option_tokens = sum(min(48, 1 + _piece_count(option)) for option in options)
    return 1 + _piece_count(ins) + 1 + option_tokens + 1


def _shared_state_room() -> int:
    """State tokens left for ``pool_fits`` and ``ready_to_guess``.

    The tighter of those two heads wins. Per-candidate ``match`` is its own
    sequence and is not this budget.
    """
    heads = [
        _noul_head_tokens(spec["instructions"], spec["criteria"])
        for spec in (_POOL_FITS["pool_fits"], _READY_TO_GUESS["ready_to_guess"])
    ]
    return max(0, laya_client.MAX_LEN - max(heads) - 1)


def _prefix_by_tokens(text: str, room: int) -> str:
    """The leading slice of ``text`` that fits in ``room`` pieces.

    Whitespace is not a piece, same as the wordpiece stand-in. ``room`` <= 0
    keeps nothing: the head already filled the window.
    """
    if room <= 0:
        return ""
    count = 0
    end = 0
    for match in _TOKEN_PIECE.finditer(text):
        count += 1
        end = match.end()
        if count >= room:
            return text[:end]
    return text


def _trait_score(question: dict[str, Any], answer: str) -> float | None:
    """Numeric score for one evidence row, or None if it is not a bank trait.

    Free-text clues are prose. They stay on the overlap path and are not
    copied into the window, where Laya would treat the sentence as a fact.
    """
    if question.get("clues") or question.get("category") == "clue":
        return None
    if not question.get("id"):
        return None
    if question.get("soft_chip"):
        return _SOFT_YES_SCORE if answer == "yes" else _SOFT_NO_SCORE
    if is_choice(question):
        return 1.0
    if answer == "yes":
        return 1.0
    if answer == "no":
        return 0.0
    return None


def _answered_trait_pack(sess: GuessSession) -> dict[str, Any]:
    """Durable rows for every answered bank or chip trait, in evidence order.

    Each row is ``[id, answer, score]``. Query-LLM rewrites are not rows:
    they are search strings, and pasting them here would make Laya treat
    generated prose as identity evidence.
    """
    rows: list[list[Any]] = []
    for question, answer in sess.evidence.values():
        score = _trait_score(question, answer)
        if score is None:
            continue
        rows.append([question["id"], answer, score])
    return {"fields": list(_TRAIT_FIELDS), "rows": rows}


def _trait_signature(state: dict[str, Any]) -> list[tuple[str, str, float]]:
    """``(id, answer, score)`` for every durable row in ``state``."""
    rows = (state.get("answered_traits") or {}).get("rows") or []
    return [(row[0], row[1], row[2]) for row in rows]


def _traits_surviving_window(
    state: dict[str, Any], room: int | None = None,
) -> list[tuple[str, str, float]]:
    """Rows whose JSON is still intact after the English-window cut.

    Compare this to ``_trait_signature``. A shorter list means the cut still
    landed inside the trait block.
    """
    budget = _shared_state_room() if room is None else room
    prefix = _prefix_by_tokens(json.dumps(state, ensure_ascii=False), budget)
    kept: list[tuple[str, str, float]] = []
    for row in (state.get("answered_traits") or {}).get("rows") or []:
        if json.dumps(row, ensure_ascii=False) not in prefix:
            break
        kept.append((row[0], row[1], row[2]))
    return kept


def _laya_state(sess: GuessSession, candidates: list[Candidate]) -> dict[str, Any]:
    """Compact state for one shared Laya call (``pool_fits``, ``ready_to_guess``).

    Answered trait rows lead so the right-truncation drops profiles and
    history first. Per-candidate ``match`` carries the same rows ahead of
    that candidate's profile.
    """
    facts = sess.constraints[-_LAYA_ROUND_FACTS:] or ["nothing confirmed yet"]
    history = sess.history()[-_LAYA_ROUND_HISTORY:] or [
        {"question": "none yet", "answer": "", "detail": ""},
    ]
    state = {
        "answered_traits": _answered_trait_pack(sess),
        "goal": GOAL,
        "characters": {c.name: c.profile(budget=_LAYA_ROUND_PROFILE) for c in candidates},
        "confirmed_facts": facts,
        "answer_history": history,
        "questions_asked": sess.turn,
    }
    if _traits_surviving_window(state) != _trait_signature(state):
        # Still truncation: readiness and pool_fits would judge without
        # those answers. Do not raise — the turn has to keep playing.
        timing.note(trait_window="truncated")
    return state


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
    answers = laya_client.ask(_laya_state(sess, sess.scoring_pool()), _READY_TO_GUESS)
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


def _specific_work(text: str) -> str:
    """Work named in ``text``, preferring a longer headliner title.

    "final fantasy vii" contains the marker "final fantasy". The headliner
    title keeps the VII so a different numbered entry is not the same cast.
    "wonder woman" is not a series marker; the headliner list still names it.
    """
    marker = web_search.franchise_label(text)
    head = web_search.headliner_label(text)
    if head and marker:
        marker_key, head_key = series_key(marker), series_key(head)
        if head_key.startswith(marker_key) and len(head_key) > len(marker_key):
            return head
        return marker
    return marker or head


def _typed_franchise(sess: GuessSession) -> str:
    """Canonical series named in the player's own text, or "".

    "Another series" plus a detail never sets ``_confirmed_series``, so
    "cowboy bebop" / "star wars" / "simpsons" used to fall out of the lead
    query. A whole-phrase alias keeps that work in every later search.
    A headliner phrase ("wonder woman", "final fantasy vii") counts too.
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
    for text in texts:
        found = _specific_work(text)
        if found:
            return found
    return ""


def _series_trait_anchor(sess: GuessSession) -> str:
    """Work, personal alias, and rare visual traits that every search keeps.

    Template groups used to retry the newest fact alone. After a series was
    only the first fact, that retry was "female character" or "long-haired
    character" and the pool filled with other franchises and later
    mantle-holders. The pin is passed through separately so it survives that
    narrow retry. A typed franchise counts: the player never clicked a chip.
    Several co-leads are not stuffed into this string; name search asks for
    each of them. One title name that is not the work title is included, as
    is a disjoint personal alias ("Diana Prince").
    """
    series = _confirmed_series(sess) or _typed_franchise(sess)
    if not series:
        return ""
    texts = [series, *_free_text(sess)]
    parts = web_search.fulltext_pin_parts(texts)
    if not parts:
        parts = [series]
    elif not any(series.lower() in part.lower() for part in parts):
        parts = [series, *parts]
    phrases: list[str] = []
    for text in (sess.seed, *(a.get("detail") for a in sess.asked)):
        phrases.extend(visual_search_phrases(text or ""))
    for text, _rank in _fact_entries(sess):
        phrases.extend(visual_search_phrases(text))
    for phrase in phrases:
        if phrase not in parts:
            parts.append(phrase)
    return " ".join(dict.fromkeys(parts))[:180]


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
            label = _specific_work(a.get("detail") or "")
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


def _candidate_laya_state(
    cand: Candidate, question: dict[str, Any], prior: dict[str, Any] | None,
) -> dict[str, Any]:
    """One candidate's match state: durable rows, then the profile.

    The question being scored is omitted from ``prior``. Including its own
    answer would leak the label into the judgment. A long profile follows
    the rows, so the window cuts the blurb rather than the traits.
    """
    state: dict[str, Any] = {
        "answered_traits": prior or {"fields": list(_TRAIT_FIELDS), "rows": []},
        "candidate": cand.profile(),
    }
    if question.get("clues"):
        state["clues"] = question["clues"]
    return state


def _prior_trait_pack(sess: GuessSession, skip_id: str) -> dict[str, Any]:
    """Trait rows answered before ``skip_id``, not the question being scored.

    Later rows are omitted so a replay for a new candidate sees the same
    prefix the first cohort saw. Including answers that landed afterwards
    would change the match prompt and make the cache lie.
    """
    pack = _answered_trait_pack(sess)
    earlier: list[list[Any]] = []
    for row in pack["rows"]:
        if row[0] == skip_id:
            break
        earlier.append(row)
    pack["rows"] = earlier
    return pack


def _choice_probabilities(
    pool: list[Candidate], question: dict[str, Any],
    prior: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Option probabilities per candidate, one Laya ``choice`` call each.

    Same reasoning as ``_match_probabilities``: a shared state blurs the
    candidates together. Answers that do not cover every option, or carry
    non-probabilities, are dropped so the heuristic takes over. Prior trait
    rows lead each state so a wide option head cannot cut them first.
    """
    out: dict[str, dict[str, float]] = {}
    keys = list(question["options"])
    for c in pool:
        answers = laya_client.ask(
            _candidate_laya_state(c, question, prior),
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
    pool: list[Candidate], question: dict[str, Any],
    prior: dict[str, Any] | None = None,
) -> dict[str, float]:
    """P(question is true) for each candidate, one Laya call per candidate.

    One shared state with every candidate in it does not work: Laya attends to
    the whole state, so a batch of ``match_<i>`` questions comes back with
    near-identical probabilities (measured: 1.0 for all ten, anime and games
    alike). Scoring each candidate against its own state separates them cleanly
    -- and is faster, because each sequence is short. Answered traits lead
    that state so the profile tail is what a tight head cuts.
    """
    probs: dict[str, float] = {}
    # Explicit true/false option text beats Laya's generic "yes, the statement
    # holds" -- the model is scoring against a restatement of the real claim.
    criteria = question.get("criteria") or noul_criteria(question["instructions"])
    for c in pool:
        answers = laya_client.ask(
            _candidate_laya_state(c, question, prior),
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
    Typed chips and free-text clues never take a raw noul, so one unmatched
    word cannot floor the pool.
    A visual clue the profile clearly misses is a strong down-rank, not a
    removal, so a thin blurb can still recover on a later answer. Alias rows
    are one identity before any of that is added up.
    """
    sess.collapse_identities()
    for question, answer in sess.evidence.values():
        _eliminate_by_medium(sess, question, answer)
    live = sess.alive_candidates()
    if not live:
        return
    for c in live:
        c.logodds = popularity_prior(c.popularity)
    for qid, (question, answer) in sess.evidence.items():
        prior = _prior_trait_pack(sess, qid)
        if is_choice(question):
            # Known medium/series is data, not a judgment: no model call.
            known = {c.id: d for c in live if (d := _known_choice(question, c))}
            missing = [c for c in live
                       if c.id not in known and (c.id, qid) not in sess.choice_cache]
            dists = _choice_probabilities(missing, question, prior)
            if dists:
                sess.laya_used = True
                sess.choice_cache.update({(cid, qid): d for cid, d in dists.items()})
            for c in live:
                dist = (known.get(c.id) or sess.choice_cache.get((c.id, qid))
                        or _tag_choice(question, c))
                c.logodds += math.log(min(0.98, max(0.02, dist.get(answer, 0.0))))
            continue
        # A typed chip is a nudge, not a model vote. Laya's noul on the raw
        # words ("angel", "white dress") came back near 0 for the whole pool
        # and floored everyone the earlier facts still fit.
        if question.get("soft_chip"):
            for c in live:
                p = chip_likelihood(question["id"], c.blurb, c.tags)
                # "not a demon" is stored as no. Invert so the named trait
                # falls, and a profile that simply lacks it stays near a half.
                if answer != "yes":
                    p = 1.0 - p
                c.logodds += math.log(min(0.98, max(0.02, p)))
            continue
        # Free-text clues are not a Laya vote. A noul of ~0 on "angel" or
        # "white dress" floored every candidate the earlier facts still fit.
        # Visual combinations still use the profile likelihood; anything else
        # stays in the heuristic band.
        if question.get("clues"):
            for c in live:
                visual = _profile_likelihood(question, c)
                if visual is None:
                    visual = clue_overlap_likelihood(question["clues"], c.blurb)
                likelihood = visual if answer == "yes" else 1.0 - visual
                c.logodds += math.log(min(0.98, max(0.02, likelihood)))
            continue
        missing = [c for c in live if (c.id, qid) not in sess.match_cache]
        probs = _match_probabilities(missing, question, prior)
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


def _in_franchise(cand: Candidate, franchise: str) -> bool:
    """Whether ``cand`` belongs to ``franchise``: its series if known, else its page.

    A missing series still counts when the page names the work: Spike's row
    was "Unknown" while Ed's said Cowboy Bebop. A known series is trusted
    over the blurb, or a Tekken fighter whose page mentions a Vocaloid
    collaboration would join the Vocaloid cast.
    """
    if series_key(cand.series):
        return same_series(cand.series, franchise)
    return bool(web_search.franchise_mentioned(franchise, f"{cand.name} {cand.blurb}"))


def _player_work_texts(sess: GuessSession) -> list[str]:
    """Player-typed clues plus a series chip they actually picked."""
    texts = list(_free_text(sess))
    for asked in sess.asked:
        option = (asked.get("options") or {}).get(asked.get("answer") or "")
        if option and option.get("series_key"):
            fact = option.get("fact") or ""
            if fact:
                texts.append(fact)
    return texts


def _active_headliner(sess: GuessSession) -> tuple[str, tuple[str, ...]] | None:
    """Franchise and its title characters named by the player, or None.

    Reads the player's own text, not candidate blurbs. "wonder woman" is a
    headliner phrase even though it is not a series-chip marker.
    """
    return web_search.headliner_from_texts(_player_work_texts(sess))


def _name_is_headliner(name: str, names: tuple[str, ...]) -> bool:
    """Whether ``name`` is one of ``names``, including a known alias spelling."""
    iid = identity_id(name)
    for listed in names:
        if iid and iid == identity_id(listed):
            return True
        if same_character(name, listed):
            return True
    return False


def _eligible_headliner(cand: Candidate, franchise: str, names: tuple[str, ...]) -> bool:
    """Whether ``cand`` is a title character of ``franchise`` and not another work.

    A publisher series such as DC Comics is not a different work: Diana's
    page is often filed there. A known other series (Kingdom Hearts Cloud)
    does not take the Final Fantasy VII bonus.
    """
    if not _name_is_headliner(cand.name, names):
        return False
    if not series_key(cand.series) or is_publisher_series(cand.series):
        return True
    blob = f"{cand.name} {cand.blurb}"
    return _in_franchise(cand, franchise) or bool(web_search.franchise_mentioned(franchise, blob))


def _in_headliner_cast(cand: Candidate, franchise: str, names: tuple[str, ...]) -> bool:
    """Whether ``cand`` belongs to the named work's cast, including the lead."""
    if _name_is_headliner(cand.name, names):
        return _eligible_headliner(cand, franchise, names)
    if is_publisher_series(cand.series):
        return bool(web_search.franchise_mentioned(franchise, f"{cand.name} {cand.blurb}"))
    if _in_franchise(cand, franchise):
        return True
    return bool(web_search.franchise_mentioned(franchise, f"{cand.name} {cand.series} {cand.blurb}"))


def _yesno_likelihood(sess: GuessSession, question: dict[str, Any], cand: Candidate) -> float:
    """P(yes) for one stored yes/no, using the same fallback as rescoring."""
    visual = _profile_likelihood(question, cand)
    p = sess.match_cache.get((cand.id, question["id"]))
    if visual is not None:
        p = visual
    elif p is None:
        p = min(0.6, max(0.4, _tag_match(question, cand)))
    return p if p is not None else 0.5


def _lift_title_royalty(
    sess: GuessSession, live: list[Candidate], franchise: str, names: tuple[str, ...],
) -> None:
    """Give the title character at least the cast's royalty likelihood.

    Wonder Woman is the princess the book is named after. ``job_royalty=yes``
    used to move the posterior onto Nubia because only her row carried the
    tag. The lift runs only when the franchise title is itself a headliner,
    so an Eva pilot does not inherit a side character's royalty score.
    """
    if not any(series_key(franchise) == series_key(listed) for listed in names):
        return
    answered = [q for q, a in sess.evidence.values() if q.get("id") == "job_royalty" and a == "yes"]
    if not answered:
        return
    question = answered[0]
    cast = [c for c in live if _in_headliner_cast(c, franchise, names)]
    heads = [c for c in cast if _eligible_headliner(c, franchise, names)]
    if not heads or len(cast) < 2:
        return
    deltas = {
        c.id: math.log(min(0.98, max(0.02, _yesno_likelihood(sess, question, c))))
        for c in cast
    }
    best = max(deltas.values())
    for head in heads:
        gap = best - deltas[head.id]
        if gap > 0:
            head.logodds += gap


def _apply_identity_priors(sess: GuessSession, live: list[Candidate]) -> None:
    """Prefer a franchise lead, a title character, and an exact short name.

    Called at the end of a rescore, after trait evidence. A clear trait miss
    is still larger than these bonuses. Publisher buckets are not a series,
    so Marvel characters are not all treated as one cast. When the player
    named a work, its headliners outrank supporting cast that only matches
    a shared tag such as royalty.
    """
    franchise = _confirmed_series(sess) or _typed_franchise(sess)
    headliner = _active_headliner(sess)
    work, title_names = headliner if headliner else ("", ())
    if franchise:
        for cand in live:
            # Diana's row is often filed under DC Comics. That publisher bucket
            # must not strip the in-franchise bonus from the title character
            # while Nubia, filed under Wonder Woman, keeps it.
            named = bool(title_names) and _eligible_headliner(cand, work, title_names)
            if is_publisher_series(cand.series) and not named:
                continue
            if named or _in_franchise(cand, franchise):
                cand.logodds += _SERIES_LEAD_BONUS
            elif series_key(cand.series):
                cand.logodds -= _SIDE_CHARACTER_PENALTY
    focus = series_key(franchise)
    groups: dict[str, list[Candidate]] = {}
    for cand in live:
        if not focus or is_publisher_series(cand.series):
            continue
        if not _in_franchise(cand, franchise):
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
    if title_names:
        for cand in live:
            if _eligible_headliner(cand, work, title_names):
                cand.logodds += _HEADLINER_BONUS
        _lift_title_royalty(sess, live, work, title_names)
    typed = [t.strip().lower() for t in _free_text(sess)]
    if not typed or not (franchise or _medium_hint(sess)):
        return
    exact = [c for c in live if c.name.strip().lower() in typed]
    if not exact:
        return
    for cand in live:
        shorts = [short for short in exact if longer_namesake(short.name, cand.name)]
        if not shorts:
            continue
        # An unknown series ("Web result") cannot prove two different people:
        # a stray "Spike" hit must not sink Spike Spiegel.
        if any(not series_key(short.series) or not series_key(cand.series)
               or same_series(short.series, cand.series) for short in shorts):
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


def _fill_portraits(cands: list[Candidate]) -> None:
    """Resolve a public image for shown candidates that still have none.

    The cat is what the page draws for a null URL. Guess, win, and the top
    list search once for "name series character" before they are sent.
    """
    from ..sources.portraits import fill_portraits

    fill_portraits(cands, limit=4)


def _top_payload(sess: GuessSession, n: int = 3) -> list[dict[str, Any]]:
    ranked = sess.posterior()[:n]
    _fill_portraits([c for c, _p in ranked])
    return [c.public(p) for c, p in ranked]


def _alias_split_blocks_guess(sess: GuessSession) -> bool:
    """Whether two high-mass rows are still unresolved spellings of one person.

    Collapsing should have merged them. If it did not, committing the top
    fragment repeats the Asuka miss: Sohryu guessed while Soryu still holds mass.
    """
    buckets: dict[str, int] = {}
    for cand, prob in sess.posterior():
        iid = identity_id(cand.name)
        if iid and prob >= _ALIAS_SPLIT_MASS:
            buckets[iid] = buckets.get(iid, 0) + 1
    return any(count >= 2 for count in buckets.values())


def _guess_payload(sess: GuessSession) -> dict[str, Any]:
    """Name the leading real character. Alias rows are one person first.

    Franchise and species pages are already absent from ``posterior``.
    """
    sess.collapse_identities()
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
    _fill_portraits([cand])
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
    even without that support. A still-split alias identity does not commit
    before the turn cap: two high-mass spellings of one person are not a guess.
    """
    sess.collapse_identities()
    if _alias_split_blocks_guess(sess) and sess.turn < MAX_TURNS:
        return False
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


def _predicted_pole(question: dict[str, Any], cand: Candidate) -> str:
    """The option or yes/no a candidate's tags pick, or ``""`` when they are silent."""
    tags = set(cand.tags)
    if is_choice(question):
        hits = [key for key, opt in question["options"].items() if set(opt.get("tags") or []) & tags]
        return hits[0] if len(hits) == 1 else ""
    if set(question.get("tags_true") or []) & tags:
        return "yes"
    if set(question.get("tags_false") or []) & tags:
        return "no"
    return ""


def _same_work(a: Candidate, b: Candidate) -> bool:
    """Whether two candidates share a real series, not a publisher bucket."""
    if not series_key(a.series) or not series_key(b.series):
        return False
    if is_publisher_series(a.series) or is_publisher_series(b.series):
        return False
    return same_series(a.series, b.series)


def _recovery_question(sess: GuessSession) -> dict[str, Any] | None:
    """One unasked trait that separates a rejected guess from the new leader.

    After Kyoko, the next commit used to be another Soryu spelling. A trait
    the two answer differently is asked once. Returns None when the reject
    was not a same-work miss, and clears the one-shot flag either way.
    """
    rejected_id = sess.recovery_from
    if not rejected_id:
        return None
    sess.recovery_from = ""
    rejected = sess.by_id(rejected_id)
    if rejected is None:
        return None
    ranked = sess.posterior()
    if not ranked:
        return None
    leader = ranked[0][0]
    if identity_id(leader.name) and identity_id(leader.name) == identity_id(rejected.name):
        return None
    if not _same_work(rejected, leader):
        return None
    asked = sess.asked_ids()
    for qid in _RECOVERY_QIDS:
        if qid in asked or qid not in QUESTIONS_BY_ID:
            continue
        question = QUESTIONS_BY_ID[qid]
        left = _predicted_pole(question, rejected)
        right = _predicted_pole(question, leader)
        if left and right and left != right:
            return question
    return None


def _emit_asking(sess: GuessSession, question: dict[str, Any]) -> dict[str, Any]:
    """Record ``question`` as the pending turn and return the asking payload."""
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


def _near_twin_question(sess: GuessSession) -> dict[str, Any] | None:
    """An unasked rare look that splits the top two same-work candidates.

    Heinkel, Myuri and Mihawk were guessed while a prosthetic, animal ears
    or an eyepatch was still sitting in the bank. Recovery only runs after a
    wrong guess, and its poles require both rows to carry an explicit tag.
    A rare look separates when exactly one of the two shows it. The turn cap
    still guesses: deferring there would hide the best name we have.
    """
    if sess.turn >= MAX_TURNS:
        return None
    ranked = sess.posterior()
    if len(ranked) < 2 or ranked[1][1] < _NEAR_TWIN_MASS:
        return None
    leader, runner = ranked[0][0], ranked[1][0]
    if not _same_work(leader, runner):
        return None
    asked = sess.asked_ids()
    for qid in _SERIES_SPLIT_QIDS:
        if qid in asked:
            continue
        question = QUESTIONS_BY_ID.get(qid)
        if question is None:
            continue
        if _candidate_has_trait(question, leader) != _candidate_has_trait(question, runner):
            return question
    return None


def _advance(sess: GuessSession) -> dict[str, Any]:
    """Emit the next question, or a guess when the evidence is strong enough.

    A same-work near-twin with an unasked rare look is asked that look
    instead of committing. Alias-split still refuses the guess and keeps
    the information-gain question: those two rows are one person.
    """
    if sess.turn < MAX_TURNS:
        recovery = _recovery_question(sess)
        if recovery is not None:
            timing.note(recover=recovery["id"])
            return _emit_asking(sess, recovery)
    with timing.span("pick"):
        question, laya_answers = _pick_question(sess)
    alive = len(sess.alive_candidates())
    timing.note(turn=sess.turn, cands=f"{alive}/{len(sess.candidates)}")
    # Collapse happens inside the guess check. The split flag is read after
    # that, so a merge that just succeeded is allowed to commit.
    should = _should_guess(sess, laya_answers)
    blocked = bool(question) and _alias_split_blocks_guess(sess) and sess.turn < MAX_TURNS
    if should and not blocked and sess.turn < MAX_TURNS:
        twin = _near_twin_question(sess)
        if twin is not None:
            timing.note(defer=twin["id"])
            return _emit_asking(sess, twin)
    if (should or not question) and not blocked:
        timing.note(guess=True)
        return _guess_payload(sess)
    return _emit_asking(sess, question)


def _coerce_free_answer(current: dict[str, Any], answer: str, detail: str) -> tuple[str, str]:
    """Return ``(answer, detail)``. An empty answer means "reject".

    "angel" is not a yes/no key on "Is your character human?". Rejecting it
    handed the client an error payload with no pool. A known species word or
    a phrase becomes a detail and the closed question stays unanswered. A
    stray token such as "maybe" is still rejected.
    """
    answer = (answer or "").strip().lower()
    detail = (detail or "").strip()
    if answer in valid_answers(current):
        return answer, detail
    # A phrase or a known species/job word is a detail. A stray token such as
    # "maybe", or "yes" on a choice question, is still a bad answer.
    if not answer or not (free_text_trait_ids(answer) or any(ch in answer for ch in " ,")):
        return "", detail
    merged = " ".join(part for part in (detail, answer) if part)
    return "detail", merged[:300]


def _overlap_words(text: str) -> bool:
    """Whether ``text`` has a word the mild overlap clue will actually read.

    ``clue_overlap_likelihood`` ignores tokens under four letters. An empty
    residue would store a clue that adds the same half to every candidate.
    """
    return bool(re.search(r"[A-Za-z]{4,}", text or ""))


def _score_free_text(sess: GuessSession, text: str, qid: str) -> None:
    """Record a typed clue as a visual judgment, soft chips, and mild overlap.

    Visual phrases and chip words are stripped before the overlap clue, so
    "pink hair and a white dress" still ranks the dress. Nothing here writes
    a hard bank yes: chips stay ``soft_chip``, and other words stay a clue.
    """
    text = (text or "").strip()
    if not text:
        return
    hits = free_text_trait_hits(text)
    visual = bool(_clue_requirements(text))
    if visual:
        score_candidates(sess, clue_question(qid, text), "yes")
    residual = visual_residual(text) if visual else text
    if hits:
        residual = chip_residual(residual)
    residual = " ".join(residual.split())
    if residual and _overlap_words(residual):
        # Evidence is keyed by question id. Reusing the visual clue's id
        # would replace that judgment with the leftover words.
        overlap_id = f"{qid}_rest" if visual else qid
        score_candidates(sess, clue_question(overlap_id, residual), "yes")
    for trait_id, polarity in hits:
        if trait_id in sess.evidence:
            continue
        question = dict(QUESTIONS_BY_ID[trait_id])
        question["soft_chip"] = True
        score_candidates(sess, question, polarity)


@timing.traced("start")
def start(seed: str = "") -> dict[str, Any]:
    sess = new_session(seed)
    if seed:
        sess.constraints.append(seed)
        _score_free_text(sess, seed, "clue_seed")
    refresh_candidates(sess, limit=16, initial=True)
    payload = _advance(sess)
    payload["seed"] = sess.seed
    return payload


@timing.traced("answer")
def submit_answer(sess: GuessSession, answer: str, detail: str = "") -> dict[str, Any]:
    if sess.stage != "asking" or not sess.asked:
        return {"error": "no question is pending", "session_id": sess.id, "stage": sess.stage}
    current = sess.asked[-1]
    answer, detail = _coerce_free_answer(current, answer, detail)
    if not answer:
        allowed = valid_answers(current)
        return {"error": f"answer must be one of {sorted(allowed)}", "session_id": sess.id}

    current["answer"] = answer
    current["detail"] = detail or None

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
        _score_free_text(sess, current["detail"], f"clue_{sess.turn}")

    # Every answer opens a new search branch. Discover and replay evidence
    # before choosing the next question or declaring a winner.
    refresh_candidates(sess)

    sess.touch()
    return _advance(sess)


def _reject_identity(sess: GuessSession, cand: Candidate) -> None:
    """Hard-eliminate ``cand`` and every stored spelling of that person.

    A wrong guess of one Asuka spelling used to leave Sohryu and Shikinami
    alive, so the next commit was the same person. Elimination is the
    down-weight: log-odds go to -inf and every id is recorded in ``rejected``.
    """
    iid = identity_id(cand.name)
    for other in sess.candidates:
        if other.id == cand.id or (iid and identity_id(other.name) == iid):
            other.alive = False
            other.logodds = -math.inf
            sess.rejected.add(other.id)
    shown = canonical_name(cand.name) or cand.name
    sess.constraints.append(f"The character is not {shown}")


@timing.traced("guess_result")
def submit_guess_result(sess: GuessSession, correct: bool) -> dict[str, Any]:
    """Record whether the pending guess was right and continue the round.

    A wrong guess eliminates that whole identity, then merges whatever alias
    rows are still live before the next question or commit.
    """
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
        sess.recovery_from = guessed.id
        _reject_identity(sess, guessed)
        sess.collapse_identities()
        if sess.evidence:
            _rescore_candidates(sess)

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
            {
                "qid": a["qid"],
                "text": a["text"],
                "answer": a["answer"],
                "detail": a["detail"],
                # Choice answers are stored as option keys. The label is what
                # the player picked, so a refresh can rebuild the chip trail.
                "label": answer_label(a) or None,
            }
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
            _fill_portraits([cand])
            payload["guess"] = cand.public(prob)
            payload["guess_number"] = sess.guesses_made
    elif sess.stage == "done" and sess.winner:
        cand = sess.by_id(sess.winner)
        if cand is not None:
            _fill_portraits([cand])
            payload["winner"] = cand.public(1.0)
            payload["correct"] = True
            payload["turns"] = sess.turn
    return payload
