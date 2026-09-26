"""Structured memory for offline honest-answer benches.

Empty-seed runs used to keep only the engine's ``asked`` list. A miss was
then a posterior with no record of which chip, negation, or free-text
residual produced it, and the next honest answer could contradict a trait
already given. This module snapshots each turn and answers from that record
plus a known target card. It does not start a server and it does not search.

A game that ends wrong calls ``record_miss`` (or ``finish(..., correct=False)``)
before the wrong guess is rejected. The export's ``miss`` object carries five
fields read from the session posterior and from timing notes already on the
turn payloads (``trait_window``, ``defer``). Beside those, ``look_ask`` and
``soft_cast_delta`` record when the wolf looks were asked and the fame gap
after each answer. ``miss_pack`` lists one stable row per character and does
not include the look-pin log.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import timing
from ..names import is_publisher_series, same_character, same_series, series_key
from .chips import chip_residual, free_text_trait_hits
from .session import Candidate, GuessSession
from .traits import is_choice

# A yes on one side forces the other. Recorded answers win over the card so
# a bench cannot flip gender after it has already said female.
_GENDER_OPPOSITE = {"gender_female": "gender_male", "gender_male": "gender_female"}

# Guide band for a crowded top two. Kept as a literal so a bench label does
# not move when ``WAIFU_NEKOMINI_LEADER_MARGIN`` changes the guess gate.
_MARGIN_BAND = 0.15
# Fact ranking ignores a choice whose top mass is below 1.5/n. For a pair,
# n = 2, so a renormalized leader under 0.75 is that same flatness bar.
# Binary entropy of a 0.75/0.25 split is about 0.81 bits; a coin flip is 1.
_PAIR_UNIFORM_MASS = 1.5 / 2

# Stable key order for a miss row. ``franchise_pair`` is the sibling that
# says whether ``franchise_match_entropy`` was taken on a same-work pair.
MISS_KEYS = (
    "target_in_shortlist",
    "top_two_posterior_margin",
    "franchise_match_entropy",
    "franchise_pair",
    "trait_window",
    "defer",
    "miss_label",
)


def _timing_view(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return timing notes from a payload, a timing dict, and the live trace.

    Traced handlers copy ``timing.note`` fields onto ``payload["timing"]``.
    A bench that still sits inside the trace has those notes only on the
    trace. Payload keys win when both sides set one, and a missing payload
    key must not erase a live note.
    """
    fields: dict[str, Any] = {}
    trace = timing.current()
    if trace is not None:
        fields.update(trace.fields)
    if not raw:
        return fields
    nested = raw.get("timing") if isinstance(raw.get("timing"), Mapping) else None
    source = nested if nested is not None else raw
    for key, value in source.items():
        if value is not None:
            fields[key] = value
    return fields


def _defer_qid(value: Any) -> str | None:
    """Return a defer qid, or None when the note did not name one."""
    if value is None or value is False:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _is_series_split(qid: str | None) -> bool:
    """Whether ``qid`` is a rare-look question near-twin deferral can ask.

    The id list is the engine's, so the label cannot treat some other note
    as a series_split defer.
    """
    if not qid:
        return False
    from .engine import _SERIES_SPLIT_QIDS

    return qid in _SERIES_SPLIT_QIDS


def _same_work(a: Candidate, b: Candidate) -> bool:
    """Whether two candidates share a real series, not a publisher bucket.

    Same predicate as ``engine._same_work``. Copied so the miss log does not
    call question pick or guess commit.
    """
    if not series_key(a.series) or not series_key(b.series):
        return False
    if is_publisher_series(a.series) or is_publisher_series(b.series):
        return False
    return same_series(a.series, b.series)


def _series_hard_pinned(sess: GuessSession) -> bool:
    """Whether the player confirmed a series, not merely typed around one.

    ``engine._confirmed_series`` is the hard pin: a chosen series option, or
    "Another series" plus a detail that names a work. Seed text that mentions
    a franchise is only a search anchor, so an empty-seed run with no chip
    stays eligible for ``pin-thin``.
    """
    from .engine import _confirmed_series

    return bool(_confirmed_series(sess))


def _shortlist_names(sess: GuessSession) -> list[str]:
    """Names in the commit pool.

    ``posterior`` is who a guess can name. ``scoring_pool`` is the readiness
    shortlist. The pending guess is included so a row posterior filtered
    still counts when it is the name just committed.
    """
    names = [cand.name for cand, _prob in sess.posterior()]
    for cand in sess.scoring_pool():
        if cand.name not in names:
            names.append(cand.name)
    if sess.pending_guess:
        pending = sess.by_id(sess.pending_guess)
        if pending is not None and pending.name not in names:
            names.append(pending.name)
    return names


def _target_in_shortlist(sess: GuessSession, target: str) -> bool:
    """Whether ``target`` matches a name still in the commit shortlist."""
    if not (target or "").strip():
        return False
    return any(same_character(target, name) for name in _shortlist_names(sess))


def _pair_entropy(p1: float, p2: float) -> tuple[float, float]:
    """Return ``(entropy bits, larger renormalized mass)`` for two probs.

    The masses are renormalized so a third candidate does not shrink the
    entropy of the pair. Zero total mass is a flat 0, not a coin flip.
    """
    total = p1 + p2
    if total <= 0.0:
        return 0.0, 1.0
    left, right = p1 / total, p2 / total
    entropy = 0.0
    for prob in (left, right):
        if prob > 0.0:
            entropy -= prob * math.log2(prob)
    return entropy, max(left, right)


def miss_label(
    *,
    target_in_shortlist: bool,
    margin: float,
    pair_top_mass: float,
    same_work: bool,
    have_pair: bool,
    trait_window: str,
    defer: str | None,
    series_pinned: bool,
) -> str:
    """Pick one stable miss label. First match wins.

    ``search`` — the target is not in the shortlist, so later labels would
    describe other people. ``truncation`` — a turn noted
    ``trait_window=truncated``; shared Laya state dropped trait rows.
    ``near-twin-fired`` — a series_split ``defer`` was noted, and the top
    two are close (margin ≤ 0.15) or a same-work pair is near-uniform
    (renormalized leader under 0.75). ``pin-thin`` — the target is in the
    pool, defer did not ask a series_split, and no series was hard-pinned.
    A clear leader (margin above 0.15) is the usual case; a tight pair
    without that defer is the same bucket, so it is not called a Laya miss.
    ``laya-match`` — in the shortlist, window intact, and a series pin or a
    series_split defer already happened without the near-twin band. This
    does not compare a match score to a ground-truth answer.
    """
    if not target_in_shortlist:
        return "search"
    if trait_window == "truncated":
        return "truncation"
    deferred = _is_series_split(defer)
    near_uniform = same_work and pair_top_mass < _PAIR_UNIFORM_MASS
    if deferred and have_pair and (margin <= _MARGIN_BAND or near_uniform):
        return "near-twin-fired"
    if not (series_pinned or deferred):
        return "pin-thin"
    return "laya-match"


def miss_diagnostic(
    sess: GuessSession,
    *,
    target: str,
    trait_window: str = "ok",
    defer: str | None = None,
) -> dict[str, Any]:
    """Return the five miss fields, the pair sibling, and ``miss_label``.

    Margin and entropy come from ``sess.posterior()`` at this instant, which
    is why the bench snapshots before a rejected guess leaves the pool.
    Entropy is always over the top two probabilities. ``franchise_pair`` is
    ``same-work`` when those two share a real series, ``cross-work`` when
    they do not, and ``none`` when the pool has fewer than two. A cross-work
    pair still exports the entropy; only the near-uniform clause of the
    label requires same-work. ``trait_window`` and ``defer`` are the timing
    notes accumulated for the run, not a second channel.
    """
    ranked = sess.posterior()
    have_pair = len(ranked) >= 2
    if have_pair:
        (first, p1), (second, p2) = ranked[0], ranked[1]
        margin = p1 - p2
        entropy, pair_top = _pair_entropy(p1, p2)
        pair = "same-work" if _same_work(first, second) else "cross-work"
        same = pair == "same-work"
    else:
        margin = 0.0
        entropy = 0.0
        pair_top = 1.0
        pair = "none"
        same = False
    window = "truncated" if trait_window == "truncated" else "ok"
    defer_qid = _defer_qid(defer)
    in_pool = _target_in_shortlist(sess, target)
    pinned = _series_hard_pinned(sess)
    label = miss_label(
        target_in_shortlist=in_pool,
        margin=margin,
        pair_top_mass=pair_top,
        same_work=same,
        have_pair=have_pair,
        trait_window=window,
        defer=defer_qid,
        series_pinned=pinned,
    )
    return {
        "target_in_shortlist": in_pool,
        "top_two_posterior_margin": round(margin, 4),
        "franchise_match_entropy": round(entropy, 4),
        "franchise_pair": pair,
        "trait_window": window,
        "defer": defer_qid,
        "miss_label": label,
    }


_WOLF_LOOKS = ("look_animal_ears", "look_tail")


def look_pin_report(sess: GuessSession) -> dict[str, Any]:
    """Ask step and Lawrence−Holo soft-cast gap for the wolf look pins.

    ``never_asked`` means that series-split question was not emitted before
    the miss, which is the pin-timing failure. A positive soft-cast delta is
    fame still favoring Kraft Lawrence after that answer (the rebound). The
    two stay separate so a high-margin miss does not have to widen defer.
    """
    ask: dict[str, Any] = {qid: "never_asked" for qid in _WOLF_LOOKS}
    for step, row in enumerate(sess.asked, start=1):
        qid = row.get("qid")
        if qid in ask and ask[qid] == "never_asked":
            ask[qid] = step
    stored = getattr(sess, "soft_cast_delta", None) or {}
    delta: dict[str, Any] = {}
    for qid in _WOLF_LOOKS:
        if ask[qid] == "never_asked":
            delta[qid] = None
        else:
            delta[qid] = stored.get(qid)
    return {"look_ask": ask, "soft_cast_delta": delta}


def miss_pack(memories: Iterable[SessionMemory]) -> list[dict[str, Any]]:
    """Return one stable miss row per character, in the order given.

    Hits and games that never called ``record_miss`` are left out, so a
    5-pack JSON is only the misses.
    """
    return [memory.miss_row() for memory in memories if memory.miss]


@dataclass
class TurnRecord:
    """One bench turn: the question, the answer, and the pool it left."""

    turn: int
    stage: str
    qid: str
    question: str
    kind: str
    answer: str
    detail: str
    chips: list[dict[str, str]] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)
    residual: str = ""
    pool_size: int = 0
    top: list[dict[str, Any]] = field(default_factory=list)
    # This turn's timing notes only. The session keeps the run-long values.
    trait_window: str = ""
    defer: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return this turn as a JSON-ready mapping."""
        out = {
            "turn": self.turn,
            "stage": self.stage,
            "qid": self.qid,
            "question": self.question,
            "kind": self.kind,
            "answer": self.answer,
            "detail": self.detail,
            "chips": list(self.chips),
            "negated": list(self.negated),
            "residual": self.residual,
            "pool_size": self.pool_size,
            "top": list(self.top),
        }
        if self.trait_window:
            out["trait_window"] = self.trait_window
        if self.defer:
            out["defer"] = self.defer
        return out


@dataclass
class SessionMemory:
    """Per-session transcript a bench reuses when it answers as a target.

    ``traits`` is the known card (qid → yes/no/option key). ``answers`` is
    what this run has already said. Later turns read ``answers`` first so a
    repeated question cannot contradict the earlier reply.

    ``trait_window`` and ``defer`` accumulate ``timing.note`` fields from
    each turn payload. ``miss`` is set by ``record_miss`` when the game ends
    on a wrong guess or the turn cap.
    """

    target: str = ""
    seed: str = ""
    traits: dict[str, str] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    session_id: str = ""
    trait_window: str = "ok"
    defer: str | None = None
    miss: dict[str, Any] | None = None
    # Wolf look ask-step and soft-cast gap. Kept off ``MISS_KEYS`` so the
    # stable miss row does not change shape; ``export`` still dumps it.
    look_pins: dict[str, Any] | None = None

    def answer_for(self, question: dict[str, Any]) -> str:
        """Return the honest answer for ``question``.

        A qid already answered in this session is repeated. Otherwise the
        target card wins, then a forced opposite (female excludes male, a
        confirmed non-human species excludes human). An unknown yes/no is
        ``no`` and an unknown choice is ``other`` when that option exists:
        the bench must not invent a trait the card never stated.
        """
        qid = question.get("id") or ""
        if qid in self.answers:
            return self.answers[qid]
        if qid in self.traits:
            return self.traits[qid]
        implied = self._implied(qid)
        if implied:
            return implied
        if is_choice(question):
            options = question.get("options") or {}
            if "other" in options:
                return "other"
            return next(iter(options), "other")
        return "no"

    def _implied(self, qid: str) -> str:
        """Return an answer forced by a trait already on record, or ""."""
        known = {**self.traits, **self.answers}
        opposite = _GENDER_OPPOSITE.get(qid)
        if opposite and known.get(opposite) == "yes":
            return "no"
        if qid == "species_human":
            if any(key.startswith("species_") and key != "species_human" and known.get(key) == "yes"
                   for key in known):
                return "no"
        if qid.startswith("species_") and qid != "species_human" and known.get("species_human") == "yes":
            return "no"
        return ""

    def absorb_timing(self, raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Remember ``trait_window`` and ``defer`` from this turn's timing notes.

        A later turn's payload omits keys it did not note, so a truncated
        window sticks for the rest of the run and a defer qid is replaced
        only when a newer one was noted. Returns the fields seen this call.
        """
        fields = _timing_view(raw)
        if fields.get("trait_window") == "truncated":
            self.trait_window = "truncated"
        defer = _defer_qid(fields.get("defer"))
        if defer:
            self.defer = defer
        return fields

    def note(
        self,
        sess: GuessSession,
        *,
        answer: str = "",
        detail: str = "",
        timing: Mapping[str, Any] | None = None,
    ) -> TurnRecord:
        """Append a turn from ``sess`` after ``answer`` was applied.

        Chips are the soft-evidence rows, not the closed question. Negated
        chips are listed on their own so a post-mortem can see a "no" that
        never appeared as a button. Pool size and the top five names are the
        state after scoring. ``timing`` is the turn payload or its
        ``timing`` dict; notes already on the live trace are read too.
        """
        fields = self.absorb_timing(timing)
        pending = sess.asked[-1] if sess.asked else {}
        qid = pending.get("qid") or ""
        text = (detail or answer or "").strip()
        hits = free_text_trait_hits(text) if text else []
        chips = [{"qid": chip_id, "answer": polarity} for chip_id, polarity in hits]
        negated = [chip_id for chip_id, polarity in hits if polarity == "no"]
        residual = chip_residual(text) if text else ""
        ranked = sess.posterior()[:5]
        turn_defer = _defer_qid(fields.get("defer")) or ""
        record = TurnRecord(
            turn=sess.turn,
            stage=sess.stage,
            qid=qid,
            question=pending.get("text") or "",
            kind=pending.get("kind") or "yesno",
            answer=answer or (pending.get("answer") or ""),
            detail=detail or (pending.get("detail") or ""),
            chips=chips,
            negated=negated,
            residual=residual,
            pool_size=len(sess.alive_candidates()),
            top=[
                {"id": cand.id, "name": cand.name, "probability": round(prob, 4)}
                for cand, prob in ranked
            ],
            trait_window="truncated" if fields.get("trait_window") == "truncated" else "",
            defer=turn_defer,
        )
        if qid and record.answer:
            self.answers[qid] = record.answer
        for chip in chips:
            self.answers.setdefault(chip["qid"], chip["answer"])
        self.session_id = sess.id
        self.seed = sess.seed
        self.turns.append(record)
        return record

    def record_miss(
        self,
        sess: GuessSession,
        *,
        timing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Snapshot the miss diagnostic from ``sess`` as it stands now.

        Call this when the bench ends on a wrong guess or the turn cap,
        before ``submit_guess_result(False)`` drops that identity out of
        ``posterior``. The dict is what ``export`` and ``miss_pack`` keep.
        """
        self.absorb_timing(timing)
        self.look_pins = look_pin_report(sess)
        self.miss = miss_diagnostic(
            sess,
            target=self.target,
            trait_window=self.trait_window,
            defer=self.defer,
        )
        return dict(self.miss)

    def finish(
        self,
        sess: GuessSession,
        *,
        correct: bool,
        timing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Close the game and return ``export``.

        A miss snapshots the diagnostic first. A hit clears any earlier
        snapshot so a recovered round is not dumped as a miss.
        """
        if correct:
            self.absorb_timing(timing)
            self.miss = None
        else:
            self.record_miss(sess, timing=timing)
        return self.export()

    def miss_row(self) -> dict[str, Any]:
        """Return the stable per-character miss row, target first."""
        miss = self.miss or {}
        row: dict[str, Any] = {"target": self.target}
        for key in MISS_KEYS:
            row[key] = miss.get(key)
        return row

    def export(self) -> dict[str, Any]:
        """Return the session log a post-mortem can dump as JSON.

        ``miss`` is present only after ``record_miss``. Its stable keys are
        the five diagnostic fields, ``franchise_pair``, and ``miss_label``.
        ``look_ask`` and ``soft_cast_delta`` sit beside those: ask step of
        ``look_animal_ears`` and ``look_tail`` (or ``never_asked``), and the
        Lawrence−Holo soft-cast gap after each of those answers.
        """
        out = {
            "session_id": self.session_id,
            "target": self.target,
            "seed": self.seed,
            "traits": dict(self.traits),
            "answers": dict(self.answers),
            "turns": [turn.to_dict() for turn in self.turns],
        }
        if self.miss is not None:
            miss = {key: self.miss[key] for key in MISS_KEYS}
            if self.look_pins:
                miss["look_ask"] = self.look_pins["look_ask"]
                miss["soft_cast_delta"] = self.look_pins["soft_cast_delta"]
            out["miss"] = miss
        return out

    def dumps(self) -> str:
        """Return ``export`` as a stable JSON string."""
        return json.dumps(self.export(), indent=2, sort_keys=True)
