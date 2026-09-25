"""Structured memory for offline honest-answer benches.

Empty-seed runs used to keep only the engine's ``asked`` list. A miss was
then a posterior with no record of which chip, negation, or free-text
residual produced it, and the next honest answer could contradict a trait
already given. This module snapshots each turn and answers from that record
plus a known target card. It does not start a server and it does not search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .chips import chip_residual, free_text_trait_hits
from .session import GuessSession
from .traits import is_choice

# A yes on one side forces the other. Recorded answers win over the card so
# a bench cannot flip gender after it has already said female.
_GENDER_OPPOSITE = {"gender_female": "gender_male", "gender_male": "gender_female"}


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

    def to_dict(self) -> dict[str, Any]:
        """Return this turn as a JSON-ready mapping."""
        return {
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


@dataclass
class SessionMemory:
    """Per-session transcript a bench reuses when it answers as a target.

    ``traits`` is the known card (qid → yes/no/option key). ``answers`` is
    what this run has already said. Later turns read ``answers`` first so a
    repeated question cannot contradict the earlier reply.
    """

    target: str = ""
    seed: str = ""
    traits: dict[str, str] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    session_id: str = ""

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

    def note(self, sess: GuessSession, *, answer: str = "", detail: str = "") -> TurnRecord:
        """Append a turn from ``sess`` after ``answer`` was applied.

        Chips are the soft-evidence rows, not the closed question. Negated
        chips are listed on their own so a post-mortem can see a "no" that
        never appeared as a button. Pool size and the top five names are the
        state after scoring.
        """
        pending = sess.asked[-1] if sess.asked else {}
        qid = pending.get("qid") or ""
        text = (detail or answer or "").strip()
        hits = free_text_trait_hits(text) if text else []
        chips = [{"qid": chip_id, "answer": polarity} for chip_id, polarity in hits]
        negated = [chip_id for chip_id, polarity in hits if polarity == "no"]
        residual = chip_residual(text) if text else ""
        ranked = sess.posterior()[:5]
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
        )
        if qid and record.answer:
            self.answers[qid] = record.answer
        for chip in chips:
            self.answers.setdefault(chip["qid"], chip["answer"])
        self.session_id = sess.id
        self.seed = sess.seed
        self.turns.append(record)
        return record

    def export(self) -> dict[str, Any]:
        """Return the session log a post-mortem can dump as JSON."""
        return {
            "session_id": self.session_id,
            "target": self.target,
            "seed": self.seed,
            "traits": dict(self.traits),
            "answers": dict(self.answers),
            "turns": [turn.to_dict() for turn in self.turns],
        }

    def dumps(self) -> str:
        """Return ``export`` as a stable JSON string."""
        return json.dumps(self.export(), indent=2, sort_keys=True)
