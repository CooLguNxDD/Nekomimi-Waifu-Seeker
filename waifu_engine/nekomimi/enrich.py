"""Map player text onto bank traits with the enrich.yml templates.

The templates are a fixed phrase list. They do not invent a person, write a
question, or call Laya or the query LLM. A hit is soft evidence so the engine
can skip a question the player already answered without wiping the pool.
"""

from __future__ import annotations

import re
from typing import Any

from .lexicon import ENRICH_TEMPLATES
from .session import Candidate
from .traits import QUESTIONS_BY_ID, _negated_before, halo_is_title, yesno_visual_likelihood

# A clear hit or contradiction has to move rank, and it has to stay inside
# the band that cannot floor a thin blurb. log(0.88/0.32) is about one nat:
# enough to pass a modest fame gap, not enough to delete the other row.
# Silence stays at one half, same as an unmatched chip.
_ENRICH_HIT = 0.88
_ENRICH_MISS = 0.32
_ENRICH_UNKNOWN = 0.5


def _boundary(marker: str) -> re.Pattern[str]:
    """Whole-phrase pattern for one template marker.

    A substring match is how ``he`` used to tag every blurb that said ``the``.
    The same boundary is required on player text.
    """
    body = re.escape(" ".join(marker.lower().split()))
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.I)


def _compiled_templates() -> tuple[dict[str, Any], ...]:
    """Template rows with their markers compiled once."""
    rows: list[dict[str, Any]] = []
    for row in ENRICH_TEMPLATES:
        rows.append({**row, "patterns": tuple(_boundary(marker) for marker in row["markers"])})
    return tuple(rows)


_COMPILED = _compiled_templates()


def _option_shown(option: dict[str, Any], cand: Candidate) -> bool:
    """Whether ``cand`` shows one choice option's tags or search fact."""
    tags = set(option.get("tags") or ())
    if tags and tags & set(cand.tags or ()):
        return True
    fact = " ".join((option.get("fact") or "").split())
    if len(fact) < 3:
        return False
    blob = f"{cand.name} {cand.blurb}"
    return _boundary(fact).search(blob) is not None


def _row_answer(row: dict[str, Any], raw: str) -> str:
    """The answer one template gives ``raw``, or ``""`` when it has none.

    Every occurrence of every marker is checked. A negated first mention
    ("not blonde as a child, blonde hair now") used to end the scan and hide
    the affirmative one. A choice needs an affirmative hit; a yes/no is yes
    on any affirmative hit and no when every hit is negated. "Halo" as the
    game title ("a character from Halo") is not a halo.
    """
    negated = False
    for pattern in row["patterns"]:
        for match in pattern.finditer(raw):
            if row["qid"] == "look_halo" and halo_is_title(raw, match.start(), match.end()):
                continue
            if not _negated_before(raw, match.start()):
                return "yes" if row["kind"] == "yesno" else row["answer"]
            negated = True
    if negated and row["kind"] == "yesno":
        return "no"
    return ""


def enrich_hits(text: str) -> list[tuple[str, str]]:
    """``(question id, answer)`` for each template that matches ``text``.

    Two different answers for one question (blonde and black in the same
    sentence) are dropped so the question can still be asked. A negated
    choice ("not blonde hair") is not the blonde option. A negated yes/no
    ("no wings") is stored as no. Unknown bank ids are omitted: a template
    must not invent a question.
    """
    raw = " ".join((text or "").split())
    if not raw:
        return []
    found: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in _COMPILED:
        qid = row["qid"]
        if qid in ambiguous or qid not in QUESTIONS_BY_ID:
            continue
        answer = _row_answer(row, raw)
        if not answer:
            continue
        previous = found.get(qid)
        if previous is not None and previous != answer:
            ambiguous.add(qid)
            found.pop(qid, None)
        elif previous is None:
            found[qid] = answer
    return [(qid, answer) for qid, answer in found.items() if qid not in ambiguous]


def soft_enrich_likelihood(question: dict[str, Any], answer: str, cand: Candidate) -> float:
    """P(the enriched answer) for one candidate, kept inside the soft band.

    Choice misses used to take the full option likelihood and log(0.02),
    which floored everyone who was not already tagged. A silent profile stays
    at one half. A profile that shows a different option drops to the miss
    band. Yes/no uses the same band: the sharp 0.10 visual miss stays on the
    clue path, where a stated combination is supposed to be decisive.
    """
    if question.get("kind") == "choice":
        options = question.get("options") or {}
        picked = options.get(answer)
        if not isinstance(picked, dict):
            return _ENRICH_UNKNOWN
        if _option_shown(picked, cand):
            return _ENRICH_HIT
        for key, option in options.items():
            if key == answer or not isinstance(option, dict):
                continue
            if _option_shown(option, cand):
                return _ENRICH_MISS
        return _ENRICH_UNKNOWN
    visual = yesno_visual_likelihood(question.get("tags_true"), cand.blurb, cand.tags)
    if answer == "no":
        if visual is None:
            return _ENRICH_UNKNOWN
        if visual >= 0.9:
            return _ENRICH_MISS
        if visual <= 0.15:
            return _ENRICH_HIT
        return _ENRICH_UNKNOWN
    if visual is None:
        return _ENRICH_UNKNOWN
    if visual >= 0.9:
        return _ENRICH_HIT
    if visual <= 0.15:
        return _ENRICH_MISS
    return _ENRICH_UNKNOWN
