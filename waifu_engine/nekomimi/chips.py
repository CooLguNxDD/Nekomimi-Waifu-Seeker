"""Soft chips named by free-text player details.

A typed word such as "angel" or "not a princess" is evidence for one bank
question. It is not a hard filter and it does not answer a different closed
question. The table lives here so the trait bank does not own the vocabulary.
"""

from __future__ import annotations

import re

# Whole words only: "king" must not fire inside "kingdom", and "god" must not
# fire inside "goddess".
CHIP_TRAITS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("species_angel", re.compile(r"\b(?:archangel|angel)\b", re.I)),
    ("species_demon", re.compile(r"\b(?:demon|devil|oni|vampire)\b", re.I)),
    ("species_god", re.compile(r"\b(?:goddess|deity|god)\b", re.I)),
    ("species_robot", re.compile(r"\b(?:android|cyborg|robot)\b", re.I)),
    ("species_beast", re.compile(r"\b(?:beastkin|kemonomimi|nekomimi)\b", re.I)),
    ("species_alien", re.compile(r"\b(?:extraterrestrial|alien)\b", re.I)),
    ("species_undead", re.compile(r"\b(?:revenant|undead|ghost)\b", re.I)),
    ("job_royalty", re.compile(r"\b(?:princess|prince|queen|king|nobility|noble)\b", re.I)),
)
_CHIP_BY_ID = {qid: pattern for qid, pattern in CHIP_TRAITS}


def chip_pattern(qid: str) -> re.Pattern[str] | None:
    """The phrase pattern for one chip question, or None when it is not a chip."""
    return _CHIP_BY_ID.get(qid)


def free_text_trait_hits(text: str) -> list[tuple[str, str]]:
    """``(question id, "yes"|"no")`` for each chip named in ``text``.

    "not a demon" and "not a princess" are nos. The same short window as
    visual clues ("no wings") decides that, so a negated chip is not stored
    as a yes and does not boost the trait the player ruled out.
    """
    # Lazy: traits imports this module while it is still building the bank.
    from .traits import QUESTIONS_BY_ID, _negated_before

    raw = text or ""
    found: list[tuple[str, str]] = []
    for qid, pattern in CHIP_TRAITS:
        match = pattern.search(raw)
        if match is None or qid not in QUESTIONS_BY_ID:
            continue
        polarity = "no" if _negated_before(raw, match.start()) else "yes"
        found.append((qid, polarity))
    return found


def free_text_trait_ids(text: str) -> list[str]:
    """Bank question ids named by player text, in table order.

    "angel" on the human question is a species chip, not a vote against every
    human. Unknown words are omitted so they stay search text only. Negated
    chips are included; ``free_text_trait_hits`` carries the polarity.
    """
    return [qid for qid, _polarity in free_text_trait_hits(text)]


def chip_residual(text: str) -> str:
    """``text`` with recognized chip phrases removed.

    "princess in a white dress" still has to score "white dress". Leaving the
    chip words in the overlap clue would count royalty twice and, on a
    negation, would boost the trait the player denied.
    """
    raw = text or ""
    spans: list[tuple[int, int]] = []
    for _qid, pattern in CHIP_TRAITS:
        spans.extend((match.start(), match.end()) for match in pattern.finditer(raw))
    if not spans:
        return " ".join(raw.split())
    spans.sort()
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        parts.append(raw[cursor:start])
        cursor = end
    parts.append(raw[cursor:])
    return " ".join("".join(parts).split())
