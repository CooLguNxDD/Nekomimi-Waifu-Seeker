"""ACG (Anime / Comic / Game) trait question bank.

Laya cannot write questions -- it only picks among typed options. The wording
lives in ``question_bank.json`` (yes/no rows, choice rows, and ``trait_block``
templates). This module expands that file into the runtime shape. Dynamic
questions mined from search snippets use the same shape (see ``make_dynamic``).

Yes/no question shape (``_q``)::

    {
      "id": "medium_game",
      "text": "Is your character from a video game?",     # shown to the user
      "instructions": "...",                              # given to Laya per candidate
      "category": "medium",
      "tags_true": [...],   # candidate tags that imply the answer is yes
      "tags_false": [...],  # candidate tags that imply the answer is no
      "prior": 0.35,        # rough P(yes) across all ACG characters
      "kind": "yesno",
    }

Multiple-choice questions (``_choice``) carry ``"kind": "choice"`` and an
``options`` map; the player answers with an option key.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any

# A "detail" answer adds free text to the session constraints instead of
# scoring the current question, so it carries no evidence weight of its own.
ANSWER_WEIGHT: dict[str, float] = {"yes": 1.0, "no": -1.0, "detail": 0.0}
ANSWERS = tuple(ANSWER_WEIGHT)


_LEAD = re.compile(
    r"^(is|does|did|has|are|do)\s+(the\s+)?(character\s+)?"
    r"(described\s+)?(in\s+`candidate`|`candidate`)\s*",
    re.I,
)


def noul_criteria(instructions: str, detail: str | None = None) -> dict[str, str]:
    """Explicit true/false option text for a ``noul`` question.

    Without this Laya scores against its generic "yes, the statement holds",
    which is a much weaker target than a restatement of the actual claim.
    """
    clause = (detail or "").strip() or _LEAD.sub("", instructions.strip()).rstrip("?").strip()
    clause = clause[0].lower() + clause[1:] if clause else "the statement holds"
    # Framed rather than inflected: the clause can be adjectival ("female") or
    # verbal ("die at some point"), and "the character is die" reads as noise.
    return {
        "true": f"yes, this is true of the character: {clause}",
        "false": f"no, this is not true of the character: {clause}",
    }


def _q(
    qid: str,
    category: str,
    text: str,
    instructions: str,
    criteria_detail: str | None = None,
    tags_true: list[str] | None = None,
    tags_false: list[str] | None = None,
    prior: float = 0.5,
) -> dict[str, Any]:
    """Keep ``instructions`` short.

    Measured on the typed-decisions checkpoint, the same question asked with a
    140-character instruction scored Mario 0.38 for "is this a video game
    character"; asked in 50 characters it scored 0.71. Elaboration belongs in
    ``criteria_detail``, which only shapes the true/false option text.
    """
    return {
        "criteria": noul_criteria(instructions, criteria_detail),
        "id": qid,
        "category": category,
        "text": text,
        "instructions": instructions,
        "tags_true": tags_true or [],
        "tags_false": tags_false or [],
        "prior": prior,
        "dynamic": False,
        "kind": "yesno",
    }


def _choice(
    qid: str,
    category: str,
    text: str,
    instructions: str,
    options: list[tuple[str, str, str, list[str], str, float]],
    criteria_prefix: str = "the character has",
) -> dict[str, Any]:
    """A multiple-choice question, scored with one Laya ``choice`` per candidate.

    options: (key, user-facing label, Laya option text, candidate tags that
    imply this option, search keyword fact or "" for none, prior). The Laya
    option text reads "<criteria_prefix> <text>". Keep it to eight options at
    most: Laya is weak on wide choice sets and every option string counts
    against ``head_max_len``.
    """
    assert 2 <= len(options) <= 8, qid
    return {
        "kind": "choice",
        "id": qid,
        "category": category,
        "text": text,
        "instructions": instructions,
        "criteria": {key: f"{criteria_prefix} {crit}" for key, _, crit, _, _, _ in options},
        "options": {
            key: {"label": label, "tags": list(tags), "fact": fact, "prior": prior}
            for key, label, _, tags, fact, prior in options
        },
        # Kept for code that treats every question alike; choice evidence
        # never reads them.
        "tags_true": [],
        "tags_false": [],
        "prior": 0.5,
        "dynamic": False,
    }


def is_choice(question: dict[str, Any]) -> bool:
    return question.get("kind") == "choice"


def valid_answers(question: dict[str, Any]) -> tuple[str, ...]:
    """Answers accepted for ``question``: option keys for choice, else yes/no."""
    if is_choice(question):
        return (*question["options"], "detail")
    return ANSWERS


def _trait_block(
    category: str,
    prefix: str,
    items: list[tuple[str, str, str, float]],
    tag_prefix: str = "",
) -> list[dict[str, Any]]:
    """items: (slug, user-facing phrase, Laya instruction clause, prior).

    ``tag_prefix`` namespaces the tag so two blocks cannot share a slug -- a
    bare "blue" would otherwise satisfy both the hair and the eye question.
    """
    out = []
    for slug, label, clause, prior in items:
        out.append(
            _q(
                f"{prefix}_{slug}",
                category,
                f"Is your character {label}?",
                f"Is the character in `candidate` {clause}?",
                tags_true=[f"{tag_prefix}{slug}"],
                prior=prior,
            )
        )
    return out


MEDIUM_VALUES = ("anime", "manga", "comic", "game", "movie", "tv")

# Which candidate media each option of the "medium" question accepts. A pick
# is a hard fact: candidates of a known, non-accepted medium are removed.
# Movie and TV accept each other because franchises cross between them (Star
# Wars); only soft evidence separates those two. "other" accepts no known medium.
MEDIUM_ACCEPTS: dict[str, frozenset[str]] = {
    "anime": frozenset({"anime", "manga"}),
    "game": frozenset({"game"}),
    "comic": frozenset({"comic"}),
    "movie": frozenset({"movie", "tv"}),
    "tv": frozenset({"tv", "movie"}),
    "other": frozenset(),
}

MAX_SERIES_OPTIONS = 6


def series_question(qid: str, series: list[tuple[str, str]]) -> dict[str, Any]:
    """"Which series is your character from?" over the given series.

    ``series``: (series_key, display label) for up to ``MAX_SERIES_OPTIONS``
    series taken from the live candidates; "Another series" is appended. The
    wording is fixed here -- only the option labels come from search data, and
    the caller has already sanitised them.
    """
    # Prior 0.0: a confirmed series is as distinctive a search fact as a
    # typed detail, so it leads the search query.
    opts = [(f"s{i}", label, f"from {label}", [], label, 0.0)
            for i, (_, label) in enumerate(series[:MAX_SERIES_OPTIONS], start=1)]
    opts.append(("other", "Another series", "from a different series than these", [], "", 0.3))
    question = _choice(qid, "series", "Which series is your character from?",
                       "Which series is the character in `candidate` from?", opts,
                       criteria_prefix="the character is")
    for (key, _), opt_key in zip(series, question["options"]):
        question["options"][opt_key]["series_key"] = key
    question["series_question"] = True
    return question


def clue_question(qid: str, clues: str) -> dict[str, Any]:
    """Evaluate a user's free-text search clue against one retrieved identity."""
    question = _q(qid, "clue", "Does your character match these clues?",
                  "Does `candidate` match the supplied `clues`?")
    question["clues"] = clues
    return question


def _expand_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one JSON template row into runtime questions.

    ``trait_block`` fills the shared hair/job/personality wording so those
    rows stay a slug plus a clause. Choice and yes/no rows pass through
    ``_choice`` / ``_q``, which still build Laya's criteria text.
    """
    kind = entry["template"]
    if kind == "trait_block":
        items = [(i["slug"], i["label"], i["clause"], i["prior"]) for i in entry["items"]]
        return _trait_block(entry["category"], entry["prefix"], items,
                            tag_prefix=entry.get("tag_prefix", ""))
    if kind == "choice":
        options = [
            (o["key"], o["label"], o["criteria"], o.get("tags") or [], o.get("fact") or "", o["prior"])
            for o in entry["options"]
        ]
        return [_choice(entry["id"], entry["category"], entry["text"], entry["instructions"],
                        options, criteria_prefix=entry.get("criteria_prefix", "the character has"))]
    if kind == "yesno":
        return [_q(entry["id"], entry["category"], entry["text"], entry["instructions"],
                   criteria_detail=entry.get("criteria_detail"),
                   tags_true=entry.get("tags_true"), tags_false=entry.get("tags_false"),
                   prior=entry.get("prior", 0.5))]
    raise ValueError(f"unknown question template {kind!r}")


def load_question_bank() -> list[dict[str, Any]]:
    """Load ``question_bank.json`` and expand it into the runtime question list.

    The file is the editable template. Criteria strings are derived here so a
    wording edit cannot drift from the true/false option text Laya scores.
    """
    raw = json.loads(files("waifu_engine.nekomimi").joinpath("question_bank.json").read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for entry in raw["questions"]:
        out.extend(_expand_entry(entry))
    ids = [q["id"] for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("question_bank.json has duplicate ids")
    return out


QUESTION_BANK: list[dict[str, Any]] = load_question_bank()
QUESTIONS_BY_ID: dict[str, dict[str, Any]] = {q["id"]: q for q in QUESTION_BANK}
CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(q["category"] for q in QUESTION_BANK))




def make_dynamic(slug: str, phrase: str) -> dict[str, Any]:
    """Build a question from a trait phrase mined out of search snippets."""
    q = _q(
        f"dyn_{slug}",
        "mined",
        f'Is your character associated with "{phrase}"?',
        f'Is the character in `candidate` associated with "{phrase}"?',
        tags_true=[slug],
        prior=0.3,
    )
    q["dynamic"] = True
    return q
