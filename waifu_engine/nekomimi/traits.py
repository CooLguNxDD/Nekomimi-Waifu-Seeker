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

# Accept sets live in lexicon/medium.yml. ``other`` is ``filter: false``: an
# empty set, which ``engine._eliminate_by_medium`` treats as "no hard filter".
from .lexicon import MEDIUM_ACCEPTS, MEDIUM_VALUES

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
    """items: (slug, user-facing phrase, Laya criteria clause, prior).

    The instruction stays the short label. A 140-character instruction on
    typed-decisions scored a clear case at 0.38; the same question in about
    50 characters scored 0.71. The clause is ``criteria_detail`` only.
    ``tag_prefix`` namespaces the tag so two blocks cannot share a slug -- a
    bare "blue" would otherwise satisfy both the hair and the eye question.
    """
    out = []
    for slug, label, clause, prior in items:
        # A parenthetical is for the player ("two eye colours"). Laya's
        # instruction stays the head phrase; the clause is the option text.
        head = label.split("(", 1)[0].strip()
        out.append(
            _q(
                f"{prefix}_{slug}",
                category,
                f"Is your character {label}?",
                f"Is the character in `candidate` {head}?",
                criteria_detail=clause,
                tags_true=[f"{tag_prefix}{slug}"],
                prior=prior,
            )
        )
    return out


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


# Player phrases for traits that are rare in combination. Slugs match question
# tags and ``web_search.TRAIT_PATTERNS``. Wings are detected separately: the
# word also appears on wing-shaped halos.
_VISUAL_PHRASES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("pink", re.compile(r"\b(?:light\s+|pale\s+|bright\s+|dark\s+)?pink[\s-]+hair(?:ed)?\b", re.I),
     "pink hair"),
    ("halo", re.compile(r"\bhalos?\b", re.I), "halo"),
    ("wings", re.compile(r"\bwings\b|\bwinged\b", re.I), "wings"),
    ("horns", re.compile(r"\bhorns?\b|\bhorned\b", re.I), "horns"),
)
_VISUAL_YESNO = frozenset({"halo", "wings", "horns"})

_SENTENCE = re.compile(r"[^.!?\n]+")
_WINGS_WORD = re.compile(r"\bwings\b", re.I)
_WINGS_NEG = re.compile(
    r"\b(?:no|not|without|lacking|lacks|lack)\b(?:\W+\w+){0,3}\W+\bwings\b",
    re.I,
)
# Colour and size words are not enough: "small wings" in a halo sentence is
# still the halo. Angel, feathered, a pair, or wings on the back are the body.
# The back qualifier is matched in either order: "on her back are wings" and
# "wings on her back". The second used to miss, and a halo in that same
# sentence then classified real wings as decoration.
_BODY_WING = re.compile(
    r"\b(?:angel|feathered|feather|bat|bird|fairy|dragon|demon|insect)\s+wings\b"
    r"|\bpair of\b(?:\W+\w+){0,4}\W+\bwings\b"
    r"|\bwings\b(?:\W+\w+){0,6}\W+\b(?:sprout|sprouting|grow|growing)\b"
    r"|\bon (?:her|his|their) back\b(?:\W+\w+){0,8}\W+\bwings\b"
    r"|\bwings\b(?:\W+\w+){0,6}\W+\bon (?:her|his|their) back\b",
    re.I,
)
# A few words in front of a player phrase. "no wings" / "without a halo" must
# not be read as a request for that trait.
_CLUE_NEG = re.compile(
    r"\b(?:no|not|without|lacking|lacks|lack|doesn't|does\s+not|isn't|is\s+not|don't)\b"
    r"(?:\W+\w+){0,3}\W*$",
    re.I,
)
_HALO_DECOR = re.compile(r"\b(?:halo|halos|winged halo|heart with wings)\b", re.I)
_APPEARANCE = re.compile(
    r"\b(?:hair|haired|eyes|eyed|halo|halos|wings|horns|horned|blonde|redhead)\b",
    re.I,
)
# A one-line snippet is not an appearance description. Absence is only evidence
# once the profile is long enough to have mentioned the trait.
_APPEARANCE_MIN = 80


def _sentences(text: str) -> list[str]:
    """Split ``text`` on sentence boundaries so a halo line is not a wing line."""
    parts = [p.strip() for p in _SENTENCE.findall(text or "") if p.strip()]
    return parts or ([text.strip()] if (text or "").strip() else [])


def _sentence_has_body_wings(sentence: str) -> bool:
    """Whether ``sentence`` describes wings on the body, not a halo's shape."""
    if not _WINGS_WORD.search(sentence) or _WINGS_NEG.search(sentence):
        return False
    if _BODY_WING.search(sentence):
        return True
    if _HALO_DECOR.search(sentence):
        return False
    return True


def has_body_wings(text: str) -> bool:
    """Whether ``text`` gives the character wings of their own.

    Fandom blurbs call a halo "a heart with wings". That is not the angel
    wings on Mika's back, and treating it as a yes tied her with Hanae.
    """
    return any(_sentence_has_body_wings(s) for s in _sentences(text))


def _appearance_bearing(text: str) -> bool:
    """Whether ``text`` is long enough, and visual enough, to judge a missing trait."""
    return len(text or "") >= _APPEARANCE_MIN and bool(_APPEARANCE.search(text or ""))


def _trait_hit(slug: str, blurb: str, tags: set[str]) -> bool:
    """Whether the profile positively shows ``slug``.

    A ``wings`` tag mined from a halo sentence loses to the blurb. An empty
    blurb still trusts the tag, because there is no text to contradict it.
    """
    text = blurb or ""
    if slug == "wings":
        if has_body_wings(text):
            return True
        if text.strip():
            return False
        return "wings" in tags
    for name, pattern, _search in _VISUAL_PHRASES:
        if name == slug:
            return bool(pattern.search(text)) or slug in tags
    return slug in tags


def _trait_observed(slug: str, blurb: str, tags: set[str]) -> bool | None:
    """True, False, or None when the profile never describes appearance.

    None keeps a short or plot-only blurb from being punished for a trait it
    simply does not mention.
    """
    if _trait_hit(slug, blurb, tags):
        return True
    if not _appearance_bearing(blurb):
        return None
    return False


def _negated_before(text: str, start: int) -> bool:
    """Whether the words just before ``start`` negate the phrase that begins there."""
    window = text[max(0, start - 48):start]
    return bool(_CLUE_NEG.search(window))


def _visual_mentions(text: str) -> list[tuple[str, str, bool]]:
    """``(slug, search phrase, negated)`` for each visual trait named in ``text``.

    Negation is part of the clue. "pink hair and no wings" names both traits,
    and the wings half is a request that the profile lack them.
    """
    raw = text or ""
    found = []
    for slug, pattern, search in _VISUAL_PHRASES:
        match = pattern.search(raw)
        if match:
            found.append((slug, search, _negated_before(raw, match.start())))
    return found


def _clue_requirements(text: str) -> list[tuple[str, bool]]:
    """``(slug, wanted)`` for player text. ``wanted`` is False when they negated it.

    "no wings" is a requirement that the profile lack wings, not a request
    for them. Scoring the word as a positive trait reversed the clue.
    """
    return [(slug, not negated) for slug, _search, negated in _visual_mentions(text)]


def visual_residual(text: str) -> str:
    """Return ``text`` with pink-hair, halo, wings and horns phrases removed.

    Those slugs already score through ``clue_likelihood``. A sentence that
    also said "white dress" used to take only that path, so the dress never
    reached the mild overlap clue and did not move the posterior.
    """
    raw = text or ""
    spans: list[tuple[int, int]] = []
    for _slug, pattern, _search in _VISUAL_PHRASES:
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


# Seed phrases that should pull a bank question forward. Hair colour is one
# question; halo, wings and horns are separate so a yes to one does not skip
# the others.
_APPEARANCE_QUESTIONS = (
    ("pink hair", "hair_color"),
    ("halo", "look_halo"),
    ("wings", "look_wings"),
    ("horns", "look_horns"),
)


def appearance_question_ids(text: str) -> list[str]:
    """Bank question ids for the visual traits named in ``text``, in table order."""
    phrases = set(visual_search_phrases(text))
    return [qid for phrase, qid in _APPEARANCE_QUESTIONS if phrase in phrases]


def visual_search_phrases(text: str) -> list[str]:
    """Search words for the visual traits the player asked for.

    Kept beside a confirmed series so "Blue Archive" is not searched alone
    after the player already said pink hair, a halo and wings. A negated
    trait is left out: search engines treat "no wings" as a search for wings.
    """
    return [search for _slug, search, negated in _visual_mentions(text) if not negated]


def mined_visual_slugs(text: str) -> list[str]:
    """Halo, horns and body-wing slugs actually described in ``text``."""
    tags: set[str] = set()
    if _trait_hit("halo", text, tags):
        tags.add("halo")
    if _trait_hit("horns", text, tags):
        tags.add("horns")
    if has_body_wings(text):
        tags.add("wings")
    return [slug for slug in ("halo", "wings", "horns") if slug in tags]


def clue_likelihood(clues: str, blurb: str, tags: list[str] | set[str] | None = None) -> float | None:
    """P(profile matches a free-text clue), or None when it cannot be judged.

    A soft noul stays near 0.6 for every Trinity student whose blurb shares a
    few of the words. One missing trait of a stated combination has to cost
    more than a fame prior (about 1.3 log-odds) or the popular lookalike wins.
    A profile that never describes appearance returns None so the noul stands.
    """
    reqs = _clue_requirements(clues)
    if not reqs:
        return None
    have = set(tags or ())
    statuses: list[bool | None] = []
    for slug, wanted in reqs:
        observed = _trait_observed(slug, blurb, have)
        # None stays unknown. A negated trait matches when the profile lacks it.
        statuses.append(None if observed is None else observed == wanted)
    if not any(status is not None for status in statuses):
        return None
    hits = sum(status is True for status in statuses)
    misses = sum(status is False for status in statuses)
    if misses == 0 and hits == len(reqs):
        return 0.96 if len(reqs) >= 2 else 0.86
    if misses == 0:
        return None
    if len(reqs) >= 2:
        return 0.04 + 0.10 * (hits / len(reqs))
    return 0.22


# Soft-chip vocabulary lives in ``chips``. Re-exported so existing
# ``from .traits import free_text_trait_hits`` call sites keep working.
from .chips import chip_pattern, chip_residual, free_text_trait_hits, free_text_trait_ids


def clue_overlap_likelihood(clues: str, blurb: str) -> float:
    """P(yes) from shared words, kept inside the heuristic band.

    A chip like "white dress" has no bank trait. Mentioning it should nudge a
    blurb that says the same words. A miss stays near a half: the old path
    applied a noul of 0 and floored the rest of the pool.
    """
    words = list(dict.fromkeys(re.findall(r"[a-z]{4,}", (clues or "").lower())))
    if not words:
        return 0.5
    blob = (blurb or "").lower()
    hits = sum(1 for word in words if re.search(rf"\b{re.escape(word)}\b", blob))
    if hits == 0:
        return 0.45
    return min(0.6, 0.45 + 0.15 * (hits / len(words)))


def chip_likelihood(qid: str, blurb: str, tags: list[str] | set[str] | None = None) -> float:
    """P(yes) for one typed chip. A miss stays at one half.

    A free-text "yes" used to take Laya's noul even when that noul was ~0 for
    the whole pool ("angel", "white dress"). One unmatched word then floored
    every candidate who still fit the earlier facts. Hits rise; misses do not.
    """
    question = QUESTIONS_BY_ID.get(qid) or {}
    if set(question.get("tags_true") or ()) & set(tags or ()):
        return 0.82
    pattern = chip_pattern(qid)
    if pattern is not None and pattern.search(blurb or ""):
        return 0.82
    return 0.5


def yesno_visual_likelihood(
    tags_true: list[str] | None, blurb: str, tags: list[str] | set[str] | None = None,
) -> float | None:
    """P(yes) for a halo, wings or horns question, or None if the profile is silent.

    These three are separate questions because a single "horns or wings or
    halo" yes matched every Blue Archive student. The likelihood is left
    sharp on purpose: the usual [0.4, 0.6] tag cap cannot separate them.
    """
    slugs = [slug for slug in (tags_true or []) if slug in _VISUAL_YESNO]
    if len(slugs) != 1 or len(tags_true or []) != 1:
        return None
    status = _trait_observed(slugs[0], blurb, set(tags or ()))
    if status is True:
        return 0.92
    if status is False:
        return 0.10
    return None


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
