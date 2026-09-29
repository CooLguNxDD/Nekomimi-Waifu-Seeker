"""Bound the evidence sent to Laya while keeping its real inputs traceable.

Laya truncates the serialized state at its configured model window. These
builders budget against that same question head and tokenizer when loaded,
then fall back to a deterministic word-piece estimate in offline runs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

_TOKEN_PIECE = re.compile(r"[A-Za-z]+|[0-9]+|[^\s]")
_WORD = re.compile(r"[a-z0-9]+", re.I)


@dataclass(frozen=True)
class BoundedContext:
    """A bounded Laya state and IDs of facts included or omitted from it."""

    state: dict[str, Any]
    included_fact_ids: tuple[str, ...]
    omitted_fact_ids: tuple[str, ...]
    state_fingerprint: str
    model_fingerprint: str

    def input_fingerprint(self, question: dict[str, Any]) -> str:
        """Fingerprint the exact bounded state and one question head."""
        payload = {
            "state": self.state,
            "question": _question_signature(question),
            "model_fingerprint": self.model_fingerprint,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()


def _question_signature(question: dict[str, Any]) -> dict[str, Any]:
    """Keep every prompt field that can change the model's judgment."""
    return {
        "id": question.get("id"),
        "kind": question.get("kind", "yesno"),
        "instructions": question.get("instructions") or "",
        "criteria": question.get("criteria") or {},
        "options": question.get("options") or {},
        "clues": question.get("clues") or "",
    }


def _count(text: str, tokenizer: Any = None) -> int:
    """Count text with Laya's tokenizer, or a stable offline word-piece estimate."""
    if tokenizer is not None:
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            return len(encoded["input_ids"])
        except Exception:  # noqa: BLE001 - tokenizer wrappers differ by backend
            pass
    return sum(1 for _match in _TOKEN_PIECE.finditer(text or ""))


def _rendered_options(question: dict[str, Any]) -> list[str]:
    """Render the false/true or choice options in the order Laya receives them."""
    kind = question.get("kind", "yesno")
    if kind == "choice":
        options = question.get("options") or {}
        criteria = question.get("criteria") or {}
        return [
            key if criteria.get(key) in (None, "")
            else f"{key}: {criteria[key]}"
            for key in options
        ]
    criteria = question.get("criteria") or {}
    return [
        "false: " + str(criteria.get("false") or "no, the statement does not hold"),
        "true: " + str(criteria.get("true") or "yes, the statement holds"),
    ]


def _question_room(
    question: dict[str, Any], max_len: int, head_max_len: int, tokenizer: Any,
) -> int:
    """Mirror Laya's head packing to reserve the actual state-token room."""
    kind = "choice" if question.get("kind") == "choice" else "noul"
    head = _count(f"{kind} question: {question.get('instructions') or ''}", tokenizer)
    option_sizes = [1 + min(48, _count(" " + option, tokenizer))
                    for option in _rendered_options(question)]
    option_budget = head_max_len - sum(option_sizes)
    if option_budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_sizes)))
        option_sizes = [min(size, per_option) for size in option_sizes]
        option_budget = head_max_len - sum(option_sizes)
    head = min(head, max(8, option_budget))
    sequence_head = 1 + head + 1 + sum(option_sizes) + 1
    return max(0, max_len - sequence_head - 1)


def state_room(questions: list[dict[str, Any]], max_len: int,
               head_max_len: int, tokenizer: Any = None) -> int:
    """State-token room shared by the supplied question heads."""
    return min((_question_room(question, max_len, head_max_len, tokenizer)
                for question in questions), default=max_len)


def _state_size(state: dict[str, Any], tokenizer: Any) -> int:
    """Count the same JSON state string that Laya serializes before truncation."""
    return _count(json.dumps(state, ensure_ascii=False).replace("[MASK]", " "), tokenizer)


def _tokens(value: str) -> set[str]:
    """Normalize prompt words for relevance ranking without changing evidence."""
    return {word.lower() for word in _WORD.findall(value or "") if len(word) > 2}


def _relevance(question: dict[str, Any], facts: Iterable[dict[str, Any]]) -> set[str]:
    """Collect question and player-fact words used to order profile passages."""
    parts = [question.get("text") or "", question.get("instructions") or ""]
    parts.extend((question.get("criteria") or {}).values())
    parts.extend(str(fact.get("value") or "") for fact in facts)
    return _tokens(" ".join(str(part) for part in parts))


def candidate_profile(candidate: Any, question: dict[str, Any],
                      facts: Iterable[dict[str, Any]]) -> str:
    """Keep candidate identity first and the profile passage most relevant to this question."""
    series = f" ({candidate.series})" if candidate.series else ""
    head = f"{candidate.name}{series} [{candidate.medium}]."
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", candidate.blurb or "")
                 if part.strip()]
    if not sentences:
        return head
    words = _relevance(question, facts)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -len(words & _tokens(item[1])),
            item[0],
        ),
    )
    picked = [sentence for _index, sentence in ranked[:3]]
    return head + " " + " ".join(picked)


def _row_priority(qid: str, meta: dict[str, Any], question: dict[str, Any]) -> int:
    """Rank medium/series anchors and facts relevant to the current judgment first."""
    if qid == "medium":
        return 10000
    if qid.startswith("series"):
        return 9500
    current_category = question.get("category")
    category = meta.get("category")
    priority = 500 if current_category and category == current_category else 0
    wanted = set(question.get("tags_true") or ()) | set(question.get("tags_false") or ())
    if wanted & set(meta.get("tags") or ()):
        priority += 800
    priority += 100 if meta.get("source") == "button" else 0
    return priority + min(99, int(meta.get("turn") or 0))


def _fact_priority(fact: dict[str, Any], question: dict[str, Any]) -> int:
    """Rank seed, rejection, and recent player details for bounded context."""
    source = fact.get("source")
    base = {"seed": 1000, "rejection": 900, "player_detail": 600}.get(source, 0)
    words = _tokens((question.get("text") or "") + " " +
                    (question.get("instructions") or ""))
    return base + min(99, int(fact.get("turn") or 0)) + 200 * bool(
        words & _tokens(str(fact.get("value") or "")))


def _trim_profile(value: str, identity: str, tokenizer: Any, excess: int) -> str:
    """Remove trailing profile words while preserving the candidate identity."""
    if value == identity:
        return value
    words = value[len(identity):].strip().split()
    remove = max(1, excess + 2)
    return (identity + " " + " ".join(words[:-remove])).strip() if len(words) > remove else identity


def build_bounded_context(
    state: dict[str, Any],
    questions: list[dict[str, Any]],
    *,
    max_len: int,
    head_max_len: int,
    tokenizer: Any = None,
    row_meta: dict[str, dict[str, Any]] | None = None,
    row_identity: dict[str, str] | None = None,
    working_identity: dict[str, str] | None = None,
    protected_profiles: dict[str, str] | None = None,
    state_fingerprint_context: dict[str, Any] | None = None,
) -> BoundedContext:
    """Fit state to the tightest question while reporting every omitted fact ID."""
    packed = copy.deepcopy(state)
    row_meta = row_meta or {}
    row_identity = row_identity or {}
    working_identity = working_identity or {}
    protected_profiles = protected_profiles or {}
    head_questions = questions or [{"instructions": "", "kind": "yesno"}]
    room = min(_question_room(q, max_len, head_max_len, tokenizer) for q in head_questions)
    all_row_ids = [row_identity.get(str(row[0]), str(row[0]))
                   for row in (packed.get("answered_traits") or {}).get("rows") or []]
    all_working_ids = [working_identity.get(str(fact.get("id") or ""),
                                             str(fact.get("id") or ""))
                       for fact in packed.get("working_facts") or []]
    all_choice_ids = [str(fact.get("id") or "")
                      for fact in packed.get("choice_context") or []]
    priority_question = head_questions[0]

    def over_budget() -> bool:
        """Whether the current serialized state exceeds the shared head's room."""
        return _state_size(packed, tokenizer) > room

    # Candidate identity stays present. Trim profile prose before deciding that
    # a durable answer has to be omitted from a narrow state window.
    for name in reversed(list((packed.get("characters") or {}).keys())):
        while over_budget():
            characters = packed.get("characters") or {}
            identity = protected_profiles.get(name, name)
            old = characters.get(name, "")
            if not old or old == identity:
                break
            trimmed = _trim_profile(old, identity, tokenizer, _state_size(packed, tokenizer) - room)
            if trimmed == old:
                break
            characters[name] = trimmed
    if over_budget() and isinstance(packed.get("candidate"), str):
        identity = protected_profiles.get("candidate", packed["candidate"].split(". ", 1)[0] + ".")
        while over_budget() and packed["candidate"] != identity:
            old = packed["candidate"]
            packed["candidate"] = _trim_profile(
                old, identity, tokenizer, _state_size(packed, tokenizer) - room)
            if packed["candidate"] == old:
                break

    # History and redundant prose are expendable before scored answers.
    while over_budget() and packed.get("answer_history"):
        packed["answer_history"].pop(0)
    while over_budget() and packed.get("confirmed_facts"):
        packed["confirmed_facts"].pop(0)
    while over_budget() and packed.get("working_facts"):
        facts = packed["working_facts"]
        index = min(range(len(facts)), key=lambda i: (
            _fact_priority(facts[i], priority_question), i,
        ))
        facts.pop(index)
    while over_budget() and (packed.get("answered_traits") or {}).get("rows"):
        rows = packed["answered_traits"]["rows"]
        index = min(range(len(rows)), key=lambda i: (
            max(_row_priority(str(rows[i][0]), row_meta.get(str(rows[i][0]), {}), question)
                for question in head_questions), i,
        ))
        rows.pop(index)
    while over_budget() and packed.get("choice_context"):
        packed["choice_context"].pop(0)
    while over_budget() and packed.get("characters"):
        characters = packed["characters"]
        name = next(reversed(characters))
        identity = protected_profiles.get(name, name)
        old = characters[name]
        if old == identity:
            if len(characters) <= 1:
                break
            characters.pop(name)
        else:
            characters[name] = _trim_profile(
                old, identity, tokenizer, _state_size(packed, tokenizer) - room)
    if over_budget() and isinstance(packed.get("candidate"), str):
        identity = protected_profiles.get("candidate", packed["candidate"].split(". ", 1)[0] + ".")
        packed["candidate"] = _trim_profile(
            packed["candidate"], identity, tokenizer, _state_size(packed, tokenizer) - room)
    if over_budget():
        # Only impossible question heads can reach here; keep the decision goal
        # and compact IDs while dropping optional profile prose.
        for key in ("answer_history", "confirmed_facts", "working_facts", "choice_context"):
            packed.pop(key, None)
        rows = (packed.get("answered_traits") or {}).get("rows")
        while over_budget() and rows:
            rows.pop(0)

    kept_rows = {str(row[0]) for row in
                 (packed.get("answered_traits") or {}).get("rows") or []}
    kept_facts = {str(fact.get("id") or "") for fact in packed.get("working_facts") or []}
    kept_choices = {str(fact.get("id") or "") for fact in packed.get("choice_context") or []}
    included = [fact_id for qid, fact_id in zip(
        [str(row[0]) for row in (state.get("answered_traits") or {}).get("rows") or []],
        all_row_ids,
    ) if qid in kept_rows]
    included.extend(fact_id for fact_id in all_working_ids if fact_id in kept_facts)
    included.extend(fact_id for fact_id in all_choice_ids if fact_id in kept_choices)
    included_set = set(included)
    omitted = [fact_id for fact_id in [*all_row_ids, *all_working_ids, *all_choice_ids]
               if fact_id and fact_id not in included_set]
    model_context = state_fingerprint_context or {}
    model_fingerprint = hashlib.sha256(
        json.dumps(model_context, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    fingerprint_input = {
        "state": packed,
        "questions": [_question_signature(q) for q in head_questions],
        "max_len": max_len,
        "head_max_len": head_max_len,
        "context": model_context,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return BoundedContext(
        packed, tuple(included), tuple(omitted), fingerprint, model_fingerprint,
    )
