"""Derived, normalized player facts for one Nekomimi round.

The transcript and scored evidence remain authoritative. This module rebuilds
the compact memory view from them so button answers, typed details, and
rejections have one durable shape without becoming duplicate score evidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class SessionMemory:
    """Serializable compatibility view and normalized facts for a session."""

    confirmed_traits: dict[str, Any] = field(default_factory=dict)
    soft_dispreferred: list[str] = field(default_factory=list)
    rejected_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    player_details: list[str] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready memory while keeping the legacy summary fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMemory":
        """Load old and current memory files, ignoring fields from newer schemas."""
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def summary_prompt(self) -> str:
        """Format player facts for local tooling without including scraped profiles."""
        lines = []
        if self.confirmed_traits:
            lines.append("Confirmed facts: " + ", ".join(
                f"{key}: {value}" for key, value in self.confirmed_traits.items()))
        if self.player_details:
            lines.append("Player details: " + "; ".join(self.player_details))
        if self.soft_dispreferred:
            lines.append("Player answered NO to: " + ", ".join(self.soft_dispreferred[:8]))
        if self.rejected_hypotheses:
            lines.append("Already rejected: " + ", ".join(
                f"{row.get('name', '')} ({row.get('series', '')})"
                for row in self.rejected_hypotheses))
        return "\n".join(lines)


def _button_fact(asked: dict[str, Any], turn: int) -> dict[str, Any] | None:
    """Normalize one settled button answer and retain its offered choices."""
    answer = asked.get("answer")
    qid = str(asked.get("qid") or "")
    if not qid or answer is None:
        return None
    options = asked.get("options") or {}
    offered = [
        {
            "key": key,
            "label": str(option.get("label") or key),
            "series_key": str(option.get("series_key") or ""),
            "fact": str(option.get("fact") or ""),
        }
        for key, option in options.items()
        if isinstance(option, dict)
    ]
    selected = options.get(answer) if isinstance(options, dict) else None
    if not isinstance(selected, dict):
        selected = None
    if selected is not None and answer == "other":
        value = "none of the offered choices"
        polarity = "none_of_offered"
    elif selected is not None:
        value = str(selected.get("label") or answer)
        polarity = "selected"
    elif answer == "yes":
        value, polarity = str(asked.get("text") or qid), "positive"
    elif answer == "no":
        value, polarity = str(asked.get("text") or qid), "negative"
    else:
        value, polarity = str(answer), "answered"
    return {
        "fact_id": f"button:{qid}:{turn}",
        "question_id": qid,
        "value": value,
        "answer": str(answer),
        "polarity": polarity,
        "source": "button",
        "turn": turn,
        "explicit": True,
        "question": str(asked.get("text") or ""),
        "offered": offered,
    }


def working_memory(sess: Any) -> list[dict[str, Any]]:
    """Derive ordered facts from the seed, answered turns, details, and rejections."""
    facts: list[dict[str, Any]] = []
    if sess.seed.strip():
        facts.append({
            "fact_id": "seed:0", "question_id": "seed", "value": sess.seed.strip(),
            "answer": "", "polarity": "asserted", "source": "seed", "turn": 0,
            "explicit": True, "question": "", "offered": [],
        })
    for index, asked in enumerate(sess.asked, start=1):
        turn = int(asked.get("turn") or index)
        button = _button_fact(asked, turn)
        if button is not None:
            facts.append(button)
        detail = " ".join(str(asked.get("detail") or "").split())
        if detail:
            facts.append({
                "fact_id": f"detail:{turn}", "question_id": str(asked.get("qid") or ""),
                "value": detail, "answer": "", "polarity": "asserted",
                "source": "player_detail", "turn": turn, "explicit": True,
                "question": str(asked.get("text") or ""), "offered": [],
            })
    known = {fact["fact_id"] for fact in facts}
    for question, answer in sess.evidence.values():
        if not question.get("soft_chip"):
            continue
        fact_id = str(question.get("fact_id") or f"chip:{question.get('id', '')}")
        if fact_id in known:
            continue
        turn = int(question.get("turn") or 0)
        facts.append({
            "fact_id": fact_id, "question_id": str(question.get("id") or ""),
            "value": str(question.get("text") or question.get("id") or ""),
            "answer": str(answer), "polarity": "positive" if answer == "yes" else "negative",
            "source": "inferred_chip", "turn": turn, "explicit": False,
            "question": "", "offered": [],
        })
        known.add(fact_id)
    rejected = [
        row for row in (getattr(sess, "exclusion_log", []) or [])
        if row.get("reason") == "wrong_guess_identity"
    ]
    if not rejected:
        rejected = list(sess.memory.rejected_hypotheses)
    for index, row in enumerate(rejected):
        cid = str(row.get("candidate_id") or row.get("id") or index)
        facts.append({
            "fact_id": f"rejection:{cid}", "question_id": "identity_rejection",
            "value": str(row.get("name") or ""), "answer": "no",
            "polarity": "excluded", "source": "rejection",
            "turn": int(row.get("turn") or 0), "explicit": True,
            "question": "", "series": str(row.get("series") or ""), "offered": [],
        })
    return facts


def refresh_session_memory(sess: Any) -> list[dict[str, Any]]:
    """Refresh the compatibility fields from authoritative transcript state."""
    facts = working_memory(sess)
    memory = sess.memory
    memory.facts = facts
    memory.confirmed_traits = {
        fact["question_id"]: fact["answer"]
        for fact in facts if fact["source"] == "button"
    }
    memory.soft_dispreferred = list(dict.fromkeys(
        fact["question"] for fact in facts
        if fact["source"] == "button" and fact["polarity"] == "negative"
        and fact["question"]
    ))
    memory.rejected_hypotheses = [
        {
            "candidate_id": fact["fact_id"].removeprefix("rejection:"),
            "name": fact["value"], "series": fact.get("series", ""),
            "turn": fact["turn"],
        }
        for fact in facts if fact["source"] == "rejection"
    ]
    details = [fact["value"] for fact in facts if fact["source"] == "player_detail"]
    memory.player_details = list(dict.fromkeys([*memory.player_details, *details]))
    return facts


def _redact_rejected_names(text: str, names: list[str]) -> str:
    """Remove rejected identity names from player text before a proposal call."""
    clean = text
    for name in sorted({name.strip() for name in names if name.strip()},
                       key=len, reverse=True):
        clean = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", " ", clean, flags=re.I)
    return " ".join(clean.split()).strip(" ,;:-")


def proposal_facts(sess: Any, limit: int = 16) -> list[str]:
    """Build bounded player-only proposal input without rejected identity names."""
    facts = working_memory(sess)
    rejected_names = [fact["value"] for fact in facts if fact["source"] == "rejection"]
    selected: list[tuple[dict[str, Any], str]] = []
    for fact in facts:
        source = fact["source"]
        if source == "rejection":
            continue
        if source == "seed":
            line = fact["value"]
        elif source == "player_detail":
            line = fact["value"]
        elif source == "button":
            question = fact["question"].rstrip("?")
            if fact["polarity"] == "positive":
                line = f"Yes: {question}"
            elif fact["polarity"] == "negative":
                line = f"No: {question}"
            elif fact["polarity"] == "none_of_offered":
                options = ", ".join(option["label"] for option in fact["offered"]
                                     if option["key"] != "other")
                line = f"{question}: none of these offered choices ({options})"
            else:
                line = f"{question}: {fact['value']}"
        else:
            # Chips are inferred from the same typed text and would repeat it.
            continue
        clean = _redact_rejected_names(" ".join(line.split()), rejected_names)
        if clean and clean not in {line for _fact, line in selected}:
            selected.append((fact, clean))
    if len(selected) <= limit:
        return [line for _fact, line in selected]
    anchors = [(fact, line) for fact, line in selected
               if fact.get("question_id") in {"medium", "series", "series_2"}]
    anchored_ids = {fact["fact_id"] for fact, _line in anchors}
    remaining = [(fact, line) for fact, line in selected if fact["fact_id"] not in anchored_ids]
    recent = remaining[-max(0, limit - len(anchors)):]
    return [line for _fact, line in [*anchors, *recent]][:limit]
