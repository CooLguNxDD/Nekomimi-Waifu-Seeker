"""Guessing-session state: candidate pool, log-odds evidence, in-memory store.

The store is a plain process-local dict. That is deliberate and documented --
the web app runs as a single uvicorn process. Running more than one worker
would split sessions across workers; swap in Redis/SQLite before doing that.
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

SESSION_TTL_SECONDS = int(os.getenv("WAIFU_NEKOMINI_TTL", "1800"))
MAX_TURNS = int(os.getenv("WAIFU_NEKOMINI_MAX_TURNS", "20"))
MAX_GUESSES = int(os.getenv("WAIFU_NEKOMINI_MAX_GUESSES", "3"))

# A candidate this far below the leader in log-odds is out of the running.
ELIMINATION_MARGIN = 4.0
# Limit the readiness summary only. Evidence scoring evaluates every eligible
# candidate, independently of its current rank.
MAX_SCORED_CANDIDATES = int(os.getenv("WAIFU_NEKOMINI_SCORED", "10"))
# Famous characters are a real prior -- but a weak one, or a popular
# character would outrank a perfectly matching obscure one.
POPULARITY_WEIGHT = float(os.getenv("WAIFU_NEKOMINI_POPULARITY_WEIGHT", "0.22"))
POPULARITY_CAP = 1.5


def popularity_prior(popularity: int | float | None) -> float:
    """Log-odds bonus for fame, capped so evidence always wins."""
    if not popularity or popularity <= 0:
        return 0.0
    return min(POPULARITY_CAP, POPULARITY_WEIGHT * math.log10(1.0 + float(popularity)))


def _name_key(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def logit(p: float, floor: float = 0.02) -> float:
    p = min(max(p, floor), 1.0 - floor)
    return math.log(p / (1.0 - p))


@dataclass
class Candidate:
    id: str
    name: str
    series: str = ""
    medium: str = "unknown"
    blurb: str = ""
    tags: list[str] = field(default_factory=list)
    image_url: str | None = None
    source_url: str | None = None
    popularity: int = 0
    logodds: float = 0.0
    alive: bool = True

    @classmethod
    def from_search(cls, raw: dict[str, Any]) -> "Candidate":
        return cls(
            id=raw["id"],
            name=raw.get("name", ""),
            series=raw.get("series", ""),
            medium=raw.get("medium", "unknown"),
            blurb=raw.get("blurb", ""),
            tags=list(raw.get("tags") or []),
            image_url=raw.get("image_url"),
            source_url=raw.get("source_url"),
            popularity=int(raw.get("popularity") or 0),
        )

    def profile(self, budget: int = 600) -> str:
        """Compact profile, decisive facts first.

        ``laya.common.build_sequence`` lays the input out as
        ``[CLS] <type> instructions [SEP] [MASK] opts [SEP] state [SEP]`` -- the
        state goes last and is the first thing truncated, so identity leads
        and the prose description is what gets cut.
        """
        head = f"{self.name} ({self.series}) [{self.medium}]"
        # Mined tags can describe other people mentioned in the prose (Mario's
        # description mentions Princess Peach and Bowser). Do not present them
        # to the model as confirmed facts about this identity.
        room = max(0, budget - len(head) - 2)
        blurb = self.blurb[:room].strip()
        return f"{head}. {blurb}".strip()

    def public(self, probability: float | None = None) -> dict[str, Any]:
        out = {
            "id": self.id,
            "name": self.name,
            "series": self.series,
            "medium": self.medium,
            "blurb": self.blurb,
            "image_url": self.image_url,
            "source_url": self.source_url,
            "popularity": self.popularity,
        }
        if probability is not None:
            out["probability"] = round(probability, 4)
        return out


@dataclass
class GuessSession:
    id: str
    created: float
    updated: float
    seed: str = ""
    turn: int = 0
    stage: str = "asking"  # asking | guessing | done
    asked: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    rejected: set[str] = field(default_factory=set)
    guesses_made: int = 0
    pending_guess: str | None = None
    winner: str | None = None
    notes: list[str] = field(default_factory=list)
    laya_used: bool = False
    evidence: dict[str, tuple[dict[str, Any], str]] = field(default_factory=dict)
    match_cache: dict[tuple[str, str], float] = field(default_factory=dict)

    # -- candidate pool ------------------------------------------------
    def alive_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.alive and c.id not in self.rejected]

    def by_id(self, cid: str) -> Candidate | None:
        for c in self.candidates:
            if c.id == cid:
                return c
        return None

    def add_candidates(self, raws: list[dict[str, Any]]) -> int:
        known = {c.id for c in self.candidates}
        # The same character turns up under several URLs (wiki, MAL, fandom),
        # and duplicates split their own posterior mass.
        seen_names = {_name_key(c.name) for c in self.candidates}
        added = 0
        live = self.alive_candidates()
        base = sorted(c.logodds for c in live)[len(live) // 2] if live else 0.0
        for raw in raws:
            if not raw.get("id") or raw["id"] in known or raw["id"] in self.rejected:
                continue
            key = _name_key(raw.get("name", ""))
            if not key or key in seen_names:
                continue
            seen_names.add(key)
            cand = Candidate.from_search(raw)
            # New arrivals start at the median of the live pool so they are not
            # instantly eliminated by evidence they were never scored against,
            # plus a small bonus for fame.
            cand.logodds = base + popularity_prior(cand.popularity)
            self.candidates.append(cand)
            known.add(cand.id)
            added += 1
        return added

    def scoring_pool(self) -> list[Candidate]:
        """Compact leader summary for the readiness call, never an evidence gate."""
        live = sorted(self.alive_candidates(), key=lambda c: c.logodds, reverse=True)
        return live[:MAX_SCORED_CANDIDATES]

    def posterior(self) -> list[tuple[Candidate, float]]:
        live = self.alive_candidates()
        if not live:
            return []
        top = max(c.logodds for c in live)
        weights = [math.exp(c.logodds - top) for c in live]
        total = sum(weights) or 1.0
        pairs = [(c, w / total) for c, w in zip(live, weights)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def prune(self) -> int:
        live = self.alive_candidates()
        if len(live) <= 2:
            return 0
        top = max(c.logodds for c in live)
        dropped = 0
        for c in live:
            if top - c.logodds > ELIMINATION_MARGIN:
                c.alive = False
                dropped += 1
        return dropped

    # -- transcript ----------------------------------------------------
    def asked_ids(self) -> set[str]:
        return {a["qid"] for a in self.asked}

    def settled_categories(self) -> set[str]:
        """Categories where a yes already pins the answer down."""
        return {a["category"] for a in self.asked if a.get("answer") == "yes"}

    def history(self) -> list[dict[str, Any]]:
        return [
            {"question": a["text"], "answer": a["answer"], "detail": a.get("detail") or ""}
            for a in self.asked
        ]

    def touch(self) -> None:
        self.updated = time.time()


# --- store -----------------------------------------------------------
_STORE: dict[str, GuessSession] = {}
_STORE_LOCK = threading.Lock()


def new_session(seed: str = "") -> GuessSession:
    now = time.time()
    sess = GuessSession(id=uuid.uuid4().hex[:16], created=now, updated=now, seed=seed.strip())
    with _STORE_LOCK:
        _STORE[sess.id] = sess
    purge_expired()
    return sess


def get_session(session_id: str) -> GuessSession | None:
    with _STORE_LOCK:
        sess = _STORE.get(session_id)
    if sess is None:
        return None
    if time.time() - sess.updated > SESSION_TTL_SECONDS:
        drop_session(session_id)
        return None
    return sess


def drop_session(session_id: str) -> None:
    with _STORE_LOCK:
        _STORE.pop(session_id, None)


def purge_expired(now: float | None = None) -> int:
    now = time.time() if now is None else now
    with _STORE_LOCK:
        stale = [sid for sid, s in _STORE.items() if now - s.updated > SESSION_TTL_SECONDS]
        for sid in stale:
            _STORE.pop(sid, None)
    return len(stale)


def active_count() -> int:
    with _STORE_LOCK:
        return len(_STORE)
