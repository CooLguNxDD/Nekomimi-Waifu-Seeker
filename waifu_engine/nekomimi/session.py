"""Guessing-session state: candidate pool, log-odds evidence, in-memory store.

The store is a plain process-local dict. That is deliberate and documented --
the web app runs as a single uvicorn process. Running more than one worker
would split sessions across workers; swap in Redis/SQLite before doing that.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import uuid
import ast
from dataclasses import dataclass, field
from typing import Any
import json
import dataclasses
from .memory import SessionMemory

from ..names import (
    already_seen,
    canonical_name,
    identity_id,
    name_keys,
    same_character,
    same_series,
    series_key,
)

SESSION_TTL_SECONDS = int(os.getenv("WAIFU_NEKOMINI_TTL", "1800"))
SESSION_SCHEMA_VERSION = 2
MAX_TURNS = int(os.getenv("WAIFU_NEKOMINI_MAX_TURNS", "20"))
MAX_GUESSES = int(os.getenv("WAIFU_NEKOMINI_MAX_GUESSES", "3"))
# Extra questions granted after each wrong guess. Without them every guess
# after the first fired back to back at the cap, on the same evidence: the
# Arue -> Yunyun -> Funifura run named posterior ranks 1-3 and never asked.
RECOVERY_TURNS = int(os.getenv("WAIFU_NEKOMINI_RECOVERY_TURNS", "2"))

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




def _may_absorb(existing: Candidate, raw: dict[str, Any]) -> bool:
    """Whether a duplicate hit may fold its tags into ``existing``.

    A shared lexicon id still folds when the series labels differ (Diana
    Prince under Wonder Woman, the title row under DC Comics). Two known
    series that are not the same work do not: Kingdom Hearts Aqua must not
    take a KonoSuba blurb. An unknown series does not contradict.
    """
    iid = identity_id(raw.get("name") or "")
    if iid and iid == identity_id(existing.name):
        return True
    left, right = series_key(existing.series), series_key(raw.get("series") or "")
    if not left or not right:
        return True
    return same_series(existing.series, raw.get("series") or "")


def answer_label(asked: dict[str, Any]) -> str:
    """The player's answer as they saw it: an option label for choice questions."""
    answer = asked.get("answer")
    option = (asked.get("options") or {}).get(answer)
    return option["label"] if option else (answer or "")


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
        """The candidate as sent to the page (scraped fields; the page escapes them)."""
        out = {
            "id": self.id,
            "name": self.name,
            # "Web result" is a source's placeholder for "series unknown".
            "series": "" if self.series == "Web result" else self.series,
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
    schema_version: int = SESSION_SCHEMA_VERSION
    revision: int = 0
    evidence_revision: int = 0
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
    memory: SessionMemory = field(default_factory=SessionMemory)
    evidence: dict[str, tuple[dict[str, Any], str]] = field(default_factory=dict)
    match_cache: dict[tuple[str, str], float] = field(default_factory=dict)
    match_fingerprints: dict[tuple[str, str], str] = field(default_factory=dict)
    # Per-candidate option probabilities for multiple-choice questions.
    choice_cache: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    choice_fingerprints: dict[tuple[str, str], str] = field(default_factory=dict)
    # Last good query-LLM rewrite, reused until a newer one lands.
    llm_queries: list[str] = field(default_factory=list)
    # How many consecutive guess-checks this candidate has led. A crowded
    # pool often stalls a correct leader under the 0.80 bar; the streak is
    # what makes that lead commitable. See ``engine._should_guess``.
    leader_id: str = ""
    leader_streak: int = 0
    # Id of a rejected guess that still shares a work with the live pool.
    # The next turn asks one trait that separates them, then clears this.
    recovery_from: str = ""
    # Leader/runner ids already given one near-twin deferral. A second
    # series_split question for the same pair would burn the turns the first
    # answer was supposed to free. A different pair is not in this set.
    near_twin_pairs: set[tuple[str, str]] = field(default_factory=set)
    # Kraft Lawrence minus Holo, in soft-cast log-odds, snapshotted when that
    # look was the newest evidence. A later popularity absorb refreshes only
    # the newest look, so pin-timing and the fame rebound stay separable.
    soft_cast_delta: dict[str, float] = field(default_factory=dict)
    # Guesses the player rejected. Each one extends ``turn_cap`` so the next
    # guess can follow new evidence instead of the runner-up of the old.
    wrong_guesses: int = 0
    # Best Laya lookahead information gain from the latest question pick, in
    # bits, or None when Laya did not score the shortlist.
    last_eig: float | None = None
    # Laya ``match`` judgments made ahead of asking, keyed by (candidate,
    # question), with the answered-trait rows they were judged under. They
    # join ``match_cache`` only if that prefix is still current at answer
    # time, so a lookahead never stands in for a judgment on other evidence.
    lookahead: dict[tuple[str, str], tuple[tuple[tuple[Any, ...], ...], float]] = field(
        default_factory=dict)
    lookahead_fingerprints: dict[tuple[str, str], str] = field(default_factory=dict)
    # Log-odds each evidence row added per candidate on the latest rescore,
    # keyed by (candidate, question id); ``__prior__`` is popularity and
    # ``__adjust__`` the pins/priors applied after the answers. A miss report
    # read only the final posterior and could not say which answer sank the
    # target. See ``engine.evidence_breakdown``.
    contrib: dict[tuple[str, str], float] = field(default_factory=dict)
    # Full ranking after each answer: (turn, qid, answer, [(id, name, p)]).
    # A target that never led has no trace in the top-5 snapshots.
    rank_log: list[tuple[int, str, str, list[tuple[str, str, float]]]] = field(
        default_factory=list)
    # One row per guess, in order: id, name, turn, probability, and copies of
    # ``contrib`` and the ranking at that moment. A rejected guess leaves the
    # next rescore, so its evidence exists only in this snapshot.
    guess_log: list[dict[str, Any]] = field(default_factory=list)
    # Hard exclusions and identity merges are kept with the trace so a stored
    # but inactive target can be distinguished from a retrieval miss.
    exclusion_log: list[dict[str, Any]] = field(default_factory=list)
    merge_log: list[dict[str, Any]] = field(default_factory=list)
    # Last bounded Laya context per role; useful for explaining omissions.
    context_log: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Proposal state is durable for deduplication; workers themselves are
    # process-local and are reconciled against sources on each request.
    proposal_calls: int = 0
    proposal_fingerprint: str = ""
    proposal_revision: int = -1
    proposal_turn: int = -1
    proposal_pending: bool = False

    def turn_cap(self) -> int:
        """Question limit for this round: ``MAX_TURNS`` plus recovery turns per miss."""
        return MAX_TURNS + RECOVERY_TURNS * self.wrong_guesses

    # -- candidate pool ------------------------------------------------
    def alive_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.alive and c.id not in self.rejected]

    def by_id(self, cid: str) -> Candidate | None:
        for c in self.candidates:
            if c.id == cid:
                return c
        return None

    def _absorb_raw(self, cand: Candidate, raw: dict[str, Any]) -> None:
        """Fold a duplicate hit into ``cand`` so a second spelling keeps its tags.

        Diana Prince's page often carries the royalty tag that the Wonder
        Woman page lacks. Dropping the duplicate used to drop that evidence
        with it. The display name becomes the lexicon canonical spelling.
        """
        extra = [tag for tag in (raw.get("tags") or []) if tag not in cand.tags]
        if extra:
            cand.tags = [*cand.tags, *extra]
        cand.popularity = max(cand.popularity, int(raw.get("popularity") or 0))
        blurb = raw.get("blurb") or ""
        if len(blurb) > len(cand.blurb):
            cand.blurb = blurb
        if not cand.image_url and raw.get("image_url"):
            cand.image_url = raw.get("image_url")
        if not cand.source_url and raw.get("source_url"):
            cand.source_url = raw.get("source_url")
        series = raw.get("series") or ""
        if series and series not in {"Unknown", "Web result"} and cand.series in {"", "Unknown", "Web result"}:
            cand.series = series
        medium = raw.get("medium") or ""
        if medium and medium != "unknown" and cand.medium in {"", "unknown"}:
            cand.medium = medium
        canon = canonical_name(cand.name) or canonical_name(raw.get("name") or "")
        if canon:
            cand.name = canon

    def _absorb_other(self, keeper: Candidate, other: Candidate) -> None:
        """Fold a live alias row into ``keeper`` (tags, fame, blurb, links)."""
        self._absorb_raw(keeper, {
            "name": other.name,
            "tags": other.tags,
            "popularity": other.popularity,
            "blurb": other.blurb,
            "image_url": other.image_url,
            "source_url": other.source_url,
            "series": other.series,
            "medium": other.medium,
        })

    def collapse_identities(self) -> int:
        """Merge live rows that the alias table says are one person.

        Returns how many extra rows were folded in. Search returns Soryu,
        Sohryu, Shikinami and bare Langley as separate hits; each used to
        hold its own posterior, so Kyoko's single row outranked every fragment.
        A lone fragment is renamed to the canonical spelling.
        """
        groups: dict[str, list[Candidate]] = {}
        for cand in self.alive_candidates():
            iid = identity_id(cand.name)
            if iid:
                groups.setdefault(iid, []).append(cand)
        folded = 0
        for group in groups.values():
            canon = canonical_name(group[0].name)
            keeper = next((c for c in group if canon and c.name == canon), None)
            if keeper is None:
                keeper = max(group, key=lambda c: (c.popularity, len(c.name)))
            for other in group:
                if other is keeper:
                    continue
                self.merge_log.append({
                    "candidate_id": other.id,
                    "name": other.name,
                    "keeper_id": keeper.id,
                    "keeper_name": keeper.name,
                    "reason": "same_character_identity",
                    "turn": self.turn,
                })
                self._absorb_other(keeper, other)
                other.alive = False
                other.logodds = -math.inf
                folded += 1
            if canon:
                keeper.name = canon
        return folded

    def add_candidates(self, raws: list[dict[str, Any]]) -> int:
        """Add search hits not already in the pool; returns how many were added.

        A hit is a duplicate if its id or its name (either word order, a
        trailing series title, or a known alias) is already known, or it was
        rejected as a guess. Alias duplicates donate their tags to the row
        already in the pool. List, category, franchise and species pages are
        dropped so a roster or an "Evangelion" article cannot lead the pool.
        """
        known = {c.id for c in self.candidates}
        # The same character turns up under several URLs (wiki, MAL, fandom),
        # and duplicates split their own posterior mass. Names, not key
        # sets: "Link" absorbs "Link (The Legend of Zelda)", while
        # "Young Link" and a different work's "Aqua (other)" stay.
        from ..web_search import is_non_character

        seen_names = [c.name for c in self.candidates]
        added = 0
        live = self.alive_candidates()
        base = sorted(c.logodds for c in live)[len(live) // 2] if live else 0.0
        for raw in raws:
            if not raw.get("id") or raw["id"] in known or raw["id"] in self.rejected:
                continue
            name = raw.get("name", "")
            if not name_keys(name):
                self.exclusion_log.append({
                    "candidate_id": str(raw.get("id") or ""),
                    "name": str(name), "reason": "missing_character_name",
                    "turn": self.turn,
                })
                continue
            # List, category, franchise and species pages must not enter the
            # pool, or they lead the posterior and get guessed.
            if is_non_character(name, raw.get("source_url") or "", raw.get("blurb") or ""):
                self.exclusion_log.append({
                    "candidate_id": str(raw.get("id") or ""),
                    "name": str(name), "reason": "non_character_source",
                    "source_url": str(raw.get("source_url") or ""),
                    "turn": self.turn,
                })
                continue
            match = next((c for c in self.candidates if same_character(name, c.name)), None)
            if match is not None:
                # Bare "Aqua" matches "Aqua (KonoSuba)" by name. Absorb only
                # when they are one lexicon identity, or the series fields
                # do not name two works. Otherwise drop the hit.
                if not _may_absorb(match, raw):
                    self.exclusion_log.append({
                        "candidate_id": str(raw.get("id") or ""),
                        "name": str(name), "reason": "same_name_different_series",
                        "matched_id": match.id, "turn": self.turn,
                    })
                    continue
                if match.id in self.rejected or not match.alive:
                    iid = identity_id(name)
                    keeper = next(
                        (c for c in self.alive_candidates()
                         if iid and identity_id(c.name) == iid and c.id not in self.rejected),
                        None,
                    )
                    if keeper is not None and _may_absorb(keeper, raw):
                        self._absorb_raw(keeper, raw)
                        self.merge_log.append({
                            "candidate_id": str(raw.get("id") or ""),
                            "name": str(name), "keeper_id": keeper.id,
                            "keeper_name": keeper.name,
                            "reason": "live_identity_alias_absorbed", "turn": self.turn,
                        })
                    else:
                        self.exclusion_log.append({
                            "candidate_id": str(raw.get("id") or ""),
                            "name": str(name), "reason": "previously_excluded_duplicate",
                            "matched_id": match.id, "turn": self.turn,
                        })
                else:
                    self._absorb_raw(match, raw)
                    if name and name != match.name:
                        self.merge_log.append({
                            "candidate_id": str(raw.get("id") or ""),
                            "name": str(name), "keeper_id": match.id,
                            "keeper_name": match.name, "reason": "duplicate_search_hit",
                            "turn": self.turn,
                        })
                continue
            if already_seen(name, seen_names):
                continue
            seen_names.append(name)
            cand = Candidate.from_search(raw)
            canon = canonical_name(name)
            if canon:
                cand.name = canon
            # New arrivals start at the median of the live pool so they are not
            # instantly eliminated by evidence they were never scored against,
            # plus a small bonus for fame.
            cand.logodds = base + popularity_prior(cand.popularity)
            self.candidates.append(cand)
            known.add(cand.id)
            added += 1
        self.collapse_identities()
        return added

    def scoring_pool(self) -> list[Candidate]:
        """Compact leader summary for the readiness call, never an evidence gate."""
        live = sorted(self.alive_candidates(), key=lambda c: c.logodds, reverse=True)
        return live[:MAX_SCORED_CANDIDATES]

    def posterior(self) -> list[tuple[Candidate, float]]:
        """Live candidates by probability, with non-characters left out.

        A roster, franchise article, or species page that slipped into the
        pool must not be the guess and must not soak probability mass.
        """
        from ..web_search import is_non_character

        live = [
            c for c in self.alive_candidates()
            if not is_non_character(c.name, c.source_url or "", c.blurb)
        ]
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
            {"question": a["text"], "answer": answer_label(a), "detail": a.get("detail") or ""}
            for a in self.asked
        ]

    def touch(self) -> None:
        self.updated = time.time()


# --- store -----------------------------------------------------------
_STORE: dict[str, GuessSession] = {}
_STORE_LOCK = threading.Lock()
def _default_session_dir() -> str:
    """Place persistent sessions in the user's writable application data directory."""
    if os.name == "nt":
        data_root = os.getenv("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local",
        )
    elif sys.platform == "darwin":
        data_root = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        data_root = os.getenv("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share",
        )
    return os.path.join(data_root, "nekomimi-waifu-seeker", "sessions")


_SESSION_DIR_SETTING = os.getenv("WAIFU_SESSION_DIR")
if _SESSION_DIR_SETTING and not os.path.isabs(_SESSION_DIR_SETTING):
    _SESSION_DIR_SETTING = os.path.join(
        os.path.dirname(_default_session_dir()), _SESSION_DIR_SETTING,
    )
SESSION_DIR = os.path.abspath(
    _SESSION_DIR_SETTING or _default_session_dir()
)


def _session_path(session_id: str) -> str | None:
    """Return a safe persisted path, rejecting anything outside generated IDs."""
    if (not isinstance(session_id, str) or len(session_id) != 16
            or any(char not in "0123456789abcdef" for char in session_id)):
        return None
    return os.path.join(SESSION_DIR, f"{session_id}.json")

def _keyed_rows(mapping: dict[tuple[str, str], Any]) -> list[list[Any]]:
    """Encode tuple-keyed cache entries as portable JSON rows."""
    return [[key[0], key[1], value] for key, value in mapping.items()]


def _to_json_dict(sess: GuessSession) -> dict[str, Any]:
    """Encode every authoritative round field without discarding trace state."""
    d = dataclasses.asdict(sess)
    d["schema_version"] = SESSION_SCHEMA_VERSION
    d["rejected"] = sorted(sess.rejected)
    d["near_twin_pairs"] = [list(pair) for pair in sorted(sess.near_twin_pairs)]
    d["evidence"] = {
        qid: {"question": question, "answer": answer}
        for qid, (question, answer) in sess.evidence.items()
    }
    d["match_cache"] = _keyed_rows(sess.match_cache)
    d["choice_cache"] = _keyed_rows(sess.choice_cache)
    d["match_fingerprints"] = _keyed_rows(sess.match_fingerprints)
    d["choice_fingerprints"] = _keyed_rows(sess.choice_fingerprints)
    d["lookahead"] = [
        [cid, qid, [list(row) for row in value[0]], value[1]]
        for (cid, qid), value in sess.lookahead.items()
    ]
    d["contrib"] = _keyed_rows(sess.contrib)
    d["lookahead_fingerprints"] = _keyed_rows(sess.lookahead_fingerprints)
    d["rank_log"] = [
        [turn, qid, answer, [list(row) for row in ranked]]
        for turn, qid, answer, ranked in sess.rank_log
    ]
    d["guess_log"] = []
    for entry in sess.guess_log:
        row = dict(entry)
        row["contrib"] = _keyed_rows(entry.get("contrib") or {})
        row["ranked"] = [list(item) for item in entry.get("ranked") or []]
        d["guess_log"].append(row)
    return d


def _decode_keyed_rows(value: Any) -> dict[tuple[str, str], Any]:
    """Restore current row encoding and legacy tuple-string cache keys."""
    decoded: dict[tuple[str, str], Any] = {}
    if isinstance(value, list):
        for row in value:
            if isinstance(row, list) and len(row) == 3:
                decoded[(str(row[0]), str(row[1]))] = row[2]
        return decoded
    if not isinstance(value, dict):
        return decoded
    for key, item in value.items():
        try:
            pair = ast.literal_eval(key)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(pair, tuple) and len(pair) == 2:
            decoded[(str(pair[0]), str(pair[1]))] = item
    return decoded

def _from_json_dict(raw: dict[str, Any]) -> GuessSession:
    """Restore current or legacy session files while retaining round history."""
    d = dict(raw)
    stored_schema = int(d.get("schema_version") or 1)
    d["candidates"] = [Candidate(**candidate) for candidate in d.get("candidates", [])]
    d["memory"] = SessionMemory.from_dict(d.get("memory") or {})
    d["rejected"] = set(d.get("rejected") or [])
    d["near_twin_pairs"] = {tuple(pair) for pair in d.get("near_twin_pairs") or []}
    evidence: dict[str, tuple[dict[str, Any], str]] = {}
    for qid, value in (d.get("evidence") or {}).items():
        if isinstance(value, dict):
            question, answer = value.get("question"), value.get("answer")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            question, answer = value
        else:
            continue
        if isinstance(question, dict) and answer is not None:
            evidence[str(qid)] = (question, str(answer))
    d["evidence"] = evidence
    d["match_cache"] = _decode_keyed_rows(d.get("match_cache"))
    d["choice_cache"] = _decode_keyed_rows(d.get("choice_cache"))
    d["match_fingerprints"] = _decode_keyed_rows(d.get("match_fingerprints"))
    d["choice_fingerprints"] = _decode_keyed_rows(d.get("choice_fingerprints"))
    d["contrib"] = _decode_keyed_rows(d.get("contrib"))
    d["lookahead_fingerprints"] = _decode_keyed_rows(d.get("lookahead_fingerprints"))
    lookahead: dict[tuple[str, str], tuple[tuple[tuple[Any, ...], ...], float]] = {}
    for row in d.get("lookahead") or []:
        if isinstance(row, list) and len(row) == 4:
            cid, qid, rows, probability = row
            lookahead[(str(cid), str(qid))] = (tuple(tuple(item) for item in rows), float(probability))
    if not lookahead and isinstance(d.get("lookahead"), dict):
        # Legacy sessions used stringified tuple keys. Their inputs were not
        # fingerprinted, so do not trust those cached judgments after upgrade.
        lookahead = {}
    d["lookahead"] = lookahead
    if stored_schema < SESSION_SCHEMA_VERSION:
        # Legacy cache entries had no record of the prompt they were judged on.
        d["match_cache"] = {}
        d["choice_cache"] = {}
        d["lookahead"] = {}
        d["match_fingerprints"] = {}
        d["choice_fingerprints"] = {}
        d["lookahead_fingerprints"] = {}
    d["rank_log"] = [
        (int(row[0]), str(row[1]), str(row[2]), [tuple(item) for item in row[3]])
        for row in d.get("rank_log", []) if isinstance(row, list) and len(row) == 4
    ]
    for entry in d.get("guess_log", []):
        if isinstance(entry, dict):
            entry["contrib"] = _decode_keyed_rows(entry.get("contrib"))
            entry["ranked"] = [tuple(item) for item in entry.get("ranked") or []]
    d["guess_log"] = [entry for entry in d.get("guess_log", []) if isinstance(entry, dict)]
    d.setdefault("schema_version", 1)
    d.setdefault("revision", 0)
    d.setdefault("evidence_revision", len(evidence))
    d.setdefault("exclusion_log", [])
    d.setdefault("merge_log", [])
    d.setdefault("context_log", {})
    d.setdefault("proposal_calls", 0)
    d.setdefault("proposal_fingerprint", "")
    d.setdefault("proposal_revision", -1)
    d.setdefault("proposal_turn", -1)
    was_pending = bool(d.get("proposal_pending"))
    # A worker cannot survive a process restart. The pending question or guess
    # remains in ``asked``/``pending_guess`` and is restored below.
    d["proposal_pending"] = False
    if was_pending:
        d["proposal_fingerprint"] = ""
        d["proposal_revision"] = -1
        d["proposal_turn"] = -1
        d["proposal_calls"] = max(0, int(d.get("proposal_calls") or 0) - 1)
    import inspect
    valid_keys = set(inspect.signature(GuessSession).parameters)
    sess = GuessSession(**{key: value for key, value in d.items() if key in valid_keys})
    sess.schema_version = SESSION_SCHEMA_VERSION
    if stored_schema >= SESSION_SCHEMA_VERSION or sess.asked or sess.evidence:
        from .memory import refresh_session_memory
        refresh_session_memory(sess)
    return sess

def save_session(sess: GuessSession) -> None:
    """Atomically persist one complete transition before it is returned to the caller."""
    path = _session_path(sess.id)
    if path is None:
        raise ValueError("session id must be 16 lowercase hexadecimal characters")
    os.makedirs(SESSION_DIR, exist_ok=True)
    temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    d = _to_json_dict(sess)
    with _STORE_LOCK:
        try:
            with open(temp_path, "w", encoding="utf-8") as stream:
                json.dump(d, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            _STORE[sess.id] = sess
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def load_session(session_id: str) -> GuessSession | None:
    """Load one valid session, serializing first restore to preserve object identity."""
    path = _session_path(session_id)
    if path is None:
        return None
    with _STORE_LOCK:
        if session_id in _STORE:
            return _STORE[session_id]
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
            sess = _from_json_dict(data)
            if sess.id != session_id:
                return None
            # Keep restore under the store lock so simultaneous first reads
            # share this mutable object instead of forking one session.
            _STORE[session_id] = sess
            return sess
        except Exception as exc:
            print(f"Error loading session {session_id}: {exc}")
            return None

def new_session(seed: str = "") -> GuessSession:
    """Create and persist a new session, then purge expired sessions."""
    now = time.time()
    sess = GuessSession(id=uuid.uuid4().hex[:16], created=now, updated=now, seed=seed.strip())
    save_session(sess)
    purge_expired()
    return sess

def get_session(session_id: str) -> GuessSession | None:
    """Return a live session, removing its snapshot when its TTL has passed."""
    sess = load_session(session_id)
    if sess is None:
        return None
    if time.time() - sess.updated > SESSION_TTL_SECONDS:
        drop_session(session_id)
        return None
    return sess

def drop_session(session_id: str) -> None:
    """Remove a cached session and its file without allowing a concurrent restore."""
    path = _session_path(session_id)
    if path is None:
        return
    with _STORE_LOCK:
        _STORE.pop(session_id, None)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

def purge_expired(now: float | None = None) -> int:
    """Purge cached sessions by update time and files by modification time."""
    now = time.time() if now is None else now
    stale = 0
    with _STORE_LOCK:
        to_drop = [sid for sid, s in _STORE.items() if now - s.updated > SESSION_TTL_SECONDS]
        for sid in to_drop:
            _STORE.pop(sid, None)
        if os.path.exists(SESSION_DIR):
            for filename in os.listdir(SESSION_DIR):
                if filename.endswith(".json"):
                    path = os.path.join(SESSION_DIR, filename)
                    try:
                        if now - os.path.getmtime(path) > SESSION_TTL_SECONDS:
                            os.remove(path)
                            stale += 1
                    except OSError:
                        pass
    return stale + len(to_drop)

def active_count() -> int:
    with _STORE_LOCK:
        return len(_STORE)
