"""Character sources, best first.

Order: Playwright HTML indexes → Wikipedia + AniList (structured APIs) →
DuckDuckGo fill → Playwright enrich of the top N. DuckDuckGo stays the
long-tail fallback -- it was the primary in v1 and it is why the pool
filled up with trope pages.
"""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .. import timing
from ..names import name_keys, plain_duplicate
from ..nekomimi.lexicon import POPULAR_CATEGORIES, SEARCH_SUFFIX as _MEDIUM_SUFFIX
from . import _http, anilist, gemini, wikipedia

__all__ = ["anilist", "gemini", "wikipedia", "find_candidates", "normalize_name"]


def normalize_name(name: str) -> str:
    """Letters and digits of ``name``, lower-cased. Kept for callers; matching
    uses ``names.name_keys``, which also equates the two name orders."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


# Search suffixes: lexicon/medium.yml ``search_suffix``. ``_queries`` still
# decides the order the strings are tried.


MAX_QUERIES = 5


def _queries(
    constraints: list[str],
    medium_hint: str | None,
    focus: list[str] | None = None,
    rewritten: list[str] | None = None,
    pin: str | None = None,
) -> list[str]:
    """Search strings, best first.

    ``rewritten`` (from the optional query LLM) goes first verbatim, then the
    ``focus`` facts Laya ranked most distinctive, then the template groups.
    Without either, this is exactly the template list. ``pin`` is a resolved
    work that every template group keeps.
    """
    templates = _template_queries(constraints, medium_hint, pin=pin)
    if not focus and not rewritten:
        return templates
    suffix = _MEDIUM_SUFFIX.get(medium_hint or "", "character")
    lead = [" ".join(q.split())[:200] for q in (rewritten or []) if q and q.strip()]
    if focus:
        lead.append(f"{' '.join(dict.fromkeys(f[:80] for f in focus))[:200]} {suffix}")
    return list(dict.fromkeys([*lead, *templates]))[:MAX_QUERIES]


def _fact_in_pin(fact: str, pin: str) -> bool:
    """Whether ``fact`` is already a whole phrase inside ``pin``.

    Substring checks dropped "red" out of a pin that merely contained those
    letters. The phrase has to sit on word boundaries.
    """
    fact_l = " ".join(fact.lower().split())
    pin_l = " ".join(pin.lower().split())
    if not fact_l:
        return True
    return fact_l == pin_l or f" {fact_l} " in f" {pin_l} " or pin_l.startswith(fact_l + " ")


def _template_queries(
    constraints: list[str], medium_hint: str | None, pin: str | None = None,
) -> list[str]:
    """Positive-fact queries. A resolved work stays in every group.

    The narrow retry used to be the newest fact alone ("long-haired
    character", "female character"). After the series was only the first
    fact, that group left the work and filled the pool with other franchises
    and later mantle-holders. ``pin`` is that work (and a personal alias);
    the narrow retry is the pin itself, not the newest trait.
    """
    facts = []
    for c in constraints:
        c = (c or "").strip()
        if not c or re.match(r"(?:not true:|the character is not\b)", c, re.I):
            continue
        c = re.sub(r"^(?:The character (?:is|does|has)|(?:Is|Does|Has|Did) your character)\s+",
                   "", c, flags=re.I).rstrip("?")
        if c and c not in facts:
            facts.append(c[:80])
    suffix = _MEDIUM_SUFFIX.get(medium_hint or "", "character")
    pinned = " ".join((pin or "").split())[:160]
    if pinned:
        facts = [fact for fact in facts if not _fact_in_pin(fact, pinned)]
    if not facts and not pinned:
        return [f"fictional {suffix}"]
    if pinned:
        newest = facts[-1] if facts else ""
        tail = facts[-3:]
        groups = [[pinned, *tail], [pinned, newest] if newest else [pinned], [pinned]]
    else:
        # No resolved work: still retry the newest clue alone when the first
        # fact over-constrains the search ("Italian plumber" without "Nintendo").
        groups = [[facts[0], *facts[-3:]], [facts[0], facts[-1]], [facts[-1]]]
    return list(dict.fromkeys(
        f"{' '.join(part for part in dict.fromkeys(group) if part)[:200]} {suffix}"
        for group in groups
    ))


# --- background fills ---------------------------------------------------
#
# DuckDuckGo and Gemini are the slow sources. With a ``background_key`` (the
# session id) they run on their own worker thread while the player reads the
# next question; their hits are handed back to the next search for that key.


class _Background:
    """One slow source run off the request thread, results parked per key.

    ``max_pending`` bounds queued-or-running jobs across all sessions; a job
    beyond it is dropped, not queued, because a queued search would run long
    after its turn and keep its query data alive.
    """

    def __init__(self, name: str, pending_env: str, pending_default: int, workers: int = 1):
        """``name`` labels threads and log lines; the cap comes from ``pending_env``."""
        self.name = name
        self.pending_env = pending_env
        self.pending_default = pending_default
        self.workers = workers
        self.lock = threading.Lock()
        self.executor: ThreadPoolExecutor | None = None
        self.pending: set[str] = set()
        self.ready: dict[str, list[dict[str, Any]]] = {}

    def max_pending(self) -> int:
        """The configured cap on queued-or-running jobs (env, else default)."""
        try:
            return max(0, int(os.getenv(self.pending_env, str(self.pending_default))))
        except ValueError:
            return self.pending_default

    def start(self, key: str, fn: Callable[..., None], *args: Any) -> bool:
        """Queue ``fn(key, *args)`` unless ``key`` has a job or the queue is full."""
        with self.lock:
            if key in self.pending or len(self.pending) >= self.max_pending():
                return False
            if self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.workers, thread_name_prefix=f"{self.name}-fill")
            self.pending.add(key)
        try:
            self.executor.submit(fn, key, *args)
        except Exception:  # noqa: BLE001 - e.g. interpreter shutting down
            with self.lock:
                self.pending.discard(key)
            return False
        return True

    def finish(self, key: str, hits: list[dict[str, Any]], t0: float, errors: list[str]) -> None:
        """Park ``hits`` for ``key``, clear its pending flag and log the run."""
        with self.lock:
            self.pending.discard(key)
            if len(self.ready) >= _BG_MAX_KEYS:
                self.ready.pop(next(iter(self.ready)))
            self.ready.setdefault(key, []).extend(hits)
        if timing._log_on():
            timing.log.info("%s background key=%s %.0fms found=%d%s",
                            self.name, key[:8], (time.perf_counter() - t0) * 1000, len(hits),
                            f" err={errors[0][:120]}" if errors else "")

    def take(self, key: str) -> list[dict[str, Any]]:
        """Hits a finished job left for ``key`` (consumed once)."""
        with self.lock:
            return self.ready.pop(key, [])

    def is_pending(self, key: str) -> bool:
        """Whether ``key`` has a job queued or running."""
        with self.lock:
            return key in self.pending


_BG_MAX_KEYS = 256
# New candidates one search may take from lexicon coverage queries. The
# first query is the character's own name, so three is enough to land them
# without crowding out the player's own search.
_COVERAGE_MAX = 3
# One worker each: parallel requests only make DuckDuckGo throttle harder, and
# Gemini calls are billed.
_DDG_BG = _Background("ddg", "WAIFU_DDG_BG_MAX_PENDING", 4)
_GEMINI_BG = _Background("gemini", "WAIFU_GEMINI_BG_MAX_PENDING", 2)
# Names kept for the DuckDuckGo fill (tests and callers use them).
_BG_LOCK = _DDG_BG.lock
_BG_PENDING = _DDG_BG.pending
_BG_READY = _DDG_BG.ready


def _bg_max_pending() -> int:
    """``WAIFU_DDG_BG_MAX_PENDING``: DuckDuckGo fills queued or running at once."""
    return _DDG_BG.max_pending()


def _ddg_background_on() -> bool:
    return os.getenv("WAIFU_DDG_BACKGROUND", "1").strip().lower() in {"1", "true", "yes"}


def _bg_run(key: str, queries: list[str], limit: int) -> None:
    """Background DuckDuckGo job: capped fill, parked for the next search."""
    from .. import web_search

    t0 = time.perf_counter()
    errors: list[str] = []
    try:
        hits = web_search.ddg_quick(queries, limit, errors=errors)
    except Exception as exc:  # noqa: BLE001 - ddg_quick should not raise; be sure
        hits, errors = [], [str(exc)]
    _DDG_BG.finish(key, hits, t0, errors)


def _bg_start(key: str, queries: list[str], limit: int) -> bool:
    """Queue a background DuckDuckGo fill unless ``key`` has one or the queue is full."""
    return _DDG_BG.start(key, _bg_run, list(queries), limit)


def _gemini_run(key: str, facts: list[str], medium_hint: str | None, limit: int) -> None:
    """Background Gemini job: one grounded search, parked for the next search."""
    t0 = time.perf_counter()
    errors: list[str] = []
    hits = gemini.search_characters(facts, medium_hint, limit, errors=errors)
    _GEMINI_BG.finish(key, hits, t0, errors)


def take_background(key: str) -> list[dict[str, Any]]:
    """DuckDuckGo hits a finished background fill left for ``key`` (consumed once)."""
    return _DDG_BG.take(key)


def background_pending(key: str) -> bool:
    """Whether a DuckDuckGo fill for ``key`` is queued or running."""
    return _DDG_BG.is_pending(key)


def _gemini_background_on() -> bool:
    """``WAIFU_GEMINI_BACKGROUND`` (default on): run Gemini off the request thread."""
    return os.getenv("WAIFU_GEMINI_BACKGROUND", "1").strip().lower() in {"1", "true", "yes"}


# --- popular pool ----------------------------------------------------------
#
# With nothing specific to search for (no seed, no typed detail, no LLM query),
# queries like "female video game character" match no character name and only
# list articles in full text. Start from popular characters instead and let the
# questions narrow them. Live sources only -- never catalog.json.

# Popular-category titles: lexicon/categories.yml ``popular``. Live Wikipedia
# categories only; this path never reads catalog.json.


def popular_limit() -> int:
    """``WAIFU_POPULAR_POOL``: popular candidates kept in play at most (each one
    costs a Laya ``match`` call per answer)."""
    try:
        return max(0, int(os.getenv("WAIFU_POPULAR_POOL", "24")))
    except ValueError:
        return 24


def _popular_from_wikipedia(medium: str, n: int) -> list[dict[str, Any]]:
    per = max(1, n // len(POPULAR_CATEGORIES[medium]))
    titles: list[str] = []
    for cat in POPULAR_CATEGORIES[medium]:
        titles += wikipedia.category_members(cat, limit=per * 2)[: per * 2]
    pages = [c for c in wikipedia.pages_by_title(titles[:40])
             if c.get("medium") in (medium, "unknown")]
    pages.sort(key=lambda c: c.get("popularity") or 0, reverse=True)
    return pages[:n]


def popular_characters(medium_hint: str | None, n: int) -> list[dict[str, Any]]:
    """Popular characters for a cold start, filtered by medium once known."""
    if n <= 0:
        return []
    if medium_hint in ("anime", "manga"):
        return anilist.top_characters(per_page=min(50, n))[:n]
    if medium_hint in POPULAR_CATEGORIES:
        return _popular_from_wikipedia(medium_hint, n)
    # Medium unknown: mostly anime/manga (AniList ranks by favourites), plus a
    # slice of every other medium so the first medium question has a real split.
    others = [m for m in POPULAR_CATEGORIES]
    share = max(1, n // (2 * len(others)))
    head = max(0, n - share * len(others))
    out = anilist.top_characters(per_page=min(50, head))[:head] if head else []
    for medium in others:
        out += _popular_from_wikipedia(medium, share)
    return out


def _trait_clue(facts: list[str]) -> str:
    """Visual phrases named across ``facts``, or "" when there is no combination.

    One trait is not enough to call a Wikipedia hit junk. Pink hair plus a
    halo plus wings is the combination that name search cannot retrieve.
    """
    from ..nekomimi.traits import visual_search_phrases

    phrases: list[str] = []
    for fact in facts:
        for phrase in visual_search_phrases(fact):
            if phrase not in phrases:
                phrases.append(phrase)
    if len(phrases) < 2:
        return ""
    return " ".join(phrases)


def _hit_shows_traits(cand: dict[str, Any], clue: str) -> bool:
    """Whether ``cand`` clearly has every trait in ``clue``."""
    from ..nekomimi.traits import clue_likelihood

    if not clue:
        return False
    text = f"{cand.get('blurb') or ''} {cand.get('name') or ''}"
    score = clue_likelihood(clue, text, cand.get("tags"))
    return score is not None and score >= 0.9


def _gemini_inline(
    found: dict[str, dict[str, Any]],
    exclude_names: set[str] | list[str],
    facts: list[str],
    forced: bool | None,
) -> bool:
    """Whether Gemini should run on this turn instead of the background worker.

    ``forced=True`` is the engine saying no alive candidate shows the seed.
    ``forced=False`` means one does, so a junk page must not schedule another
    billed call. With no opinion, a trait seed ignores ``exclude_names``:
    storing "The Saint" used to set that set and push Mika to the next turn.
    """
    if forced is True:
        return True
    if forced is False:
        return not found and not exclude_names
    clue = _trait_clue(facts)
    if clue:
        return not any(_hit_shows_traits(cand, clue) for cand in found.values())
    return not found and not exclude_names


def _anilist_plan(
    constraints: list[str], pin: str | None, queries: list[str], limit: int,
) -> tuple[list[str], int, bool]:
    """Name-search strings, page size, and whether to stop once ``limit`` is full.

    AniList searches character names and sorts by favourites. A trait sentence
    returned unrelated leads (Ken Kaneki for an Evangelion clue) and one fat
    page filled the limit before the next title name was requested. Headliner
    names are a few small pages; without them the full-text queries still run.
    """
    from .. import web_search

    names = web_search.title_character_queries([pin or "", *constraints])
    if names:
        return names[:4], min(3, max(1, limit)), False
    return list(queries), limit, True


def find_candidates(
    constraints: list[str],
    medium_hint: str | None = None,
    limit: int = 12,
    exclude_names: set[str] | None = None,
    use_ddg: bool = True,
    focus: list[str] | None = None,
    rewritten: list[str] | None = None,
    background_key: str | None = None,
    ddg_gate: Callable[[], bool] | None = None,
    specific: bool = True,
    pool_size: int | None = None,
    gemini_inline: bool | None = None,
    pin: str | None = None,
) -> list[dict[str, Any]]:
    """Merge candidates from every source, best source first, deduped by name.

    DuckDuckGo is capped (``web_search.ddg_quick``) and only runs when
    ``ddg_gate()`` agrees (the engine asks Laya whether search is stuck). With a
    ``background_key`` it runs off the request thread whenever the fast sources
    already found something; its hits join the next search for that key. It
    runs inline only when nothing else was found.

    Gemini (``sources.gemini``, off unless configured) searches Google for
    characters matching the facts themselves, so it helps most when there is
    nothing specific. It shares ``ddg_gate``. It runs inline when the pool is
    empty, and also when a trait seed's only hits are junk: a Wikipedia page
    that shares one word with the seed used to count as a full pool and defer
    Gemini to the next turn. ``gemini_inline=True`` forces that; ``False``
    keeps the background path once a live candidate already shows the traits.
    ``exclude_names`` alone does not count as that pool for a trait seed.

    ``specific=False`` means the facts are only broad traits (no seed, typed
    detail or LLM query). Name and keyword searches cannot match those, so
    they are skipped, DuckDuckGo never runs inline, and the pool is topped up
    from ``popular_characters`` instead, up to ``pool_size`` (candidates
    still in play; defaults to ``len(exclude_names)``) of ``popular_limit()``.
    A coverage cluster (prosthetic and blonde, or an FMA alias such as
    automail) still name-searches before that fill, capped at
    ``_COVERAGE_MAX`` new hits so a common look cannot fill the shortlist.
    AniList only matches character names, so the popular page used to fill
    the shortlist with Naruto and never ask for Edward Elric. Those queries are retrieval only.

    ``pin`` is a resolved work. Every template group keeps it, so the narrow
    retry cannot become a series-free trait query. When that work has lexicon
    headliners, AniList is searched by those character names: its API is a
    name search sorted by favourites, and a trait sentence returns unrelated
    leads.

    A host that still answers 429 after retries is not queried again in this
    search. Further sequential calls would come back empty and only make the
    limit last longer.
    """
    from .. import web_search

    # How many names were already seen (in play or ruled out) -- a count of
    # names, taken before any key expansion.
    excluded = [n for n in (exclude_names or ()) if n]
    n_seen = len(excluded)
    found: dict[str, dict[str, Any]] = {}
    web_search._begin_search()

    hits: dict[str, int] = {}
    taken_names: list[str] = []
    # Portraits for people already in the pool. Kept out of ``found`` so they
    # do not consume the new-candidate limit.
    carried: list[dict[str, Any]] = []

    def take(items: list[dict[str, Any]], source: str = "") -> None:
        """Add hits whose character is new; count them under ``source``.

        ``plain_duplicate`` matches either word order and a trailing series
        title, and keeps two different work titles of the same given name.
        A different lexicon spelling is kept so the session can absorb it.
        The dict is keyed by id so those two titles do not overwrite each
        other: they share a bare ``name_keys`` entry.
        """
        added = 0
        for cand in items:
            name = cand.get("name", "")
            keys = name_keys(name)
            # Identity aliases (Diana Prince beside Wonder Woman) stay in the
            # list so the session can absorb their tags. Same spellings drop.
            if not keys or plain_duplicate(name, excluded):
                # Already in the pool. Still hand the portrait across so a
                # later source can fill a null image_url on the next absorb.
                if keys and plain_duplicate(name, excluded) and cand.get("image_url"):
                    carried.append(cand)
                continue
            if plain_duplicate(name, taken_names):
                from .portraits import donate_image

                donate_image(list(found.values()), cand)
                continue
            if web_search.is_aggregate_page(
                name, cand.get("source_url") or "", cand.get("blurb") or ""
            ):
                continue
            found[str(cand.get("id") or min(keys))] = cand
            taken_names.append(name)
            added += 1
        if source:
            hits[source] = hits.get(source, 0) + added

    queries = _queries(constraints, medium_hint, focus=focus, rewritten=rewritten, pin=pin)
    web_search._LAST_SEARCH["queries"] = list(queries)
    # Button answers leave ``specific`` false, so the loops below never ask
    # for a name. Coverage hits are taken first: the later ``limit`` slice
    # keeps provider order, and a popular fill must not crowd the title
    # character off the shortlist. A rewrite that drops the alias still
    # loses to this loop, because it reads the original facts. The hits are
    # capped: "blonde braid" is a common look, and an uncapped FMA cast
    # filled the limit so a typed name ("Elsa") was never searched.
    coverage = web_search.coverage_queries(constraints)
    if coverage:
        with timing.span("fetch.coverage"):
            anilist_ok = medium_hint in (None, "anime", "manga", "game")
            cap = min(_COVERAGE_MAX, limit)
            for query in coverage:
                if hits.get("coverage", 0) >= cap:
                    break
                try:
                    room = cap - hits.get("coverage", 0)
                    take(wikipedia.search_characters(query, limit=4)[:room], "coverage")
                except Exception as exc:  # noqa: BLE001
                    web_search._note_error(f"coverage: {exc}")
                room = cap - hits.get("coverage", 0)
                if room > 0 and anilist_ok and not _http.host_blocked(anilist.ENDPOINT):
                    try:
                        take(anilist.search_characters(query, limit=3)[:room], "coverage")
                    except Exception as exc:  # noqa: BLE001
                        web_search._note_error(f"coverage: {exc}")
    pw_ok = False
    pw_empty = True
    pw_error = False

    with timing.span("fetch.playwright"):
        if specific and web_search._want_playwright():
            web_search._set_state("playwright_search")
            before = len(found)
            try:
                from .. import browser_search

                if browser_search.available():
                    for q in queries:
                        if len(found) >= limit:
                            break
                        take(web_search._playwright_hits(q, medium_hint, min(50, limit + n_seen)),
                         "playwright")
                    pw_ok = True
                    pw_empty = len(found) == before
                elif web_search.search_backend() == "playwright":
                    web_search._note_error("playwright unavailable; filling with ddg")
            except Exception as exc:  # noqa: BLE001
                pw_error = True
                web_search._note_error(f"playwright: {exc}")

    # Wikipedia: covers all four media and returns real prose. Stop once the
    # host is cooling down after a 429: the next query would be empty too.
    with timing.span("fetch.wikipedia"):
        for q in queries if specific else []:
            if len(found) >= limit or _http.host_blocked(wikipedia.API):
                break
            try:
                take(wikipedia.search_characters(q, limit=min(50, limit + n_seen)),
                     "wikipedia")
            except Exception as exc:  # noqa: BLE001
                web_search._note_error(f"wikipedia: {exc}")

    # AniList adds gender, popularity and a real description. It is queried
    # for games too: a game hint used to skip it, so a name search never saw
    # characters AniList files under an anime adaptation. The medium AniList
    # reports is kept. Rewriting every anime row to "game" let unrelated
    # anime characters survive the hard medium filter. Same 429 stop as
    # Wikipedia — the two hosts are independent, so one cooling does not
    # skip the other.
    with timing.span("fetch.anilist"):
        if specific and medium_hint in (None, "anime", "manga", "game") and len(found) < limit:
            anilist_queries, per_query, stop_when_full = _anilist_plan(
                constraints, pin, queries, limit)
            for q in anilist_queries:
                if _http.host_blocked(anilist.ENDPOINT):
                    break
                if stop_when_full and len(found) >= limit:
                    break
                try:
                    take(anilist.search_characters(q, limit=per_query), "anilist")
                except Exception as exc:  # noqa: BLE001
                    web_search._note_error(f"anilist: {exc}")

    # Hits from the previous turn's background fill, after this turn's
    # fresher, more relevant sources.
    if background_key:
        take(take_background(background_key), "ddg_bg")
        take(_GEMINI_BG.take(background_key), "gemini_bg")

    in_play = n_seen if pool_size is None else pool_size
    room = popular_limit() - in_play
    if not specific and room > 0:
        with timing.span("fetch.popular"):
            try:
                # Over-fetch by the names already seen (in play or ruled out),
                # which ``take`` skips, so the free room can still be filled.
                fresh = [c for c in popular_characters(medium_hint, room + n_seen)
                         if not plain_duplicate(c.get("name", ""), excluded)]
                take(fresh[:room], "popular")
            except Exception as exc:  # noqa: BLE001
                web_search._note_error(f"popular: {exc}")

    def gate() -> bool:
        """Whether slow sources should run. The engine memoises its answer, so
        DuckDuckGo and Gemini share one Laya pool_fits call."""
        if ddg_gate is None:
            return True
        try:
            return bool(ddg_gate())
        except Exception:  # noqa: BLE001 - a failed gate means "search"
            return True

    # Gemini matches traits, not names, so broad facts are fine. The facts are
    # the player's own search terms -- never scraped names.
    # "fictional character" is the engine's placeholder for "no facts yet";
    # a billed grounded call on it would only list famous characters.
    facts = [c for c in constraints
             if c and c.strip() and c.strip().lower() != "fictional character"][-8:]
    # A full page of junk must not block the inline call: the limit slice
    # below puts trait matches first, so Mika is not cut off by "The Saint".
    # ``gate`` is called once; the engine memoises it with the DuckDuckGo gate.
    run_gemini = bool(facts and gemini.enabled() and gate())
    inline = bool(run_gemini and _gemini_inline(found, excluded, facts, gemini_inline))
    if run_gemini and (inline or len(found) < limit):
        if background_key and not inline and _gemini_background_on():
            with timing.span("fetch.gemini_background"):
                _GEMINI_BG.start(background_key, _gemini_run, facts, medium_hint, limit)
        else:
            with timing.span("fetch.gemini"):
                take(gemini.search_characters(facts, medium_hint, limit), "gemini")

    need_ddg = use_ddg and web_search._want_ddg_fill(
        len(found), limit, pw_ok=pw_ok, pw_empty=pw_empty, pw_error=pw_error
    ) and gate()
    # Inline only when the player would otherwise have no candidates at all
    # and there is something specific to search for; broad traits alone get
    # nothing useful back, so they only ever search in the background.
    inline_ok = specific and not found and not excluded
    if need_ddg and background_key and not inline_ok and _ddg_background_on():
        with timing.span("fetch.ddg_background"):
            _bg_start(background_key, queries, limit)
        need_ddg = False
    if not specific:
        need_ddg = False
    with timing.span("fetch.ddg"):
        if need_ddg:
            web_search._set_state("fill_ddg")
            try:
                take(web_search.ddg_quick(queries, min(50, limit + n_seen)), "ddg")
            except Exception as exc:  # noqa: BLE001
                web_search._note_error(f"ddg: {exc}")

    # Preserve provider relevance. Fame is only a capped prior after retrieval.
    # A trait seed is the exception: pages that actually show the combination
    # lead, so a junk hit that arrived first cannot fill the limit alone.
    out = list(found.values())
    clue = _trait_clue(facts)
    if clue:
        matched = [c for c in out if _hit_shows_traits(c, clue)]
        if matched:
            matched_ids = {id(c) for c in matched}
            out = matched + [c for c in out if id(c) not in matched_ids]
    out = out[:limit]
    # Portrait-only updates ride along after the limit so a known name can
    # still absorb an image found by a later source.
    out.extend(carried[:8])
    from .portraits import fill_portraits

    fill_portraits(out, limit=4)
    with timing.span("fetch.enrich"):
        if web_search._enrich_on() and out:
            web_search._set_state("enrich")
            try:
                web_search.enrich_candidates(out)
            except Exception as exc:  # noqa: BLE001
                web_search._note_error(f"enrich: {exc}")
    web_search._set_state("done")
    timing.note(found=len(out),
                hits=",".join(f"{k}:{v}" for k, v in hits.items()) or "none")
    errors = web_search.last_search_meta()["errors"]
    if errors:
        timing.note(err=" | ".join(errors[:2])[:160])
    return out
