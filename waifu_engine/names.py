"""Character-name identity, shared by every place that de-duplicates candidates.

Japanese names are written in both orders: sources like AniList and Japanese
wikis give "Shimoe Koharu" (family name first), while English wikis and
Gemini give "Koharu Shimoe". Compared letter by letter they differ, and one
character ended up in the pool twice, splitting her own probability (73% and
9% for the same person).

A second split is the titled form. Wikipedia stores "Link" after dropping a
disambiguation parenthetical; Fandom and other HTML indexes keep
"Link (The Legend of Zelda)". Those rows used to share no key, so the
posterior was cut in half and neither side reached the guess threshold.
"Young Link" and "Toon Link" are different people: the qualifier is part of
the name, not a trailing title, and it must stay distinct.
"""

from __future__ import annotations

import re
from typing import Iterable

# Publisher keys: lexicon/franchises.yml. ``series_key`` is the matcher.
# Identity rows: lexicon/characters.yml. ``identity_id`` is the matcher.
from .nekomimi.lexicon import CHARACTER_IDENTITIES as _CHARACTER_IDENTITIES
from .nekomimi.lexicon import PUBLISHER_KEYS as _PUBLISHER_KEYS
from .nekomimi.lexicon.load import LexiconError

# Word separators between name parts: whitespace, the Japanese middle dot and
# full-width space, commas ("Shimoe, Koharu").
_WORDS = re.compile(r"[\s・･·,　]+")
# One trailing "(...)" note. Applied repeatedly so "(video game)" after a
# series title peels off too. A note in the middle of the name is left alone.
_TRAILING_PAREN = re.compile(r"^(?P<base>.*\S)\s*\((?P<note>[^)]*)\)\s*$")


def _split_disambiguation(name: str) -> tuple[str, str]:
    """Return ``(base name, distinguishing title note)``.

    Medium-only notes ("anime", "video game") are discarded: they do not
    name a different person. The first remaining trailing note is the work
    title ("The Legend of Zelda"). "Young Link" has no parenthetical, so the
    whole string stays the base and does not collapse into "Link".
    """
    base = (name or "").strip()
    while True:
        match = _TRAILING_PAREN.match(base)
        if not match:
            return base, ""
        nxt = match.group("base").strip()
        if not nxt:
            return base, ""
        base = nxt
        note = match.group("note").strip()
        if note and not _MEDIUM_NOTE.fullmatch(f"({note})"):
            return base, note


def _keys(name: str) -> frozenset[str]:
    """Letter keys of a base name whose title note is already removed.

    Two keys: letters in order ("spiderman") and words sorted
    ("koharushimoe"), so either word order shares one. Empty when ``name``
    has no letters.
    """
    words = [w for w in ("".join(ch for ch in part.lower() if ch.isalnum())
                         for part in _WORDS.split(name or "")) if w]
    if not words:
        return frozenset()
    return frozenset({"".join(words), "".join(sorted(words))})


def index_identities(rows: Iterable[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Map each alias key to one identity id, and each id to its display name.

    A key shared by two ids raises. ``same_character`` would otherwise glue
    two people together the way a bare substring once glued every "Soryu".
    """
    by_key: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for row in rows:
        slug = row["id"]
        canonical[slug] = row["canonical"]
        for alias in row["names"]:
            base, _note = _split_disambiguation(alias)
            keys = _keys(base)
            if not keys:
                raise LexiconError(f"characters.yml identity {slug} has no letters in {alias!r}")
            for key in keys:
                prev = by_key.get(key)
                if prev is not None and prev != slug:
                    raise LexiconError(
                        f"characters.yml key {key!r} is both {prev} and {slug}"
                    )
                by_key[key] = slug
    return by_key, canonical


_IDENTITY_BY_KEY, _CANONICAL_BY_ID = index_identities(_CHARACTER_IDENTITIES)


def identity_id(name: str) -> str:
    """Lexicon id shared by known spellings of one person, or ``""``.

    "Asuka Langley Sohryu" and bare "Soryu" are one id. "Kyoko Zeppelin Soryu"
    is not: the alias is the whole name, not a surname inside a longer one.
    """
    base, _note = _split_disambiguation(name)
    found = {_IDENTITY_BY_KEY[key] for key in _keys(base) if key in _IDENTITY_BY_KEY}
    if len(found) == 1:
        return next(iter(found))
    return ""


def canonical_name(name: str) -> str:
    """Display name for a known alias, or ``""`` when ``name`` is not in the table."""
    return _CANONICAL_BY_ID.get(identity_id(name), "")


def name_keys(name: str) -> frozenset[str]:
    """Keys under which two spellings of one character's name are equal.

    A trailing parenthetical is stripped first, so "Link" and
    "Link (The Legend of Zelda)" share keys. Two different titles also share
    the bare key ("Aqua (Kingdom Hearts)" and "Aqua (KonoSuba)");
    ``same_character`` is what keeps those people apart. Empty for a name
    with no letters.
    """
    base, _note = _split_disambiguation(name)
    return _keys(base)


def same_character(a: str, b: str) -> bool:
    """Whether two names are one character.

    Word order does not matter, and a bare name matches a titled form of
    that same base. Both sides carrying different work titles do not match:
    key intersection alone would merge every "Aqua (work)". Equivalent
    titles do match, via ``same_series_key`` ("Re:Zero" and its full name).
    A qualifier in the name itself ("Young Link", "Toon Link") stays a
    different base. Known scatters (Soryu / Sohryu / Shikinami, Wonder Woman
    / Diana Prince) match through ``identity_id`` even when the tokens differ.
    """
    id_a, id_b = identity_id(a), identity_id(b)
    if id_a and id_a == id_b:
        return True
    base_a, note_a = _split_disambiguation(a)
    base_b, note_b = _split_disambiguation(b)
    if not (_keys(base_a) & _keys(base_b)):
        return False
    key_a, key_b = series_key(note_a), series_key(note_b)
    if key_a and key_b and not same_series_key(key_a, key_b):
        return False
    return True


def already_seen(name: str, seen: Iterable[str]) -> bool:
    """Whether ``name`` is the same character as any name in ``seen``.

    Intersecting ``name_keys`` merges every titled form that shares a bare
    name, including two different people. Pairwise ``same_character`` does
    not: "Link" absorbs "Link (The Legend of Zelda)", and
    "Aqua (Kingdom Hearts)" does not absorb "Aqua (KonoSuba)".
    """
    return any(same_character(name, other) for other in seen)


# Placeholders sources use when they could not tell the series.
_NO_SERIES = {"", "webresult", "unknown", "none"}

# A trailing note that only names the medium, not a different work.
# "The Office (American TV series)" and "(British TV series)" must stay apart.
_MEDIUM_NOTE = re.compile(
    r"\s*\((?:video game|game|anime|manga|comic|comics|film|movie|"
    r"tv series|television series|visual novel)\)\s*$",
    re.I,
)


def series_key(series: str) -> str:
    """Comparable form of a series name, or "" when the series is unknown.

    Letters and digits only, lower-cased, with a leading "The" dropped and a
    trailing parenthetical dropped only when it names the medium: "Blue
    Archive (video game)" and "blue archive" both give "bluearchive". A
    qualifier that distinguishes two works stays, so the American and British
    Office do not collapse into one series question option.
    """
    s = _MEDIUM_NOTE.sub("", series or "").strip()
    s = re.sub(r"^the\s+", "", s, flags=re.I)
    key = "".join(ch for ch in s.lower() if ch.isalnum())
    return "" if key in _NO_SERIES else key


def same_series(a: str, b: str) -> bool:
    """Whether two series names are the same work.

    Equal keys match, and so does a key that starts the other when it is at
    least 5 letters long -- "Re:Zero" vs "Re:Zero - Starting Life in Another
    World". Unknown series never match anything.
    """
    return same_series_key(series_key(a), series_key(b))


def same_series_key(ka: str, kb: str) -> bool:
    """``same_series`` on keys already made by ``series_key``."""
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    short, long_ = sorted((ka, kb), key=len)
    return len(short) >= 5 and long_.startswith(short)


def is_publisher_series(series: str) -> bool:
    """Whether ``series`` is a publisher bucket rather than one work.

    A series question built from these labels offers "Marvel Comics" as if it
    were Cowboy Bebop, and every character in that bucket shares the pick.
    """
    return series_key(series) in _PUBLISHER_KEYS


def name_tokens(name: str) -> list[str]:
    """Alphanumeric words of a display name, in order."""
    base, _note = _split_disambiguation(name)
    return [w for w in ("".join(ch for ch in part.lower() if ch.isalnum())
                        for part in _WORDS.split(base or "")) if w]


def longer_namesake(short: str, long: str) -> bool:
    """Whether ``long`` is ``short`` plus extra name words.

    "Mario" and "Mario Rossi" are different people. A trailing series title
    is already stripped, so "Mario (Super Mario)" is not a longer namesake.
    """
    a, b = name_tokens(short), name_tokens(long)
    if not a or len(a) >= len(b):
        return False
    return b[:len(a)] == a
