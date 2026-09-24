"""Character-name identity, shared by every place that de-duplicates candidates.

Japanese names are written in both orders: sources like AniList and Japanese
wikis give "Shimoe Koharu" (family name first), while English wikis and
Gemini give "Koharu Shimoe". Compared letter by letter they differ, and one
character ended up in the pool twice, splitting her own probability (73% and
9% for the same person).
"""

from __future__ import annotations

import re

# Word separators between name parts: whitespace, the Japanese middle dot and
# full-width space, commas ("Shimoe, Koharu").
_WORDS = re.compile(r"[\s・･·,　]+")


def name_keys(name: str) -> frozenset[str]:
    """Keys under which two spellings of one character's name are equal.

    Two keys: the letters and digits in order ("spiderman" for "Spider-Man"
    and "Spider Man"), and the name's words sorted ("koharushimoe" for either
    order of "Koharu Shimoe"). Names match when any key is shared. Empty for
    a name with no letters.
    """
    words = [w for w in ("".join(ch for ch in part.lower() if ch.isalnum())
                         for part in _WORDS.split(name or "")) if w]
    if not words:
        return frozenset()
    return frozenset({"".join(words), "".join(sorted(words))})


def same_character(a: str, b: str) -> bool:
    """Whether two names are spellings of one character (see ``name_keys``)."""
    return bool(name_keys(a) & name_keys(b))


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
