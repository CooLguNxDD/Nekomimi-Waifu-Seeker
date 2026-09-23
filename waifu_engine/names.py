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
