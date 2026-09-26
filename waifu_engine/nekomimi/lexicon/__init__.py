"""Static maps loaded from package YAML. Decision code stays in the callers.

Each name is the table a matcher already used. The YAML files are the only
definition; this module binds them once at import.
"""

from __future__ import annotations

from .load import LexiconError, load_lexicon

_LEXICON = load_lexicon()
_MEDIUM = _LEXICON["medium"]
_FRANCHISES = _LEXICON["franchises"]
_TRAITS = _LEXICON["traits"]
_CATEGORIES = _LEXICON["categories"]
_CHARACTERS = _LEXICON["characters"]
_COVERAGE = _LEXICON["coverage"]

# medium.yml — engine._eliminate_by_medium / _known_choice. Empty other is no filter.
MEDIUM_VALUES = _MEDIUM["media"]
MEDIUM_ACCEPTS = _MEDIUM["accepts"]
SEARCH_SUFFIX = _MEDIUM["search_suffix"]

# colors.yml — web_search.is_color_phrase
COLOR_WORDS = _LEXICON["colors"]

# franchises.yml — web_search._phrase_re walks MARKERS in order.
SERIES_MARKERS = _FRANCHISES["markers"]
TYPED_ALIASES = _FRANCHISES["typed_aliases"]
EXACT_ALIASES = _FRANCHISES["exact_aliases"]
SERIES_CRUMBS = _FRANCHISES["crumbs"]
AGGREGATE_EXACT = _FRANCHISES["aggregate_exact"]
PUBLISHER_LABELS = _FRANCHISES["publisher_labels"]
PUBLISHER_KEYS = _FRANCHISES["publisher_keys"]
NAME_BLOCK = _FRANCHISES["name_block"]
GENERIC_SERIES = _FRANCHISES["generic_series"]

# traits.yml — web_search._marker_re. question_bank.json stays the question catalog.
TRAIT_PATTERNS = _TRAITS["patterns"]
MINE_ONLY = _TRAITS["mine_only"]
TRAIT_RECONCILE = _TRAITS["reconcile"]

# categories.yml — engine skip/pin still decides when these sets apply.
BROAD_CATEGORIES = _CATEGORIES["broad"]
HAIR_CATEGORIES = _CATEGORIES["color_detail"]
SERIES_APPEARANCE = _CATEGORIES["series_appearance"]
SERIES_SPLIT = _CATEGORIES["series_split"]
POPULAR_CATEGORIES = _CATEGORIES["popular"]

# characters.yml — names.identity_id, web_search.headliner_label / is_non_character.
CHARACTER_IDENTITIES = _CHARACTERS["identities"]
HEADLINERS = _CHARACTERS["headliners"]
NON_CHARACTERS = _CHARACTERS["non_characters"]

# coverage.yml — web_search.coverage_queries. Name searches, not score pins.
COVERAGE = _COVERAGE

__all__ = [
    "AGGREGATE_EXACT",
    "BROAD_CATEGORIES",
    "CHARACTER_IDENTITIES",
    "COLOR_WORDS",
    "COVERAGE",
    "EXACT_ALIASES",
    "HEADLINERS",
    "NON_CHARACTERS",
    "GENERIC_SERIES",
    "HAIR_CATEGORIES",
    "LexiconError",
    "MEDIUM_ACCEPTS",
    "MEDIUM_VALUES",
    "MINE_ONLY",
    "NAME_BLOCK",
    "POPULAR_CATEGORIES",
    "PUBLISHER_KEYS",
    "PUBLISHER_LABELS",
    "SEARCH_SUFFIX",
    "SERIES_APPEARANCE",
    "SERIES_CRUMBS",
    "SERIES_SPLIT",
    "SERIES_MARKERS",
    "TRAIT_PATTERNS",
    "TRAIT_RECONCILE",
    "TYPED_ALIASES",
]
