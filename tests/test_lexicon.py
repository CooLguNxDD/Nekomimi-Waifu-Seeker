"""Lexicon YAML matches the tables the matchers used to hard-code.

Order-sensitive sequences (franchise markers, typed aliases, appearance pin)
are compared as tuples. ``other`` stays an empty accept set. Loader mistakes
raise; they do not become empty maps.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest

from waifu_engine import web_search
from waifu_engine.nekomimi import traits
from waifu_engine.nekomimi.lexicon import (
    AGGREGATE_EXACT,
    COLOR_WORDS,
    EXACT_ALIASES,
    GENERIC_SERIES,
    MEDIUM_ACCEPTS,
    MEDIUM_VALUES,
    MINE_ONLY,
    NAME_BLOCK,
    POPULAR_CATEGORIES,
    PUBLISHER_KEYS,
    PUBLISHER_LABELS,
    SEARCH_SUFFIX,
    SERIES_APPEARANCE,
    SERIES_CRUMBS,
    SERIES_MARKERS,
    TRAIT_PATTERNS,
    TYPED_ALIASES,
)
from waifu_engine.nekomimi.lexicon.load import (
    LexiconError,
    build_colors,
    build_franchises,
    build_medium,
    build_traits,
    load_document,
    parse_document,
)
from waifu_engine.sources import wikipedia


# Frozen from the Python literals this migration replaced. A reordering of
# Vocaloid aliases, or dropping ``exact`` on mario, fails here.
_VOCALOID_FIRST = (
    ("project sekai", "Vocaloid"),
    ("colorful stage", "Vocaloid"),
    ("project diva", "Vocaloid"),
    ("crypton future", "Vocaloid"),
    ("vocaloid", "Vocaloid"),
)
_TYPED = (
    ("cowboy bebop", "Cowboy Bebop"),
    ("star wars", "Star Wars"),
    ("super mario", "Super Mario"),
    ("spider-man", "Spider-Man"),
    ("spiderman", "Spider-Man"),
    ("spider man", "Spider-Man"),
    ("the simpsons", "The Simpsons"),
    ("simpsons", "The Simpsons"),
    ("mario", "Super Mario"),
)


def _bank_tags() -> set[str]:
    """Tags the question bank assigns, including trait-block prefixes."""
    raw = json.loads(
        files("waifu_engine.nekomimi").joinpath("question_bank.json").read_text(encoding="utf-8")
    )
    tags: set[str] = set()
    for entry in raw["questions"]:
        kind = entry.get("template")
        if kind == "trait_block":
            prefix = entry.get("tag_prefix", "")
            for item in entry["items"]:
                tags.add(f"{prefix}{item['slug']}")
        elif kind == "choice":
            for option in entry["options"]:
                tags.update(option.get("tags") or [])
        elif kind == "yesno":
            tags.update(entry.get("tags_true") or [])
            tags.update(entry.get("tags_false") or [])
    return tags


def test_medium_other_is_present_and_does_not_filter():
    """``other`` stays a key whose accept set is empty."""
    assert MEDIUM_VALUES == ("anime", "manga", "comic", "game", "movie", "tv")
    assert "other" in MEDIUM_ACCEPTS
    assert MEDIUM_ACCEPTS["other"] == frozenset()
    assert MEDIUM_ACCEPTS["anime"] == frozenset({"anime", "manga"})
    assert MEDIUM_ACCEPTS["movie"] == frozenset({"movie", "tv"})
    assert MEDIUM_ACCEPTS["tv"] == frozenset({"tv", "movie"})
    assert traits.MEDIUM_ACCEPTS is MEDIUM_ACCEPTS
    assert SEARCH_SUFFIX["game"] == "video game character"
    assert SEARCH_SUFFIX["tv"] == "TV series character"


def test_marker_and_alias_sequences_match_the_old_tables():
    """First-match order and the Mario exact flag survive the move."""
    assert SERIES_MARKERS[:5] == _VOCALOID_FIRST
    assert len(SERIES_MARKERS) == 56
    assert SERIES_MARKERS[5] == ("blue archive", "Blue Archive")
    assert TYPED_ALIASES == _TYPED
    assert EXACT_ALIASES == frozenset({"mario"})
    assert web_search._SERIES_MARKERS == SERIES_MARKERS
    from waifu_engine.nekomimi import engine

    assert engine._FRANCHISE_ALIASES == _TYPED
    assert engine._EXACT_ALIASES == frozenset({"mario"})


def test_set_maps_keep_the_old_membership():
    """Crumbs, colours, aggregates, publishers, and the name block stay complete."""
    assert len(COLOR_WORDS) == 28
    assert {"teal", "blonde", "blond", "pink"} <= COLOR_WORDS
    assert web_search._COLOR_WORDS == COLOR_WORDS
    assert len(SERIES_CRUMBS) == 13
    assert {"internet meme", "voice bank", "software"} <= SERIES_CRUMBS
    assert AGGREGATE_EXACT == {"vocaloids", "fanloid", "fanloids", "vocaloid characters"}
    assert "vocaloid" in NAME_BLOCK and len(NAME_BLOCK) == 63
    assert web_search.SERIES_BLOCK == NAME_BLOCK
    assert PUBLISHER_LABELS == {
        "marvel comics", "dc comics", "image comics", "dark horse comics", "dark horse",
    }
    assert "image" in PUBLISHER_KEYS and "marvelcomics" in PUBLISHER_KEYS
    assert wikipedia._PUBLISHER_SERIES == PUBLISHER_LABELS
    assert len(GENERIC_SERIES) == 11 and "video game" in GENERIC_SERIES
    assert SERIES_APPEARANCE == ("hair_color", "look_halo", "look_wings", "look_horns")
    assert list(POPULAR_CATEGORIES) == ["game", "comic", "movie", "tv"]
    assert POPULAR_CATEGORIES["comic"][0] == "Category:Marvel Comics superheroes"


def test_trait_patterns_come_from_the_lexicon_in_order():
    """Mined slugs stay in the old order, and wings stay body-wing markers."""
    assert [slug for slug, _ in TRAIT_PATTERNS][:2] == ["female", "male"]
    assert len(TRAIT_PATTERNS) == 42
    wings = dict(TRAIT_PATTERNS)["wings"]
    assert wings == ("angel wings", "feathered wings")
    assert web_search.TRAIT_PATTERNS == TRAIT_PATTERNS
    assert MINE_ONLY == ()


def test_bank_tags_and_lexicon_ids_agree():
    """A lexicon id is a bank tag, or it is listed in ``mine_only``."""
    bank = _bank_tags()
    ids = [slug for slug, _ in TRAIT_PATTERNS]
    assert len(ids) == len(set(ids))
    mine_only = set(MINE_ONLY)
    for slug in ids:
        if slug not in bank and slug not in mine_only:
            raise AssertionError(f"{slug} is neither a bank tag nor mine-only")
    assert mine_only <= set(ids)
    assert not (mine_only & bank)
    # Hair and eye option tags the bank already uses are the lexicon ids.
    for slug in ("pink", "blonde", "eyes-blue", "halo", "wings", "horns"):
        assert slug in bank and slug in ids


def test_missing_file_bad_type_and_duplicate_id_raise():
    """Loader failures raise. They do not yield an empty table."""
    with pytest.raises(LexiconError, match="missing"):
        load_document("no-such-lexicon.yml")
    with pytest.raises(LexiconError, match="must be a mapping"):
        parse_document("- just a list\n", "bad.yml")
    with pytest.raises(LexiconError, match="not valid YAML"):
        parse_document("!!python/object/apply:os.system ['echo owned']\n", "tag.yml")
    with pytest.raises(LexiconError, match="words"):
        build_colors({"words": "pink"})
    with pytest.raises(LexiconError, match="unknown keys"):
        build_colors({"words": ["pink"], "regex": ".*"})
    medium = load_document("medium.yml")
    broken = json.loads(json.dumps(medium))
    broken["accepts"]["other"] = {"media": []}
    with pytest.raises(LexiconError, match="filter: false"):
        build_medium(broken)
    missing_other = json.loads(json.dumps(medium))
    del missing_other["accepts"]["other"]
    with pytest.raises(LexiconError, match="other"):
        build_medium(missing_other)
    traits_doc = load_document("traits.yml")
    dup = json.loads(json.dumps(traits_doc))
    dup["traits"].append(dict(dup["traits"][0]))
    with pytest.raises(LexiconError, match="duplicate id"):
        build_traits(dup)
    franchises = load_document("franchises.yml")
    dup_phrase = json.loads(json.dumps(franchises))
    dup_phrase["markers"].append(dict(dup_phrase["markers"][0]))
    with pytest.raises(LexiconError, match="duplicate phrase"):
        build_franchises(dup_phrase)


def test_question_bank_stays_json():
    """The question catalog is not converted to YAML."""
    text = files("waifu_engine.nekomimi").joinpath("question_bank.json").read_text(encoding="utf-8")
    assert text.lstrip().startswith("{")
    assert "questions" in json.loads(text)
