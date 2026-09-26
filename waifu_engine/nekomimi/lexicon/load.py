"""Load package-local lexicon YAML.

PyYAML's safe loader is the only parser, wrapped so a repeated mapping key
raises. A missing file, a bad type, a duplicate key, or a duplicate id raises
``LexiconError`` at import. An empty fallback would drop every franchise
marker and look like a search bug.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml

_PKG = "waifu_engine.nekomimi.lexicon"


class LexiconError(ValueError):
    """A lexicon file is missing, mistyped, or inconsistent."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that rejects a mapping key the second time it appears.

    ``yaml.safe_load`` keeps the last value of a repeated key, so a duplicated
    ``accepts.anime`` would change medium filtering without a load error.
    """

    def __init__(self, stream: Any):
        """Remember the file name and the key path currently being built."""
        super().__init__(stream)
        self.document_name = ""
        self._key_path: list[str] = []


def _path_segment(key: Any) -> str:
    """Return ``key`` as one segment of a dotted path."""
    return key if isinstance(key, str) else repr(key)


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    """Build one mapping and raise ``LexiconError`` on a repeated key.

    Nested mappings push the key onto ``loader._key_path``, so the message
    names the file and the full path (``accepts.anime``), not only the leaf.
    """
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        segment = _path_segment(key)
        path = ".".join([*loader._key_path, segment])
        try:
            repeated = key in mapping
        except TypeError as exc:
            where = loader.document_name or "lexicon"
            raise LexiconError(f"lexicon file {where} has an unhashable key at {path}") from exc
        if repeated:
            where = loader.document_name or "lexicon"
            raise LexiconError(f"lexicon file {where} has duplicate key {path}")
        loader._key_path.append(segment)
        try:
            value = loader.construct_object(value_node, deep=deep)
        finally:
            loader._key_path.pop()
        mapping[key] = value
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def parse_document(text: str, name: str) -> dict[str, Any]:
    """Parse ``text`` as one lexicon mapping.

    The loader rejects ``!!python`` tags and duplicate mapping keys, including
    nested ones. A non-mapping root raises rather than returning an empty map.
    """
    loader = _UniqueKeyLoader(text)
    loader.document_name = name
    try:
        try:
            data = loader.get_single_data()
        except LexiconError:
            raise
        except yaml.YAMLError as exc:
            raise LexiconError(f"lexicon file {name} is not valid YAML: {exc}") from exc
    finally:
        loader.dispose()
    if not isinstance(data, dict):
        raise LexiconError(f"lexicon file {name} must be a mapping")
    return data


def load_document(name: str) -> dict[str, Any]:
    """Return the mapping stored in lexicon file ``name``.

    A missing file raises. There is no silent empty fallback.
    """
    path = files(_PKG).joinpath(name)
    if not path.is_file():
        raise LexiconError(f"lexicon file {name} is missing")
    return parse_document(path.read_text(encoding="utf-8"), name)


def _require_keys(data: dict[str, Any], where: str, allowed: set[str], required: set[str]) -> None:
    """Reject unknown or missing keys in ``data``."""
    unknown = set(data) - allowed
    if unknown:
        raise LexiconError(f"{where} has unknown keys: {sorted(unknown)}")
    missing = sorted(required - set(data))
    if missing:
        raise LexiconError(f"{where} is missing keys: {missing}")


def _str_list(value: Any, where: str, *, unique: bool = False) -> list[str]:
    """Return ``value`` as a list of non-empty strings. An empty list is allowed."""
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise LexiconError(f"{where} must be a list of non-empty strings")
    if unique and len(value) != len(set(value)):
        raise LexiconError(f"{where} has duplicate entries")
    return list(value)


def _unique(items: list[str], where: str) -> None:
    """Raise when ``items`` repeats a string."""
    if len(items) != len(set(items)):
        raise LexiconError(f"{where} has duplicate entries")


def build_medium(doc: dict[str, Any]) -> dict[str, Any]:
    """Return media, accept sets, and search suffixes from a medium document.

    ``other`` must be ``filter: false``. That sentinel means do not filter,
    not accept-nothing: an empty accept set used to wipe every known medium.
    """
    _require_keys(doc, "medium.yml", {"media", "accepts", "search_suffix"}, {"media", "accepts", "search_suffix"})
    media = _str_list(doc["media"], "medium.yml media", unique=True)
    if not media:
        raise LexiconError("medium.yml media must not be empty")
    accepts_raw = doc["accepts"]
    if not isinstance(accepts_raw, dict) or not accepts_raw:
        raise LexiconError("medium.yml accepts must be a mapping")
    if "other" not in accepts_raw:
        raise LexiconError("medium.yml accepts.other is required")
    accepts: dict[str, frozenset[str]] = {}
    for key, row in accepts_raw.items():
        if not isinstance(key, str) or not key:
            raise LexiconError("medium.yml accept keys must be strings")
        if not isinstance(row, dict):
            raise LexiconError(f"medium.yml accepts.{key} must be a mapping")
        _require_keys(row, f"medium.yml accepts.{key}", {"media", "filter"}, set())
        if row.get("filter") is False:
            if row.get("media"):
                raise LexiconError(f"medium.yml accepts.{key} sets filter false and a media list")
            accepts[key] = frozenset()
            continue
        if "filter" in row:
            raise LexiconError(f"medium.yml accepts.{key} filter must be false")
        if "media" not in row:
            raise LexiconError(f"medium.yml accepts.{key} needs media or filter: false")
        vals = _str_list(row["media"], f"medium.yml accepts.{key}.media", unique=True)
        if not vals:
            raise LexiconError(f"medium.yml accepts.{key}.media is empty; use filter: false to skip filtering")
        unknown = [v for v in vals if v not in media]
        if unknown:
            raise LexiconError(f"medium.yml accepts.{key} names unknown media {unknown}")
        accepts[key] = frozenset(vals)
    other = accepts_raw["other"]
    if not isinstance(other, dict) or other.get("filter") is not False:
        raise LexiconError("medium.yml other must be filter: false (do not filter)")
    suffix = doc["search_suffix"]
    if not isinstance(suffix, dict) or not suffix:
        raise LexiconError("medium.yml search_suffix must be a mapping")
    suffixes: dict[str, str] = {}
    for key, value in suffix.items():
        if key not in media:
            raise LexiconError(f"medium.yml search_suffix has unknown medium {key!r}")
        if not isinstance(value, str) or not value.strip():
            raise LexiconError(f"medium.yml search_suffix.{key} must be a string")
        suffixes[key] = value
    if set(suffixes) != set(media):
        raise LexiconError("medium.yml search_suffix must cover every medium")
    return {"media": tuple(media), "accepts": accepts, "search_suffix": suffixes}


def build_colors(doc: dict[str, Any]) -> frozenset[str]:
    """Return the colour-word set. Membership is what ``is_color_phrase`` checks."""
    _require_keys(doc, "colors.yml", {"words"}, {"words"})
    words = _str_list(doc["words"], "colors.yml words", unique=True)
    if not words:
        raise LexiconError("colors.yml words must not be empty")
    return frozenset(words)


def _phrase_rows(rows: Any, where: str, *, allow_exact: bool) -> list[dict[str, Any]]:
    """Return ordered phrase/label rows. Duplicate phrases fail."""
    if not isinstance(rows, list) or not rows:
        raise LexiconError(f"{where} must be a non-empty list")
    allowed = {"phrase", "label", "exact"} if allow_exact else {"phrase", "label"}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise LexiconError(f"{where}[{index}] must be a mapping")
        _require_keys(row, f"{where}[{index}]", allowed, {"phrase", "label"})
        phrase, label = row["phrase"], row["label"]
        if not isinstance(phrase, str) or not phrase.strip():
            raise LexiconError(f"{where}[{index}].phrase must be a string")
        if not isinstance(label, str) or not label.strip():
            raise LexiconError(f"{where}[{index}].label must be a string")
        if phrase in seen:
            raise LexiconError(f"{where} has duplicate phrase {phrase!r}")
        seen.add(phrase)
        item: dict[str, Any] = {"phrase": phrase, "label": label}
        if allow_exact:
            exact = row.get("exact", False)
            if not isinstance(exact, bool):
                raise LexiconError(f"{where}[{index}].exact must be a boolean")
            item["exact"] = exact
        out.append(item)
    return out


def build_franchises(doc: dict[str, Any]) -> dict[str, Any]:
    """Return ordered markers, typed aliases, and the set-shaped franchise lists.

    Marker order is the match order (first hit wins). ``exact`` aliases stay
    on the typed list so a scraped marker cannot inherit Mario's whole-clue rule.
    """
    allowed = {
        "markers", "typed_aliases", "crumbs", "aggregate_exact",
        "publishers", "name_block", "generic_series",
    }
    _require_keys(doc, "franchises.yml", allowed, allowed)
    markers = _phrase_rows(doc["markers"], "franchises.yml markers", allow_exact=False)
    aliases = _phrase_rows(doc["typed_aliases"], "franchises.yml typed_aliases", allow_exact=True)
    if not any(row["exact"] for row in aliases):
        raise LexiconError("franchises.yml typed_aliases needs an exact alias (mario)")
    crumbs = _str_list(doc["crumbs"], "franchises.yml crumbs", unique=True)
    aggregate = _str_list(doc["aggregate_exact"], "franchises.yml aggregate_exact", unique=True)
    name_block = _str_list(doc["name_block"], "franchises.yml name_block", unique=True)
    generic = _str_list(doc["generic_series"], "franchises.yml generic_series", unique=True)
    publishers = doc["publishers"]
    if not isinstance(publishers, list) or not publishers:
        raise LexiconError("franchises.yml publishers must be a non-empty list")
    labels: list[str] = []
    keys: list[str] = []
    for index, row in enumerate(publishers):
        if not isinstance(row, dict):
            raise LexiconError(f"franchises.yml publishers[{index}] must be a mapping")
        _require_keys(row, f"franchises.yml publishers[{index}]", {"label", "key"}, {"key"})
        key = row["key"]
        if not isinstance(key, str) or not key.isalnum() or key != key.lower():
            raise LexiconError(f"franchises.yml publishers[{index}].key must be a lowercase alphanumeric string")
        if key in keys:
            raise LexiconError(f"franchises.yml publishers has duplicate key {key!r}")
        keys.append(key)
        if "label" in row:
            label = row["label"]
            if not isinstance(label, str) or not label.strip():
                raise LexiconError(f"franchises.yml publishers[{index}].label must be a string")
            labels.append(label.lower())
    _unique(labels, "franchises.yml publisher labels")
    return {
        "markers": tuple((row["phrase"], row["label"]) for row in markers),
        "typed_aliases": tuple((row["phrase"], row["label"]) for row in aliases),
        "exact_aliases": frozenset(row["phrase"] for row in aliases if row["exact"]),
        "crumbs": frozenset(crumbs),
        "aggregate_exact": frozenset(aggregate),
        "publisher_labels": frozenset(labels),
        "publisher_keys": frozenset(keys),
        "name_block": frozenset(name_block),
        "generic_series": frozenset(generic),
    }


def build_traits(doc: dict[str, Any]) -> dict[str, Any]:
    """Return mined trait rows and the explicit mine-only id list.

    Duplicate ids fail. ``reconcile`` is a flag for the wing/halo matcher in
    Python; it is not a rule the loader evaluates.
    """
    _require_keys(doc, "traits.yml", {"mine_only", "traits"}, {"mine_only", "traits"})
    mine_only = _str_list(doc["mine_only"], "traits.yml mine_only", unique=True)
    rows = doc["traits"]
    if not isinstance(rows, list) or not rows:
        raise LexiconError("traits.yml traits must be a non-empty list")
    patterns: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    reconcile: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise LexiconError(f"traits.yml traits[{index}] must be a mapping")
        _require_keys(row, f"traits.yml traits[{index}]", {"id", "group", "markers", "reconcile"}, {"id", "group", "markers"})
        slug, group = row["id"], row["group"]
        if not isinstance(slug, str) or not slug.strip():
            raise LexiconError(f"traits.yml traits[{index}].id must be a string")
        if not isinstance(group, str) or not group.strip():
            raise LexiconError(f"traits.yml traits[{index}].group must be a string")
        if slug in seen:
            raise LexiconError(f"traits.yml has duplicate id {slug!r}")
        seen.add(slug)
        markers = _str_list(row["markers"], f"traits.yml traits.{slug}.markers", unique=True)
        if not markers:
            raise LexiconError(f"traits.yml traits.{slug} needs at least one marker")
        if "reconcile" in row:
            if row["reconcile"] != "body_wings":
                raise LexiconError(f"traits.yml traits.{slug} reconcile must be body_wings")
            reconcile.add(slug)
        patterns.append((slug, tuple(markers)))
    unknown_mine = [slug for slug in mine_only if slug not in seen]
    if unknown_mine:
        raise LexiconError(f"traits.yml mine_only names unknown ids {unknown_mine}")
    return {
        "patterns": tuple(patterns),
        "mine_only": tuple(mine_only),
        "reconcile": frozenset(reconcile),
    }


def build_categories(doc: dict[str, Any]) -> dict[str, Any]:
    """Return broad and colour-detail sets, appearance pin order, and popular titles.

    ``series_appearance`` is hair_color, then the angel-kit categories, in file
    order. ``series_split`` is the rare-look categories the engine may pin
    when a locked cast disagrees. The engine still decides when to ask them.
    """
    _require_keys(doc, "categories.yml", {"categories", "popular"}, {"categories", "popular"})
    rows = doc["categories"]
    if not isinstance(rows, dict) or not rows:
        raise LexiconError("categories.yml categories must be a mapping")
    broad: list[str] = []
    color_detail: list[str] = []
    appearance: list[str] = []
    split: list[str] = []
    flag_keys = {"broad", "color_detail", "series_appearance", "series_split"}
    for name, row in rows.items():
        if not isinstance(name, str) or not name:
            raise LexiconError("categories.yml category names must be strings")
        if not isinstance(row, dict):
            raise LexiconError(f"categories.yml categories.{name} must be a mapping")
        _require_keys(row, f"categories.yml categories.{name}", flag_keys, set())
        if not row:
            raise LexiconError(f"categories.yml categories.{name} needs a flag")
        for flag, value in row.items():
            if value is not True:
                raise LexiconError(f"categories.yml categories.{name}.{flag} must be true")
        if row.get("broad"):
            broad.append(name)
        if row.get("color_detail"):
            color_detail.append(name)
        if row.get("series_appearance"):
            appearance.append(name)
        if row.get("series_split"):
            split.append(name)
    popular = doc["popular"]
    if not isinstance(popular, dict) or not popular:
        raise LexiconError("categories.yml popular must be a mapping")
    titles: dict[str, tuple[str, ...]] = {}
    for medium, cats in popular.items():
        if not isinstance(medium, str) or not medium:
            raise LexiconError("categories.yml popular keys must be strings")
        titles[medium] = tuple(_str_list(cats, f"categories.yml popular.{medium}", unique=True))
        if not titles[medium]:
            raise LexiconError(f"categories.yml popular.{medium} must not be empty")
    if "hair_color" in appearance:
        raise LexiconError("hair_color is pinned in code ahead of series_appearance, not by that flag")
    return {
        "broad": frozenset(broad),
        "color_detail": frozenset(color_detail),
        "series_appearance": ("hair_color", *appearance),
        "series_split": tuple(split),
        "popular": titles,
    }


def build_characters(doc: dict[str, Any]) -> dict[str, Any]:
    """Return identity rows, headliner rows, and non-character titles.

    Identity names are whole spellings of one person. Headliner phrases are
    player-text markers; the engine applies the prior. A duplicate id, phrase,
    or spelling raises so two people cannot silently share a key.
    """
    _require_keys(
        doc, "characters.yml",
        {"identities", "headliners", "non_characters"},
        {"identities", "headliners", "non_characters"},
    )
    identities_raw = doc["identities"]
    if not isinstance(identities_raw, list) or not identities_raw:
        raise LexiconError("characters.yml identities must be a non-empty list")
    identities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, row in enumerate(identities_raw):
        where = f"characters.yml identities[{index}]"
        if not isinstance(row, dict):
            raise LexiconError(f"{where} must be a mapping")
        _require_keys(row, where, {"id", "names"}, {"id", "names"})
        slug = row["id"]
        if not isinstance(slug, str) or not slug.strip():
            raise LexiconError(f"{where}.id must be a string")
        if slug in seen_ids:
            raise LexiconError(f"characters.yml has duplicate id {slug!r}")
        seen_ids.add(slug)
        names = _str_list(row["names"], f"{where}.names", unique=True)
        if len(names) < 2:
            raise LexiconError(f"{where} needs at least two spellings")
        for name in names:
            key = " ".join(name.lower().split())
            if key in seen_names:
                raise LexiconError(f"characters.yml repeats spelling {name!r}")
            seen_names.add(key)
        identities.append({"id": slug, "canonical": names[0], "names": tuple(names)})
    head_raw = doc["headliners"]
    if not isinstance(head_raw, list) or not head_raw:
        raise LexiconError("characters.yml headliners must be a non-empty list")
    headliners: list[dict[str, Any]] = []
    seen_franchise: set[str] = set()
    seen_phrase: set[str] = set()
    for index, row in enumerate(head_raw):
        where = f"characters.yml headliners[{index}]"
        if not isinstance(row, dict):
            raise LexiconError(f"{where} must be a mapping")
        _require_keys(row, where, {"franchise", "phrases", "names"}, {"franchise", "phrases", "names"})
        franchise = row["franchise"]
        if not isinstance(franchise, str) or not franchise.strip():
            raise LexiconError(f"{where}.franchise must be a string")
        if franchise in seen_franchise:
            raise LexiconError(f"characters.yml repeats franchise {franchise!r}")
        seen_franchise.add(franchise)
        phrases = _str_list(row["phrases"], f"{where}.phrases", unique=True)
        names = _str_list(row["names"], f"{where}.names", unique=True)
        folded: list[str] = []
        for phrase in phrases:
            key = " ".join(phrase.lower().split())
            if key in seen_phrase:
                raise LexiconError(f"characters.yml repeats headliner phrase {phrase!r}")
            seen_phrase.add(key)
            folded.append(key)
        headliners.append({
            "franchise": franchise,
            "phrases": tuple(folded),
            "names": tuple(names),
        })
    blocked = _str_list(doc["non_characters"], "characters.yml non_characters", unique=True)
    return {
        "identities": tuple(identities),
        "headliners": tuple(headliners),
        "non_characters": tuple(blocked),
    }


def build_coverage(doc: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return search-coverage clusters: marker groups and the name queries they add.

    A group is AND; clusters fire when any group matches. The queries are
    retrieval strings only. The loader does not decide who to guess.
    """
    _require_keys(doc, "coverage.yml", {"clusters"}, {"clusters"})
    rows = doc["clusters"]
    if not isinstance(rows, list) or not rows:
        raise LexiconError("coverage.yml clusters must be a non-empty list")
    clusters: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        where = f"coverage.yml clusters[{index}]"
        if not isinstance(row, dict):
            raise LexiconError(f"{where} must be a mapping")
        _require_keys(row, where, {"id", "groups", "queries"}, {"id", "groups", "queries"})
        slug = row["id"]
        if not isinstance(slug, str) or not slug.strip():
            raise LexiconError(f"{where}.id must be a string")
        if slug in seen_ids:
            raise LexiconError(f"coverage.yml has duplicate id {slug!r}")
        seen_ids.add(slug)
        raw_groups = row["groups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            raise LexiconError(f"{where}.groups must be a non-empty list")
        groups: list[tuple[str, ...]] = []
        for gindex, group in enumerate(raw_groups):
            markers = _str_list(group, f"{where}.groups[{gindex}]", unique=True)
            if not markers:
                raise LexiconError(f"{where}.groups[{gindex}] must not be empty")
            folded = tuple(" ".join(marker.lower().split()) for marker in markers)
            if len(folded) != len(set(folded)):
                raise LexiconError(f"{where}.groups[{gindex}] repeats a marker")
            groups.append(folded)
        queries = _str_list(row["queries"], f"{where}.queries", unique=True)
        if not queries:
            raise LexiconError(f"{where}.queries must not be empty")
        clusters.append({"id": slug, "groups": tuple(groups), "queries": tuple(queries)})
    return tuple(clusters)


def load_lexicon() -> dict[str, Any]:
    """Load and validate every lexicon file. Called once at import."""
    return {
        "medium": build_medium(load_document("medium.yml")),
        "colors": build_colors(load_document("colors.yml")),
        "franchises": build_franchises(load_document("franchises.yml")),
        "traits": build_traits(load_document("traits.yml")),
        "categories": build_categories(load_document("categories.yml")),
        "characters": build_characters(load_document("characters.yml")),
        "coverage": build_coverage(load_document("coverage.yml")),
    }
