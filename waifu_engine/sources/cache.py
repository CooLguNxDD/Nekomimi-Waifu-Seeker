import hashlib
import json
import os
import re
import unicodedata

CACHE_DIR = os.path.join("data", "cache", "entities")

def _legacy_slug(name: str) -> str:
    """Reproduce the old ASCII cache key for validated read-only migration."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"

def _make_slug(name: str) -> str:
    """Create a safe cache key whose digest distinguishes Unicode names."""
    normalized = " ".join(unicodedata.normalize("NFKC", name or "").casefold().split())
    readable = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")[:64] or "entity"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{readable}-{digest}"


def _matches_cached_entity(candidate: dict, name: str, series: str) -> bool:
    """Reject same-name cache entries from a different requested work."""
    from ..names import same_character, same_series, series_key

    cached_name = str(candidate.get("name") or "")
    if not cached_name or not same_character(cached_name, name):
        return False
    if not series_key(series):
        return True
    cached_series = str(candidate.get("series") or "")
    return bool(series_key(cached_series) and same_series(cached_series, series))


def get_entity(name: str, series: str = "") -> dict | None:
    """Retrieve a cached profile only when its character and requested work agree."""
    keys = list(dict.fromkeys((_make_slug(name), _legacy_slug(name))))
    for slug in keys:
        path = os.path.join(CACHE_DIR, f"{slug}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                candidate = json.load(f)
            if isinstance(candidate, dict) and _matches_cached_entity(candidate, name, series):
                return candidate
        except Exception:
            pass
    return None

def save_entity(candidate_dict: dict) -> None:
    """Save a character profile to the entity cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = candidate_dict.get("name")
    if not name:
        return
    slug = _make_slug(str(name))
    path = os.path.join(CACHE_DIR, f"{slug}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(candidate_dict, f, indent=2)
    except Exception:
        pass
