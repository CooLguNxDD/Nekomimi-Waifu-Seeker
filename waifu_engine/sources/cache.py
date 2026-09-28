import os
import json
import re

CACHE_DIR = os.path.join("data", "cache", "entities")

def _make_slug(name: str) -> str:
    """Create a safe filesystem slug from a name."""
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return slug or "unknown"

def get_entity(slug_or_name: str) -> dict | None:
    """Retrieve a cached character profile."""
    slug = _make_slug(slug_or_name)
    path = os.path.join(CACHE_DIR, f"{slug}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_entity(candidate_dict: dict) -> None:
    """Save a character profile to the entity cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = candidate_dict.get("name")
    if not name:
        return
    slug = _make_slug(name)
    path = os.path.join(CACHE_DIR, f"{slug}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(candidate_dict, f, indent=2)
    except Exception:
        pass
