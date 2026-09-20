from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
DATA_PATH = next(
    (p for p in (_HERE.parent / "data" / "catalog.json", _HERE / "data" / "catalog.json") if p.exists()),
    _HERE.parent / "data" / "catalog.json",
)

@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    text = DATA_PATH.read_text(encoding="utf-8-sig")
    data = json.loads(text)
    return data if isinstance(data, list) else []

def character_text(c: dict[str, Any]) -> str:
    tags = " ".join(c.get("tags") or [])
    return f"{c['name']} ({c['series']}): {tags}. {c.get('blurb', '')}"

def clear_catalog_cache() -> None:
    load_catalog.cache_clear()