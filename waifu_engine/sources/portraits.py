"""Fill a missing character portrait from an image search, not a wiki thumbnail.

Guess and win screens drew the cat because the stored row had ``image_url:
null``. Wikipedia page images are a poor source for that: fair-use files are
omitted, and many game characters have no English article. The portrait is
resolved the same way one-shot image attach works: a query built from the
character's name and series, then ``web_search.fetch_image_url``.
"""

from __future__ import annotations

import os
from typing import Any

from ..names import plain_duplicate

_NO_SERIES = {"", "unknown", "web result"}


def portraits_enabled() -> bool:
    """``WAIFU_PORTRAITS`` (default on). Tests turn it off so a guess stays offline."""
    return os.getenv("WAIFU_PORTRAITS", "1").strip().lower() in {"1", "true", "yes"}


def usable_image(url: str | None) -> str | None:
    """An HTTPS image URL, or None. ``http`` is upgraded; SVG icons are not portraits."""
    text = (url or "").strip()
    if text.startswith("http://"):
        text = "https://" + text[len("http://"):]
    if not text.startswith("https://"):
        return None
    path = text.split("?", 1)[0].lower()
    if path.endswith(".svg"):
        return None
    return text


def _named(series: str) -> bool:
    """Whether ``series`` is a real work title worth putting in the image query."""
    return (series or "").strip().lower() not in _NO_SERIES


def donate_image(rows: list[dict[str, Any]], incoming: dict[str, Any]) -> bool:
    """Copy ``incoming``'s image onto the same character already in ``rows``.

    Search keeps the first source. A later hit often carries the only real
    portrait, and used to be discarded as a duplicate name.
    """
    url = usable_image(incoming.get("image_url"))
    if not url:
        return False
    name = incoming.get("name") or ""
    for row in rows:
        if not plain_duplicate(name, [row.get("name") or ""]):
            continue
        if usable_image(row.get("image_url")):
            return False
        row["image_url"] = url
        return True
    return False


def portrait_query(name: str, series: str = "", medium: str = "") -> str:
    """Image-search text for this character: name, then series or medium.

    The series is what separates "Hoshino Takanashi" (Blue Archive) from
    other Hoshinos. ``fetch_image_url`` must be called with ``extra=""``:
    its default suffix is " anime character", which misses game art.
    """
    parts = [(name or "").strip()]
    if _named(series):
        parts.append(series.strip())
    elif (medium or "").strip().lower() not in {"", "unknown"}:
        parts.append(medium.strip())
    parts.append("character")
    return " ".join(p for p in parts if p)


def lookup_portrait(name: str, series: str = "", medium: str = "") -> str | None:
    """HTTPS portrait from an image search for ``name``, or None.

    A single given name with no series is skipped: "Hoshino" is several
    people, and the search would attach the wrong face.
    """
    if not portraits_enabled() or not (name or "").strip():
        return None
    if len(name.split()) < 2 and not _named(series):
        return None
    from ..web_search import fetch_image_url

    try:
        return usable_image(fetch_image_url(portrait_query(name, series, medium), extra=""))
    except Exception:  # noqa: BLE001 - image search is optional
        return None


def _read_image(row: Any) -> str | None:
    """Current image on a dict or a session candidate."""
    if isinstance(row, dict):
        return row.get("image_url")
    return getattr(row, "image_url", None)


def _field(row: Any, key: str) -> str:
    """``key`` from a dict or a session candidate."""
    if isinstance(row, dict):
        return row.get(key) or ""
    return getattr(row, key, "") or ""


def _write_image(row: Any, url: str) -> None:
    """Store ``url`` on a dict or a session candidate."""
    if isinstance(row, dict):
        row["image_url"] = url
    else:
        row.image_url = url


def fill_portraits(rows: list[Any], limit: int = 4) -> int:
    """Set ``image_url`` on up to ``limit`` rows that do not have one yet.

    Returns how many were filled. Each fill is one image search for that
    character's name and series. Already-set URLs are left alone.
    """
    if not portraits_enabled():
        return 0
    filled = 0
    for row in rows:
        if filled >= limit:
            break
        if usable_image(_read_image(row)):
            continue
        url = lookup_portrait(_field(row, "name"), _field(row, "series"), _field(row, "medium"))
        if not url:
            continue
        _write_image(row, url)
        filled += 1
    return filled
