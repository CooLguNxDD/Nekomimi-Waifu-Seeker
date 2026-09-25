"""Fill a missing character portrait from sources that already publish one.

Wikipedia pages often have no page image (or the page does not exist), and a
later AniList hit for the same name used to be dropped, so the guess kept
``image_url: null`` and the UI drew the cat. This looks up a public HTTPS
image and writes it onto the candidate. It never invents a URL.
"""

from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
from typing import Any

from ..names import name_keys, plain_duplicate, same_character, same_series

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
    """Whether ``series`` is a real work title we can match against."""
    return (series or "").strip().lower() not in _NO_SERIES


def accepts_portrait(name: str, series: str, hit_name: str, hit_series: str) -> bool:
    """Whether ``hit`` is the same person, not a different character who shares a token.

    "Hoshino" must not take Ai Hoshino's face. A known series has to agree
    ("Blue Archive" with "Blue Archive The Animation"). An exact name match
    is enough when the series is unknown.
    """
    if not name or not hit_name or not same_character(name, hit_name):
        return False
    if _named(series) and _named(hit_series) and not same_series(series, hit_series):
        return False
    if name_keys(name) != name_keys(hit_name):
        return _named(series) and _named(hit_series) and same_series(series, hit_series)
    return True


def donate_image(rows: list[dict[str, Any]], incoming: dict[str, Any]) -> bool:
    """Copy ``incoming``'s image onto the same character already in ``rows``.

    Search keeps the first source. AniList often arrives second, with the
    only real portrait, and used to be discarded as a duplicate name.
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


def _from_anilist(name: str, series: str) -> str | None:
    """AniList ``image.large`` for this person, when the name and series agree."""
    from . import anilist

    hits = anilist.search_characters(name, limit=5) or []
    for hit in hits:
        if accepts_portrait(name, series, hit.get("name") or "", hit.get("series") or ""):
            url = usable_image(hit.get("image_url"))
            if url:
                return url
    return None


def _wiki_title(source_url: str) -> str:
    """Article title from an English Wikipedia ``/wiki/`` URL, or ``""``."""
    parsed = urllib.parse.urlparse(source_url or "")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host != "en.wikipedia.org" or not parsed.path.startswith("/wiki/"):
        return ""
    title = urllib.parse.unquote(parsed.path[len("/wiki/"):]).strip()
    ns = title.split(":", 1)[0].lower() if ":" in title else ""
    if ns in {"file", "category", "special", "help", "wikipedia", "template", "portal"}:
        return ""
    return title


def _wikipedia_rest_image(source_url: str) -> str | None:
    """Lead image from the REST summary when ``pageimages`` left the thumbnail empty.

    MediaWiki ``prop=pageimages`` omits many fair-use files. The summary
    endpoint still returns ``originalimage`` / ``thumbnail`` for those pages
    (Abby in The Last of Us Part II).
    """
    title = _wiki_title(source_url)
    if not title:
        return None
    from ._http import get_json

    quoted = urllib.parse.quote(title.replace(" ", "_"), safe="()_")
    data = get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quoted}", {})
    if not isinstance(data, dict):
        return None
    original = (data.get("originalimage") or {}).get("source")
    thumb = (data.get("thumbnail") or {}).get("source")
    return usable_image(original or thumb)


def _from_wikipedia(name: str, series: str) -> str | None:
    """Wikipedia page image for a character page whose title is this person."""
    from . import wikipedia

    query = name if not _named(series) else f"{name} {series}"
    hits = wikipedia.search_characters(query, limit=5) or []
    for hit in hits:
        if not accepts_portrait(name, series, hit.get("name") or "", hit.get("series") or ""):
            continue
        url = usable_image(hit.get("image_url")) or _wikipedia_rest_image(hit.get("source_url") or "")
        if url:
            return url
    return None


_FILE_NOISE = {
    "splash", "image", "portrait", "thumb", "character", "file", "wiki",
    "icon", "large", "small", "full", "body", "face", "art", "png", "jpg",
    "jpeg", "webp", "render", "official",
}
_TOKENS = re.compile(r"[a-z0-9]{2,}")


def _tokens(text: str) -> set[str]:
    """Alphabetic pieces of a name or a file title, lower-cased."""
    return set(_TOKENS.findall((text or "").lower()))


def _filename_conflicts(name: str, image_url: str) -> bool:
    """Whether the file title names a different person who shares a prefix.

    ``endfield.wiki.gg/wiki/Rossina`` publishes Rossi's splash. The page
    title and the file are "Rossi", which is not Rossina.
    """
    path = urllib.parse.urlparse(image_url or "").path
    file_tokens = _tokens(path.rsplit("/", 1)[-1]) - _FILE_NOISE
    name_tokens = _tokens(name)
    for tok in file_tokens:
        if tok in name_tokens or len(tok) < 4:
            continue
        if any(
            tok != other and min(len(tok), len(other)) >= 4
            and (other.startswith(tok) or tok.startswith(other))
            for other in name_tokens
        ):
            return True
    return False


def _page_is_this_person(name: str, series: str, page_name: str, page_series: str) -> bool:
    """Whether a cited page's title is this character, not a neighbour.

    ``accepts_portrait`` requires the whole name. A wiki title is often only
    the given name ("Rossina"), which still has to match a real token.
    "Rossi" and "Ai Hoshino" do not.
    """
    if accepts_portrait(name, series, page_name, page_series or series):
        return True
    page_tokens = _tokens(page_name)
    name_tokens = _tokens(name)
    if not page_tokens or not page_tokens <= name_tokens:
        return False
    return any(len(tok) >= 5 for tok in page_tokens)


def _from_source_page(url: str, name: str = "", series: str = "") -> str | None:
    """``og:image`` on a page we already cite, when the page is this person.

    Playwright enrich is what used to read this. A Colab session often has no
    browser, so the guess kept a null image even when ``source_url`` pointed
    at a page that publishes one. A title that is a different character
    (Rossi's splash on a Rossina URL) is refused.
    """
    if not (url or "").startswith("https://"):
        return None
    from ._http import USER_AGENT
    from ..browser_search import parse_character_page

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read(200_000).decode("utf-8", "replace")
    except (OSError, ValueError):
        return None
    info = parse_character_page(url, html) or {}
    image = usable_image(info.get("image_url"))
    if not image or _filename_conflicts(name, image):
        return None
    page_name = info.get("name") or ""
    if name and page_name and not _page_is_this_person(
        name, series, page_name, info.get("series") or ""
    ):
        return None
    return image


def _from_ddg(name: str, series: str) -> str | None:
    """Last resort for a distinctive name. A single token is too ambiguous."""
    if len(name.split()) < 2:
        return None
    from ..web_search import fetch_image_url

    query = name if not _named(series) else f"{name} {series}"
    try:
        return usable_image(fetch_image_url(query, extra=""))
    except Exception:  # noqa: BLE001 - image search is optional
        return None


def lookup_portrait(name: str, series: str = "", source_url: str = "") -> str | None:
    """A public portrait URL for ``name``, or None when every source is silent."""
    if not portraits_enabled() or not (name or "").strip():
        return None
    # The candidate's own article is already this person. pageimages often
    # omitted the fair-use lead; the summary still has it.
    own = _wikipedia_rest_image(source_url)
    if own:
        return own
    for found in (
        _from_anilist(name, series),
        _from_wikipedia(name, series),
        _from_source_page(source_url, name, series),
        _from_ddg(name, series),
    ):
        if found:
            return found
    return None


def _read_image(row: Any) -> str | None:
    """Current image on a dict or a session candidate."""
    if isinstance(row, dict):
        return row.get("image_url")
    return getattr(row, "image_url", None)


def _write_image(row: Any, url: str) -> None:
    """Store ``url`` on a dict or a session candidate."""
    if isinstance(row, dict):
        row["image_url"] = url
    else:
        row.image_url = url


def fill_portraits(rows: list[Any], limit: int = 4) -> int:
    """Set ``image_url`` on up to ``limit`` rows that do not have one yet.

    Returns how many were filled. Already-set URLs are left alone.
    """
    if not portraits_enabled():
        return 0
    filled = 0
    for row in rows:
        if filled >= limit:
            break
        if usable_image(_read_image(row)):
            continue
        name = row.get("name") if isinstance(row, dict) else getattr(row, "name", "")
        series = row.get("series") if isinstance(row, dict) else getattr(row, "series", "")
        source = row.get("source_url") if isinstance(row, dict) else getattr(row, "source_url", "")
        url = lookup_portrait(name or "", series or "", source or "")
        if not url:
            continue
        _write_image(row, url)
        filled += 1
    return filled
