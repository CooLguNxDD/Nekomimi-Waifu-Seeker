"""Process-wide headless Chromium for character-page search.

Playwright is optional. When the package, browser, or launch fails, ``available()``
is False and callers fill with DuckDuckGo. Search is sourcing only — Laya still
only answers choice/noul/score.

Mirrors ``nekomimi.laya_client``: one shared object, a lock, never raises out of
the public helpers.
"""

from __future__ import annotations

import html as html_lib
import re
import threading
import time
import urllib.parse
from typing import Any

_BROWSER: Any = None
_PLAYWRIGHT: Any = None
_LOAD_FAILED = False
_LOCK = threading.RLock()

NAV_TIMEOUT_MS = 8000
PAGE_BUDGET_S = 12.0
SEARCH_BUDGET_S = 8.0
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

_FANDOM_SKIP = re.compile(
    r"/(?:Special:|File:|Category:|Template:|Help:|User:|Talk:|Map:)",
    re.I,
)
_A_HREF = re.compile(
    r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.I | re.S,
)


def _import_sync_playwright() -> Any:
    from playwright.sync_api import sync_playwright

    return sync_playwright


def _launch_browser(playwright: Any) -> Any:
    return playwright.chromium.launch(headless=True, timeout=NAV_TIMEOUT_MS)


def get_browser() -> Any | None:
    """Return the shared Chromium browser, or None if it cannot be launched."""
    global _BROWSER, _PLAYWRIGHT, _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    if _BROWSER is not None:
        return _BROWSER
    with _LOCK:
        if _BROWSER is not None:
            return _BROWSER
        if _LOAD_FAILED:
            return None
        try:
            sync_playwright = _import_sync_playwright()
            _PLAYWRIGHT = sync_playwright().start()
            _BROWSER = _launch_browser(_PLAYWRIGHT)
        except Exception:  # noqa: BLE001 - missing pkg / browser / launch
            _LOAD_FAILED = True
            _BROWSER = None
            _PLAYWRIGHT = None
            return None
    return _BROWSER


def available() -> bool:
    """True when a headless Chromium can be used. Never raises."""
    if _LOAD_FAILED:
        return False
    if _BROWSER is not None:
        return True
    try:
        _import_sync_playwright()
    except Exception:  # noqa: BLE001
        return False
    return get_browser() is not None


def reset() -> None:
    """Drop the cached browser (tests)."""
    global _BROWSER, _PLAYWRIGHT, _LOAD_FAILED
    with _LOCK:
        if _BROWSER is not None:
            try:
                _BROWSER.close()
            except Exception:  # noqa: BLE001
                pass
        if _PLAYWRIGHT is not None:
            try:
                _PLAYWRIGHT.stop()
            except Exception:  # noqa: BLE001
                pass
        _BROWSER = None
        _PLAYWRIGHT = None
        _LOAD_FAILED = False


def _strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _abs(href: str, base: str) -> str:
    href = html_lib.unescape((href or "").strip())
    if not href or href.startswith(("#", "javascript:", "data:")):
        return ""
    return urllib.parse.urljoin(base, href)


def _meta_props(html: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"<meta\b([^>]+)/?>", html or "", re.I):
        tag = m.group(1)
        prop = re.search(r'(?:property|name)=["\']([^"\']+)["\']', tag, re.I)
        content = re.search(r'content=["\']([^"\']*)["\']', tag, re.I)
        if not prop or not content:
            continue
        key = prop.group(1).lower()
        if key.startswith("og:"):
            key = key[3:]
        out[key] = html_lib.unescape(content.group(1))
    return out


def _first_paragraph(html: str) -> str:
    m = re.search(r"<p\b[^>]*>(.*?)</p>", html or "", re.I | re.S)
    if not m:
        return ""
    return _strip_tags(m.group(1))[:700]


def _first_link_text(html: str, href_re: str) -> str:
    pat = re.compile(
        rf'<a\b[^>]+href=["\'](?:https?://[^"\']+)?{href_re}["\'][^>]*>(.*?)</a>',
        re.I | re.S,
    )
    m = pat.search(html or "")
    return _strip_tags(m.group(1))[:80] if m else ""


def _body_after(html: str, end: int, width: int = 400) -> str:
    return _strip_tags(html[end : end + width])[:240]


def _hit(title: str, href: str, body: str) -> dict[str, str] | None:
    title = _strip_tags(title)
    href = (href or "").strip()
    body = _strip_tags(body)
    if not title or not href:
        return None
    return {"title": title[:120], "href": href, "body": body[:320]}


def parse_mal_search(html: str, base: str = "https://myanimelist.net") -> list[dict[str, str]]:
    """MAL ``character.php`` result rows → ``{title, href, body}``."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(
        r'<a\b[^>]+href=["\']((?:https?://(?:www\.)?myanimelist\.net)?/character/\d+/[^"\']+)["\'][^>]*>(.*?)</a>',
        html or "",
        re.I | re.S,
    ):
        href = _abs(m.group(1).split("?")[0], base)
        title = _strip_tags(m.group(2))
        if not title or href in seen:
            continue
        seen.add(href)
        body = _body_after(html, m.end())
        item = _hit(title, href, body)
        if item:
            out.append(item)
    return out


def parse_fandom_search(html: str, base: str = "https://www.fandom.com") -> list[dict[str, str]]:
    """Fandom unified / wiki search hits."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(
        r'<a\b[^>]+href=["\'](https?://[^"\']+\.fandom\.com/wiki/[^"\']+|/?wiki/[^"\']+)["\'][^>]*>(.*?)</a>',
        html or "",
        re.I | re.S,
    ):
        href = _abs(m.group(1).split("?")[0], base)
        if not href or _FANDOM_SKIP.search(href) or href in seen:
            continue
        title = _strip_tags(m.group(2))
        if not title:
            continue
        seen.add(href)
        body = _body_after(html, m.end())
        item = _hit(title, href, body)
        if item:
            out.append(item)
    return out


def parse_vndb_search(html: str, base: str = "https://vndb.org") -> list[dict[str, str]]:
    """VNDB character browse/search (``/c<id>``)."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(
        r'<a\b[^>]+href=["\']((?:https?://vndb\.org)?/c\d+)["\'][^>]*>(.*?)</a>',
        html or "",
        re.I | re.S,
    ):
        href = _abs(m.group(1), base)
        title = _strip_tags(m.group(2))
        if not title or href in seen:
            continue
        seen.add(href)
        body = _body_after(html, m.end())
        item = _hit(title, href, body)
        if item:
            out.append(item)
    return out


def parse_comicvine_search(
    html: str, base: str = "https://comicvine.gamespot.com"
) -> list[dict[str, str]]:
    """ComicVine character search (type id ``4005``)."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(
        r'<a\b[^>]+href=["\']((?:https?://comicvine\.gamespot\.com)?/[^"\']+/4005-\d+/?)["\'][^>]*>(.*?)</a>',
        html or "",
        re.I | re.S,
    ):
        href = _abs(m.group(1), base)
        title = _strip_tags(m.group(2))
        if not title or href in seen:
            continue
        seen.add(href)
        body = _body_after(html, m.end())
        item = _hit(title, href, body)
        if item:
            out.append(item)
    return out


def parse_ddg_html(html: str) -> list[dict[str, str]]:
    """DuckDuckGo HTML SERP (Playwright last resort, not the ``ddgs`` library)."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(
        r'<a\b[^>]*class=["\'][^"\']*result__a[^"\']*["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        html or "",
        re.I | re.S,
    ):
        href = _unwrap_ddg(m.group(1))
        title = _strip_tags(m.group(2))
        if not href or not title or href in seen:
            continue
        seen.add(href)
        body = _body_after(html, m.end())
        item = _hit(title, href, body)
        if item:
            out.append(item)
    if out:
        return out
    # Some DDG HTML builds put the class on a parent, href on the child.
    for m in _A_HREF.finditer(html or ""):
        href = _unwrap_ddg(m.group(1))
        title = _strip_tags(m.group(2))
        if not href.startswith("http") or not title or href in seen:
            continue
        if "duckduckgo.com" in href and "uddg=" not in m.group(1):
            continue
        seen.add(href)
        item = _hit(title, href, _body_after(html, m.end()))
        if item:
            out.append(item)
    return out


def _unwrap_ddg(href: str) -> str:
    href = html_lib.unescape(href or "")
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs:
        return qs["uddg"][0]
    return href


def parse_search_html(kind: str, html: str) -> list[dict[str, str]]:
    if kind == "mal":
        return parse_mal_search(html)
    if kind == "fandom":
        return parse_fandom_search(html)
    if kind == "vndb":
        return parse_vndb_search(html)
    if kind == "comicvine":
        return parse_comicvine_search(html)
    if kind == "ddg":
        return parse_ddg_html(html)
    return []


def parse_og(html: str) -> dict[str, str]:
    meta = _meta_props(html)
    out: dict[str, str] = {}
    title = (meta.get("title") or "").strip()
    if title:
        out["name"] = title.split(" | ")[0].split(" - ")[0].strip()[:80]
    desc = (meta.get("description") or "").strip()
    if desc:
        out["blurb"] = desc[:900]
    image = (meta.get("image") or "").strip()
    if image and not image.lower().endswith(".svg"):
        out["image_url"] = image
    return out


def parse_mal_character(html: str) -> dict[str, str]:
    out = parse_og(html)
    series = _first_link_text(html, r"/anime/\d+/[^\"']+") or _first_link_text(
        html, r"/manga/\d+/[^\"']+"
    )
    if series:
        out["series"] = series
    if not out.get("blurb"):
        p = _first_paragraph(html)
        if p:
            out["blurb"] = p
    if not out.get("image_url"):
        m = re.search(
            r'<img\b[^>]+src=["\'](https://cdn\.myanimelist\.net/images/characters/[^"\']+)["\']',
            html or "",
            re.I,
        )
        if m:
            out["image_url"] = html_lib.unescape(m.group(1))
    return out


def parse_fandom_page(html: str) -> dict[str, str]:
    out = parse_og(html)
    m = re.search(
        r'data-source=["\'](?:series|anime|game|manga|franchise|title)["\'][^>]*>'
        r'.*?<div\b[^>]*pi-data-value[^>]*>(.*?)</div>',
        html or "",
        re.I | re.S,
    )
    if m:
        series = _strip_tags(m.group(1))[:80]
        if series:
            out["series"] = series
    if not out.get("blurb"):
        p = _first_paragraph(html)
        if p:
            out["blurb"] = p
    if not out.get("image_url"):
        img = re.search(
            r'<img\b[^>]+(?:data-src|src)=["\'](https://(?:static|vignette)\.wikia\.nocookie\.net/[^"\']+)["\']',
            html or "",
            re.I,
        )
        if img:
            out["image_url"] = html_lib.unescape(img.group(1))
    return out


def parse_vndb_character(html: str) -> dict[str, str]:
    out = parse_og(html)
    series = _first_link_text(html, r"/v\d+")
    if series:
        out["series"] = series
    if not out.get("name"):
        h = re.search(r"<h1\b[^>]*>(.*?)</h1>", html or "", re.I | re.S)
        if h:
            out["name"] = _strip_tags(h.group(1))[:80]
    if not out.get("blurb"):
        p = _first_paragraph(html)
        if p:
            out["blurb"] = p
    if not out.get("image_url"):
        img = re.search(
            r'<img\b[^>]+src=["\'](https://s\.vndb\.org/[^"\']+)["\']',
            html or "",
            re.I,
        )
        if img:
            out["image_url"] = html_lib.unescape(img.group(1))
    return out


def parse_comicvine_page(html: str) -> dict[str, str]:
    out = parse_og(html)
    series = _first_link_text(html, r"/[^\"']+/4015-\d+") or _first_link_text(
        html, r"/[^\"']+/4010-\d+"
    )
    if series:
        out["series"] = series
    if not out.get("blurb"):
        p = _first_paragraph(html)
        if p:
            out["blurb"] = p
    return out


def parse_character_page(url: str, html: str) -> dict[str, str] | None:
    """Pull name/series/blurb/image from a character page. Never raises."""
    try:
        low = (url or "").lower()
        if "myanimelist.net" in low:
            info = parse_mal_character(html)
        elif "fandom.com" in low or "wikia.com" in low:
            info = parse_fandom_page(html)
        elif "vndb.org" in low:
            info = parse_vndb_character(html)
        elif "comicvine" in low:
            info = parse_comicvine_page(html)
        else:
            info = parse_og(html)
            if not info.get("blurb"):
                p = _first_paragraph(html)
                if p:
                    info["blurb"] = p
        return info or None
    except Exception:  # noqa: BLE001
        return None


def _site_targets(query: str, medium_hint: str | None) -> list[tuple[str, str]]:
    q = urllib.parse.quote_plus((query or "").strip()[:80] or "character")
    mal = ("mal", f"https://myanimelist.net/character.php?cat=character&q={q}")
    fandom = ("fandom", f"https://www.fandom.com/search?query={q}")
    vndb = ("vndb", f"https://vndb.org/c?q={q}")
    vine = ("comicvine", f"https://comicvine.gamespot.com/search/?q={q}&i=character")
    hint = (medium_hint or "").strip().lower()
    if hint in {"anime", "manga"}:
        return [mal, fandom]
    if hint == "game":
        return [vndb, fandom]
    if hint == "comic":
        return [vine, fandom]
    return [mal, fandom, vndb, vine]


def _fetch_html(
    url: str,
    *,
    nav_timeout_ms: int = NAV_TIMEOUT_MS,
    page_budget_s: float = PAGE_BUDGET_S,
) -> str | None:
    browser = get_browser()
    if browser is None:
        return None
    start = time.monotonic()
    with _LOCK:
        page = None
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.set_default_navigation_timeout(nav_timeout_ms)
            page.set_default_timeout(nav_timeout_ms)
            page.goto(url, timeout=nav_timeout_ms, wait_until="domcontentloaded")
            if time.monotonic() - start > page_budget_s:
                return None
            return page.content()
        except Exception:  # noqa: BLE001
            return None
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    pass


def search(
    query: str,
    medium_hint: str | None = None,
    limit: int = 12,
) -> list[dict[str, str]]:
    """Hit MAL / Fandom / VNDB / ComicVine indexes. Returns ``{title, href, body}``.

    Never raises. Empty list on any failure. Aborts around ``SEARCH_BUDGET_S``.
    """
    if not query or limit <= 0:
        return []
    try:
        if not available():
            return []
    except Exception:  # noqa: BLE001
        return []
    deadline = time.monotonic() + SEARCH_BUDGET_S
    hits: list[dict[str, str]] = []
    seen: set[str] = set()

    def take(items: list[dict[str, str]]) -> None:
        for item in items:
            href = item.get("href") or ""
            if not href or href in seen:
                continue
            seen.add(href)
            hits.append(item)
            if len(hits) >= limit:
                return

    try:
        for kind, url in _site_targets(query, medium_hint):
            if len(hits) >= limit or time.monotonic() >= deadline:
                break
            remain_ms = int(max(200, (deadline - time.monotonic()) * 1000))
            html = _fetch_html(
                url,
                nav_timeout_ms=min(NAV_TIMEOUT_MS, remain_ms),
                page_budget_s=min(PAGE_BUDGET_S, max(0.2, deadline - time.monotonic())),
            )
            if not html:
                continue
            take(parse_search_html(kind, html))
        if not hits and time.monotonic() < deadline:
            q = urllib.parse.quote_plus(f"{query.strip()[:80]} character")
            remain_ms = int(max(200, (deadline - time.monotonic()) * 1000))
            html = _fetch_html(
                f"https://html.duckduckgo.com/html/?q={q}",
                nav_timeout_ms=min(NAV_TIMEOUT_MS, remain_ms),
                page_budget_s=min(PAGE_BUDGET_S, max(0.2, deadline - time.monotonic())),
            )
            if html:
                take(parse_ddg_html(html))
    except Exception:  # noqa: BLE001
        return hits[:limit]
    return hits[:limit]


def enrich(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Mutate ``blurb`` / ``tags`` / ``image_url`` / ``series`` from the source page.

    Returns the candidate, or None on skip/failure. Never raises.
    """
    url = (candidate.get("source_url") or "").strip()
    if not url:
        return None
    try:
        from .web_search import (
            JUNK_DOMAINS,
            _guess_series,
            franchise_label,
            mine_trait_slugs,
            series_is_crumb,
        )

        if any(d in url.lower() for d in JUNK_DOMAINS):
            return None
        html = _fetch_html(url)
        if not html:
            return None
        info = parse_character_page(url, html)
        if not info:
            return None
        blurb = (info.get("blurb") or "").strip()
        if blurb and len(blurb) > len(candidate.get("blurb") or ""):
            candidate["blurb"] = blurb[:900]
        image = (info.get("image_url") or "").strip()
        if image and not candidate.get("image_url"):
            candidate["image_url"] = image
        series = (info.get("series") or "").strip()
        # A usable infobox series is direct evidence: normalise it, but never
        # replace it with a franchise the blurb merely mentions (a Vocaloid
        # collaboration). Only a crumb such as "Internet meme" falls back to
        # the page text, which can still map Crypton / Vocaloid.
        if series and not series_is_crumb(series):
            named = franchise_label(series)
        else:
            named = franchise_label(
                f"{candidate.get('name', '')} {info.get('blurb') or candidate.get('blurb') or ''}"
            )
        if series_is_crumb(candidate.get("series") or ""):
            candidate["series"] = ""
        if named:
            candidate["series"] = named
        elif series_is_crumb(series):
            series = ""
        if series and candidate.get("series") in ("", "Web result", None):
            candidate["series"] = series
        elif not candidate.get("series") or candidate.get("series") == "Web result":
            guessed = _guess_series(f"{candidate.get('name', '')} {candidate.get('blurb', '')}")
            if guessed and guessed != "Web result":
                candidate["series"] = guessed
        tags = candidate.setdefault("tags", [])
        for slug in mine_trait_slugs(candidate.get("blurb") or ""):
            if slug not in tags:
                tags.append(slug)
        if "playwright" not in tags:
            tags.append("playwright")
        return candidate
    except Exception:  # noqa: BLE001
        return None
