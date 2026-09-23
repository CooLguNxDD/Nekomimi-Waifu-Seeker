"""Optional search-query rewriter over any OpenAI-compatible chat endpoint.

Laya cannot write text, so by default search queries come from templates. When
``WAIFU_QUERY_LLM=1`` a small chat model turns the player's confirmed facts
into a few web-search phrases instead. It writes **search strings only** --
never question text (that lives in ``nekomimi.traits``) and never decisions
(those are Laya's).

Input is player-supplied facts only: the seed, typed details and "yes" answers.
Scraped names and blurbs are never sent, which keeps web content from steering
the model. Negatives are not sent either; search engines ignore negation.

The default target is a local vLLM / llama.cpp server hosting
``Qwen/Qwen3.6-35B-A3B``. Point ``WAIFU_QUERY_LLM_BASE_URL`` at
``https://api.openai.com/v1`` (and pick an OpenAI model) to use OpenAI.

Stdlib HTTP only; ``rewrite`` never raises and returns ``None`` on any failure,
in which case the template queries are used unchanged.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from functools import lru_cache

_YES = {"1", "true", "yes"}
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
MAX_QUERY_CHARS = 120

SYSTEM_PROMPT = (
    "You write web-search queries for finding one fictional character from anime, "
    "manga, comics or video games. Use only the facts given. Reply with a JSON "
    "array of at most {n} short search queries and nothing else."
)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_ARRAY = re.compile(r"\[.*?\]", re.S)


def enabled() -> bool:
    if os.getenv("WAIFU_ONLINE_SEARCH", "1").lower() not in _YES:
        return False
    return os.getenv("WAIFU_QUERY_LLM", "0").lower() in _YES


def base_url() -> str:
    return os.getenv("WAIFU_QUERY_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def model() -> str:
    return os.getenv("WAIFU_QUERY_LLM_MODEL", DEFAULT_MODEL)


def _timeout() -> float:
    try:
        return float(os.getenv("WAIFU_QUERY_LLM_TIMEOUT", "20"))
    except ValueError:
        return 20.0


def _is_openai(url: str) -> bool:
    return "api.openai.com" in url


def _request_body(facts: tuple[str, ...], medium_hint: str | None, n: int) -> dict:
    lines = [f"- {f}" for f in facts]
    if medium_hint:
        lines.append(f"- medium: {medium_hint}")
    body = {
        "model": model(),
        "temperature": 0,
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
            {"role": "user", "content": "Facts:\n" + "\n".join(lines)},
        ],
    }
    if not _is_openai(base_url()):
        # Qwen3 thinks by default; vLLM/SGLang honour this switch. OpenAI
        # rejects unknown fields, so it never gets it.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def parse_queries(text: str, n: int = 3) -> list[str] | None:
    """Pull a JSON array of query strings out of a model reply."""
    text = _THINK.sub("", text or "")
    match = _ARRAY.search(text)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(items, list):
        return None
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        q = " ".join(item.split())[:MAX_QUERY_CHARS].strip()
        if q and q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out or None


@lru_cache(maxsize=256)
def _rewrite_cached(facts: tuple[str, ...], medium_hint: str | None, n: int,
                    url: str, model_id: str) -> tuple[str, ...] | None:
    headers = {"Content-Type": "application/json"}
    key = os.getenv("WAIFU_QUERY_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(_request_body(facts, medium_hint, n)).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_timeout()) as resp:  # noqa: S310 - configured URL
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    queries = parse_queries(content, n)
    return tuple(queries) if queries else None


def rewrite(facts: list[str], medium_hint: str | None = None, n: int = 3) -> list[str] | None:
    """Up to ``n`` search queries for these player facts, or ``None``. Never raises."""
    if not enabled():
        return None
    clean = tuple(dict.fromkeys(" ".join(f.split())[:200] for f in facts if f and f.strip()))
    if not clean:
        return None
    try:
        got = _rewrite_cached(clean, medium_hint, n, base_url(), model())
    except Exception:  # noqa: BLE001 - a dead endpoint must not kill the turn
        return None
    return list(got) if got else None
