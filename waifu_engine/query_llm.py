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

The game loop never waits for it: ``prefetch`` runs the call on one background
worker and ``peek`` picks the result up on a later search. Stdlib HTTP only;
nothing here raises, and on failure the template queries are used unchanged.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from . import google_config, timing

_YES = {"1", "true", "yes"}
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
MAX_QUERY_CHARS = 120

SYSTEM_PROMPT = (
    "You write web-search queries for finding one fictional character from anime, "
    "manga, comics, video games, movies or TV series. Use only the facts given. "
    "Reply with a JSON array of at most {n} short search queries and nothing else."
)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_ARRAY = re.compile(r"\[.*?\]", re.S)


# ``_call``'s endpoint argument for the Gemini backend, in place of a URL.
GEMINI = "gemini"


def provider() -> str:
    """``gemini`` when ``WAIFU_GEMINI_LLM`` is on (and keyed), else ``openai``
    (any OpenAI-compatible server, local or remote)."""
    return GEMINI if google_config.llm_enabled() else "openai"


def enabled() -> bool:
    """The query LLM is on: ``WAIFU_GEMINI_LLM=1`` or ``WAIFU_QUERY_LLM=1``, online."""
    if os.getenv("WAIFU_ONLINE_SEARCH", "1").lower() not in _YES:
        return False
    return provider() == GEMINI or os.getenv("WAIFU_QUERY_LLM", "0").lower() in _YES


def base_url() -> str:
    """The OpenAI-compatible endpoint, or ``gemini`` for the Gemini backend."""
    if provider() == GEMINI:
        return GEMINI
    return os.getenv("WAIFU_QUERY_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def model() -> str:
    """The model sent to the active backend."""
    if provider() == GEMINI:
        return google_config.llm_model()
    return os.getenv("WAIFU_QUERY_LLM_MODEL", DEFAULT_MODEL)


def _timeout() -> float:
    try:
        return float(os.getenv("WAIFU_QUERY_LLM_TIMEOUT", "20"))
    except ValueError:
        return 20.0


def _is_openai(url: str) -> bool:
    return "api.openai.com" in url


def _request_body(facts: tuple[str, ...], medium_hint: str | None, n: int) -> dict:
    """Chat-completions body: fixed system prompt, player facts as the user turn.

    The Gemini backend reuses its two messages, so both backends see the same
    instructions and the same player-facts-only input.
    """
    lines = [f"- {f}" for f in facts]
    if medium_hint:
        lines.append(f"- medium: {medium_hint}")
    body = {
        "model": model(),
        "temperature": 0,
        # Three short queries fit easily; a low cap bounds CPU decode time.
        "max_tokens": 96,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
            {"role": "user", "content": "Facts:\n" + "\n".join(lines)},
        ],
    }
    if not _is_openai(base_url()) and base_url() != GEMINI:
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


def _call_gemini(facts: tuple[str, ...], medium_hint: str | None, n: int,
                 model_id: str) -> tuple[str, ...] | None:
    """One Gemini call through the shared google-genai client. Raises on errors.

    Same fixed system prompt and player-facts-only input as the OpenAI path.
    No search tool here, so JSON mode is allowed; thinking is kept minimal
    because this is a short rewrite, not a reasoning task. Thinking tokens
    count toward ``max_output_tokens``, so the 256 cap is only safe when the
    budget is explicitly zero — otherwise the JSON array comes back empty.
    """
    from google.genai import types

    body = _request_body(facts, medium_hint, n)
    thinking = google_config.thinking_config(model_id)
    zero = thinking is not None and getattr(thinking, "thinking_budget", None) == 0
    config = types.GenerateContentConfig(
        system_instruction=body["messages"][0]["content"],
        temperature=0,
        max_output_tokens=256 if zero else 2048,
        response_mime_type="application/json",
        thinking_config=thinking,
    )
    t0 = time.perf_counter()
    try:
        response = google_config.client().models.generate_content(
            model=model_id, contents=body["messages"][1]["content"], config=config)
    except Exception as exc:
        _log_call(t0, f"failed: {exc}")
        raise
    finally:
        _STATS["calls"] += 1
        _STATS["last_ms"] = round((time.perf_counter() - t0) * 1000)
    queries = parse_queries(getattr(response, "text", "") or "", n)
    _log_call(t0, f"{len(queries or [])} queries")
    return tuple(queries) if queries else None


def _call(facts: tuple[str, ...], medium_hint: str | None, n: int,
          url: str, model_id: str) -> tuple[str, ...] | None:
    """One blocking round trip to ``url`` (or Gemini). Raises on transport errors."""
    if url == GEMINI:
        return _call_gemini(facts, medium_hint, n, model_id)
    headers = {"Content-Type": "application/json"}
    key = os.getenv("WAIFU_QUERY_LLM_API_KEY") or ""
    # The generic OpenAI key only ever goes to OpenAI over HTTPS -- never to
    # the default local server or any third-party endpoint.
    if not key and _is_openai(url) and url.startswith("https://"):
        key = os.getenv("OPENAI_API_KEY") or ""
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = _request_body(facts, medium_hint, n)
    body["model"] = model_id
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as resp:  # noqa: S310 - configured URL
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _log_call(t0, f"failed: {exc}")
        raise
    finally:
        _STATS["calls"] += 1
        _STATS["last_ms"] = round((time.perf_counter() - t0) * 1000)
    content = payload["choices"][0]["message"]["content"]
    queries = parse_queries(content, n)
    _log_call(t0, f"{len(queries or [])} queries")
    return tuple(queries) if queries else None


def _log_call(t0: float, outcome: str) -> None:
    # Runs on the background worker, outside any request trace, so it gets its
    # own line: this is the time the LLM would cost if a turn waited for it.
    if timing._log_on():
        timing.log.info("query_llm %s %.0fms %s", model(), (time.perf_counter() - t0) * 1000, outcome)


# --- result store --------------------------------------------------------
#
# Finished rewrites are kept by request key. A request that raised is not
# stored, so it may be tried again; one that returned nothing usable is stored
# as None and not retried.

_LOCK = threading.Lock()
_RESULTS: dict[tuple, tuple[str, ...] | None] = {}
_FUTURES: dict[tuple, Future] = {}
_EXECUTOR: ThreadPoolExecutor | None = None
_STATS: dict[str, Any] = {"calls": 0, "last_ms": None}
_MAX_RESULTS = 256


def _key(facts: list[str], medium_hint: str | None, n: int) -> tuple | None:
    clean = tuple(dict.fromkeys(" ".join(f.split())[:200] for f in facts if f and f.strip()))
    if not clean:
        return None
    return (clean, medium_hint, n, base_url(), model())


def _run(key: tuple) -> tuple[str, ...] | None:
    try:
        got = _call(*key)
    except Exception:  # noqa: BLE001 - a dead endpoint must not kill the turn
        with _LOCK:
            _FUTURES.pop(key, None)
        return None
    with _LOCK:
        if len(_RESULTS) >= _MAX_RESULTS:
            _RESULTS.pop(next(iter(_RESULTS)))
        _RESULTS[key] = got
        _FUTURES.pop(key, None)
    return got


def _executor() -> ThreadPoolExecutor:
    # One worker: a CPU-bound local model gains nothing from parallel calls.
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="query-llm")
    return _EXECUTOR


def rewrite(facts: list[str], medium_hint: str | None = None, n: int = 3) -> list[str] | None:
    """Blocking: up to ``n`` search queries for these facts, or ``None``. Never raises.

    The game loop does not use this -- it uses ``prefetch`` / ``peek`` so a
    slow model never holds up a turn.
    """
    if not enabled():
        return None
    key = _key(facts, medium_hint, n)
    if key is None:
        return None
    with _LOCK:
        if key in _RESULTS:
            got = _RESULTS[key]
            return list(got) if got else None
    got = _run(key)
    return list(got) if got else None


def known(facts: list[str], medium_hint: str | None = None, n: int = 3) -> bool:
    """True when this request already finished or is in flight."""
    key = _key(facts, medium_hint, n)
    with _LOCK:
        return key is not None and (key in _RESULTS or key in _FUTURES)


def prefetch(facts: list[str], medium_hint: str | None = None, n: int = 3) -> bool:
    """Start a background rewrite unless it is known already. Never blocks or raises."""
    if not enabled():
        return False
    key = _key(facts, medium_hint, n)
    if key is None:
        return False
    with _LOCK:
        if key in _RESULTS or key in _FUTURES:
            return False
        try:
            _FUTURES[key] = _executor().submit(_run, key)
        except Exception:  # noqa: BLE001 - e.g. interpreter shutting down
            return False
    return True


def peek(facts: list[str], medium_hint: str | None = None, n: int = 3) -> list[str] | None:
    """Finished queries for this request, or ``None`` if pending/failed. Never blocks."""
    if not enabled():
        return None
    key = _key(facts, medium_hint, n)
    with _LOCK:
        got = _RESULTS.get(key) if key is not None else None
    return list(got) if got else None


def wait(facts: list[str], medium_hint: str | None = None, n: int = 3,
         timeout: float = 0.0) -> list[str] | None:
    """Wait up to ``timeout`` seconds for an in-flight request."""
    key = _key(facts, medium_hint, n)
    with _LOCK:
        fut = _FUTURES.get(key) if key is not None else None
    if fut is not None and timeout > 0:
        try:
            fut.result(timeout=timeout)
        except Exception:  # noqa: BLE001 - timeout or failure: carry on without it
            pass
    return peek(facts, medium_hint, n)


def wait_seconds() -> float:
    """``WAIFU_QUERY_LLM_WAIT``: how long a turn may wait for the LLM (default 0)."""
    try:
        return max(0.0, float(os.getenv("WAIFU_QUERY_LLM_WAIT", "0")))
    except ValueError:
        return 0.0


def status() -> dict[str, Any]:
    """For ``/healthz``: backend, model, call count, last latency, in-flight jobs."""
    with _LOCK:
        inflight = len(_FUTURES)
    return {
        "enabled": enabled(),
        "provider": provider(),
        "model": model(),
        "base_url": base_url(),
        "calls": _STATS["calls"],
        "last_ms": _STATS["last_ms"],
        "inflight": inflight,
    }


def clear() -> None:
    """Forget cached results and counters (tests)."""
    with _LOCK:
        _RESULTS.clear()
        _FUTURES.clear()
        _STATS.update(calls=0, last_ms=None)
