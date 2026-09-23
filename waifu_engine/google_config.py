"""Google GenAI (Gemini) settings and the shared client.

One API key and one client serve both Gemini features, each switched on
separately:

* **search** (``WAIFU_GEMINI_SEARCH=1``): ``sources.gemini`` lists candidate
  characters with the Google Search tool.
* **llm** (``WAIFU_GEMINI_LLM=1``): ``query_llm`` rewrites the player's typed
  text into search queries with Gemini instead of an OpenAI-compatible server.

``WAIFU_GEMINI_MODEL`` picks the model for both; ``WAIFU_GEMINI_SEARCH_MODEL``
and ``WAIFU_GEMINI_LLM_MODEL`` override it per feature (e.g. a lite model for
the short query-writing call). Neither feature makes decisions or writes
question text -- Laya and ``traits.py`` keep those.
"""

from __future__ import annotations

import os
import threading
from typing import Any

DEFAULT_MODEL = "gemini-2.5-flash"
_YES = {"1", "true", "yes"}
_LOCK = threading.Lock()
_CLIENT: Any = None


def _on(name: str) -> bool:
    """Whether a ``WAIFU_*`` switch is set to 1/true/yes (default off)."""
    return os.getenv(name, "0").strip().lower() in _YES


def api_key() -> str:
    """``GEMINI_API_KEY``, else ``GOOGLE_API_KEY`` (the order the SDK uses)."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _online() -> bool:
    """``WAIFU_ONLINE_SEARCH`` (default on): no Gemini call is made offline."""
    return os.getenv("WAIFU_ONLINE_SEARCH", "1").strip().lower() in _YES


def search_enabled() -> bool:
    """Gemini search grounding: switched on, online, and a key is set."""
    return _online() and _on("WAIFU_GEMINI_SEARCH") and bool(api_key())


def llm_enabled() -> bool:
    """Gemini as the query LLM: switched on, online, and a key is set."""
    return _online() and _on("WAIFU_GEMINI_LLM") and bool(api_key())


def default_model() -> str:
    """``WAIFU_GEMINI_MODEL``: shared by search and llm unless overridden."""
    return os.getenv("WAIFU_GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def search_model() -> str:
    """Model for search grounding; must support the Google Search tool."""
    return os.getenv("WAIFU_GEMINI_SEARCH_MODEL", "").strip() or default_model()


def llm_model() -> str:
    """Model for query rewriting; any text model works."""
    return os.getenv("WAIFU_GEMINI_LLM_MODEL", "").strip() or default_model()


def timeout_ms() -> int:
    """``WAIFU_GEMINI_TIMEOUT`` seconds (default 15) as the SDK's milliseconds."""
    try:
        return int(float(os.getenv("WAIFU_GEMINI_TIMEOUT", "15")) * 1000)
    except ValueError:
        return 15000


def thinking_config(model_id: str) -> Any | None:
    """Least thinking the model allows: these calls are short lookups, and
    thinking time would dominate them. ``None`` leaves the model default."""
    from google.genai import types

    name = model_id.lower()
    if "2.5-flash" in name:  # flash and flash-lite accept a zero budget
        return types.ThinkingConfig(thinking_budget=0)
    if "gemini-3" in name:
        return types.ThinkingConfig(thinking_level="LOW")
    return None


def client() -> Any:
    """Process-wide ``genai.Client``. Raises ImportError without google-genai."""
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from google import genai
            from google.genai import types

            _CLIENT = genai.Client(
                api_key=api_key(), http_options=types.HttpOptions(timeout=timeout_ms()))
        return _CLIENT


def sdk_installed() -> bool:
    """Whether the optional ``google-genai`` package can be imported."""
    try:
        import google.genai  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def status() -> dict[str, Any]:
    """For ``/healthz``: each Gemini feature's switch and model, key presence."""
    return {
        "sdk": sdk_installed(),
        "key": bool(api_key()),
        "search": {"enabled": search_enabled(), "model": search_model()},
        "llm": {"enabled": llm_enabled(), "model": llm_model()},
    }


def clear() -> None:
    """Drop the cached client (tests, or after changing the key)."""
    global _CLIENT
    with _LOCK:
        _CLIENT = None
