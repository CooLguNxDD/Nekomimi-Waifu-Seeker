"""Process-wide Laya agent.

Laya is a non-autoregressive *decision* model: it never generates text, it only
answers typed questions (``choice`` / ``score`` / ``noul``) in one forward pass.
Loading the 421M checkpoint costs ~7-10s (~26 s on the CPU-only dev box), so it
is loaded once per process. ``web.py`` calls ``preload()`` at startup so the
first player never pays for it; ``get_agent()`` still loads lazily for the CLI
and tests. ``python -m waifu_engine.nekomimi.laya_client`` preloads and exits,
which is how the Docker image bakes the weights at build time.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any

from .. import timing

_AGENT: Any = None
_LOAD_FAILED = False
_LOCK = threading.Lock()
_STATUS: dict[str, Any] = {}

# Tiny question for the warm-up pass: the first ``predict`` pays for lazy
# initialisation that should not land on the first player.
_WARMUP_STATE = {"candidate": "Mario (Super Mario) [game]. An Italian plumber."}
_WARMUP_QUESTIONS = {
    "warmup": {
        "type": "noul",
        "instructions": "Is the character in `candidate` from a video game?",
    }
}

MODEL_ID = os.getenv("WAIFU_LAYA_MODEL", "convaiinnovations/laya")
# The `typed-decisions` checkpoint is the same 421M model trained for exactly
# this shape of work. Its temperatures are ~1.0, where the default `english`
# checkpoint uses 1.98 for `noul` -- that flattening was crushing every
# probability toward 0.5 and making confidences useless. Set to "" for the
# english checkpoint.
SUBFOLDER = os.getenv("WAIFU_LAYA_SUBFOLDER", "typed-decisions").strip()
# Option strings for a `choice` question are packed into the decision head; the
# default budget (192) is too small for candidate profiles.
HEAD_MAX_LEN = int(os.getenv("WAIFU_LAYA_HEAD_MAX_LEN", "480"))
MAX_LEN = int(os.getenv("WAIFU_LAYA_MAX_LEN", "512"))
_YES = {"1", "true", "yes"}


def _provider() -> str:
    # auto: ONNX+DirectML if a graph exists (or can be exported) and DML is present;
    # dml: same, still falls back to torch if the graph cannot be loaded;
    # cpu: ONNX CPU EP if a graph exists, else torch; torch: skip ONNX entirely.
    return os.getenv("WAIFU_LAYA_PROVIDER", "auto").strip().lower()


def _forced_off() -> bool:
    return os.getenv("WAIFU_FORCE_FALLBACK", "").lower() in _YES


def _configure(agent: Any) -> Any:
    agent.cfg["head_max_len"] = HEAD_MAX_LEN
    agent.cfg["max_len"] = MAX_LEN
    return agent


def _try_onnx() -> Any | None:
    """Load an ONNX agent when the provider setting wants it. Never raises."""
    provider = _provider()
    if provider == "torch":
        return None
    try:
        from . import laya_onnx
    except Exception as e:  # noqa: BLE001
        print("[waifu] onnx backend unavailable: %s" % e, flush=True)
        return None
    prefer_dml = provider in {"auto", "dml"}
    path = laya_onnx.onnx_path()
    should_export = (
        not path.is_file()
        and os.getenv("WAIFU_LAYA_EXPORT", "1").lower() in _YES
        and prefer_dml
        and laya_onnx.dml_available()
    )
    if should_export:
        try:
            print("[waifu] no ONNX graph yet; exporting Laya for DirectML...", flush=True)
            laya_onnx.export_onnx(path)
        except Exception as e:  # noqa: BLE001
            print("[waifu] ONNX export failed (%s); using torch CPU" % e, flush=True)
            return None
    if not path.is_file():
        return None
    agent = laya_onnx.load_onnx_agent(path, prefer_dml=prefer_dml)
    if agent is None:
        return None
    return _configure(agent)


def _try_torch() -> Any | None:
    import laya

    agent = laya.load(MODEL_ID, subfolder=SUBFOLDER or None)
    agent.backend = "torch"
    return _configure(agent)


def get_agent() -> Any | None:
    """Return the shared Laya agent, or None if it cannot be loaded."""
    global _AGENT, _LOAD_FAILED
    if _forced_off():
        return None
    if _AGENT is not None:
        return _AGENT
    if _LOAD_FAILED:
        return None
    with _LOCK:
        if _AGENT is not None:
            return _AGENT
        if _LOAD_FAILED:
            return None
        try:
            agent = _try_onnx()
            if agent is None:
                agent = _try_torch()
            if agent is None:
                _LOAD_FAILED = True
                return None
            print("[waifu] Laya backend: %s" % getattr(agent, "backend", "torch"), flush=True)
            _AGENT = agent
        except Exception:  # noqa: BLE001 - any import/download/runtime failure degrades
            _LOAD_FAILED = True
            return None
    return _AGENT


def available() -> bool:
    return get_agent() is not None


def backend() -> str:
    """Currently loaded backend (``dml`` / ``onnx-cpu`` / ``torch`` / ``none``)."""
    if _AGENT is None:
        return "none"
    return str(getattr(_AGENT, "backend", "torch"))


def ask(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Run one batched forward pass. Returns ``result["answers"]`` or None."""
    if not questions:
        return {}
    agent = get_agent()
    if agent is None:
        return None
    # One span per question kind: 40 per-candidate ``match`` calls add up into
    # a single ``laya.match`` entry with its call count.
    try:
        # Agent.predict is not documented as thread-safe. Waiting for the lock
        # is timed apart: it is other players' turns, not this one's model cost.
        with timing.span("laya.lock_wait"):
            _LOCK.acquire()
        try:
            with timing.span("laya." + next(iter(questions))):
                result = agent.predict(state, questions)
        finally:
            _LOCK.release()
    except Exception:  # noqa: BLE001
        return None
    answers = result.get("answers")
    return answers if isinstance(answers, dict) else None


def preload() -> dict[str, Any]:
    """Load the agent and run one warm-up pass. Never raises.

    Returns (and remembers, for ``status()``) what happened. A failed load
    latches ``_LOAD_FAILED`` as before, so requests do not retry it one by one.
    """
    info: dict[str, Any] = {
        "preloaded": True,
        "loaded": False,
        "backend": "none",
        "load_seconds": None,
        "warmup_seconds": None,
        "error": None,
    }
    if _forced_off():
        info["error"] = "WAIFU_FORCE_FALLBACK is set"
        _STATUS.clear()
        _STATUS.update(info)
        return dict(info)
    t0 = time.perf_counter()
    try:
        agent = get_agent()
    except Exception as exc:  # noqa: BLE001 - get_agent should not raise, but be sure
        agent, info["error"] = None, str(exc)
    info["load_seconds"] = round(time.perf_counter() - t0, 2)
    if agent is None:
        info["error"] = info["error"] or "Laya could not be loaded; using heuristics"
        print("[waifu] Laya unavailable at startup: %s" % info["error"], flush=True)
    else:
        info["loaded"] = True
        info["backend"] = backend()
        t1 = time.perf_counter()
        try:
            with _LOCK:
                agent.predict(_WARMUP_STATE, _WARMUP_QUESTIONS)
        except Exception as exc:  # noqa: BLE001 - a warm-up failure is not a load failure
            info["error"] = "warm-up failed: %s" % exc
        info["warmup_seconds"] = round(time.perf_counter() - t1, 2)
        print(
            "[waifu] Laya ready: backend=%s load=%.1fs warmup=%.1fs"
            % (info["backend"], info["load_seconds"], info["warmup_seconds"]),
            flush=True,
        )
    _STATUS.clear()
    _STATUS.update(info)
    return dict(info)


def status() -> dict[str, Any]:
    """What ``preload()`` found. Read-only: never triggers a load."""
    if not _STATUS:
        return {"preloaded": False, "loaded": _AGENT is not None, "backend": backend()}
    return dict(_STATUS, loaded=_AGENT is not None, backend=backend())


def reset() -> None:
    """Drop the cached agent (tests)."""
    global _AGENT, _LOAD_FAILED
    with _LOCK:
        _AGENT = None
        _LOAD_FAILED = False
        _STATUS.clear()


def main() -> int:
    """Preload and report; exit 1 if Laya did not load (Docker build step)."""
    info = preload()
    print(info, flush=True)
    return 0 if info["loaded"] else 1


if __name__ == "__main__":
    sys.exit(main())
