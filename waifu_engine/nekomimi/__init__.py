"""Nekomimi ACG character guessing loop, decided by Laya.

Submodules import cleanly. The turn functions stay lazy so loading
``lexicon`` from ``web_search`` does not pull the engine while it is still
importing that module.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GuessSession",
    "get_session",
    "new_session",
    "start",
    "state_payload",
    "submit_answer",
    "submit_guess_result",
]


def __getattr__(name: str) -> Any:
    """Resolve the public turn API on first use."""
    if name in {"start", "state_payload", "submit_answer", "submit_guess_result"}:
        from .engine import start, state_payload, submit_answer, submit_guess_result

        return {
            "start": start,
            "state_payload": state_payload,
            "submit_answer": submit_answer,
            "submit_guess_result": submit_guess_result,
        }[name]
    if name in {"GuessSession", "get_session", "new_session"}:
        from .session import GuessSession, get_session, new_session

        return {
            "GuessSession": GuessSession,
            "get_session": get_session,
            "new_session": new_session,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
