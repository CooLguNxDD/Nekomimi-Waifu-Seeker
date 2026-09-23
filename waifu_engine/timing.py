"""Per-request timing, to find where a turn spends its time.

A request (``start``, ``answer``, ``guess``) opens a trace; code inside it
wraps work in ``span("name")``. Spans with the same name add up and count
calls, so 40 Laya ``match`` calls show as one ``laya.match`` entry. Dotted
names are nested inside their prefix (``fetch.wikipedia`` is part of
``search``), so parents include their children's time.

At the end of the request one line is logged to the ``waifu.timing`` logger::

    [waifu] answer turn=4 total=3412ms cands=38/42 | search 2980ms, fetch.playwright 2100ms, laya.match 1450ms/12x ...

and the same numbers go back to the client as ``payload["timing"]``.
``WAIFU_TIMING_LOG=0`` silences the log line; the payload is always filled.
Outside a trace ``span`` is a no-op, so tests and the CLI pay nothing.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

_TLS = threading.local()

log = logging.getLogger("waifu.timing")
if not log.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[waifu] %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _log_on() -> bool:
    return os.getenv("WAIFU_TIMING_LOG", "1").strip().lower() in {"1", "true", "yes"}


class Trace:
    def __init__(self, label: str) -> None:
        self.label = label
        self.t0 = time.perf_counter()
        self.spans: dict[str, list[float]] = {}  # name -> [total ms, calls]
        self.fields: dict[str, Any] = {}

    def add(self, name: str, ms: float) -> None:
        entry = self.spans.setdefault(name, [0.0, 0])
        entry[0] += ms
        entry[1] += 1

    def summary(self) -> dict[str, Any]:
        total = (time.perf_counter() - self.t0) * 1000
        spans = {
            name: {"ms": round(ms, 1), "calls": calls}
            for name, (ms, calls) in sorted(self.spans.items(), key=lambda kv: -kv[1][0])
        }
        return {"step": self.label, "total_ms": round(total, 1), "spans": spans, **self.fields}

    def line(self, summary: dict[str, Any]) -> str:
        head = [summary["step"]]
        head += [f"{k}={v}" for k, v in self.fields.items()]
        head.append(f"total={summary['total_ms']:.0f}ms")
        parts = []
        for name, s in summary["spans"].items():
            if s["ms"] < 1.0:
                continue  # noise in the log line; still in the payload
            calls = f"/{s['calls']}x" if s["calls"] > 1 else ""
            parts.append(f"{name} {s['ms']:.0f}ms{calls}")
        return " ".join(head) + (" | " + ", ".join(parts) if parts else "")


def current() -> Trace | None:
    return getattr(_TLS, "trace", None)


@contextmanager
def span(name: str) -> Iterator[None]:
    """Time a block inside the current trace; a no-op without one."""
    trace = current()
    if trace is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        trace.add(name, (time.perf_counter() - t0) * 1000)


def note(**fields: Any) -> None:
    """Attach context (turn, candidate counts, ...) to the current trace."""
    trace = current()
    if trace is not None:
        trace.fields.update(fields)


def traced(label: str) -> Callable:
    """Trace a request handler; its dict result gains a ``timing`` entry."""

    def wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if current() is not None:  # already inside a trace
                return fn(*args, **kwargs)
            trace = Trace(label)
            _TLS.trace = trace
            try:
                result = fn(*args, **kwargs)
            finally:
                _TLS.trace = None
            summary = trace.summary()
            if isinstance(result, dict):
                result["timing"] = summary
            if _log_on():
                log.info(trace.line(summary))
            return result

        return inner

    return wrap
