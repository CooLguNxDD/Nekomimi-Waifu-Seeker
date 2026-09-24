"""Minimal ``.env`` loader, so settings (API keys, feature switches) can live
in a file instead of the shell. No dependency; ``KEY=value`` lines only.

Real environment variables always win: the file only fills in what is unset.
``WAIFU_ENV_FILE`` names the file (default ``.env`` in the working directory);
``WAIFU_ENV_FILE=0`` turns loading off (the test suite does this, so a local
``.env`` with a real key never reaches the offline tests).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse(text: str) -> dict[str, str]:
    """``KEY=value`` pairs from ``.env`` text.

    Blank lines and ``#`` comments are skipped, an ``export`` prefix is
    allowed, and a value in matching single or double quotes is taken
    verbatim, including when a `` #`` comment follows the closing quote.
    An unquoted value ends at `` #``. Matching the first and last character
    left the quotes on ``KEY="abc"  # note``, and the key was sent quoted.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        quoted = re.match(r"""^(['"])(.*?)\1(?:\s+#.*)?$""", value)
        if quoted:
            value = quoted.group(2)
        else:
            value = value.split(" #", 1)[0].strip()
        out[key] = value
    return out


def load(path: str | os.PathLike[str] | None = None) -> list[str]:
    """Set unset variables from the env file; returns the names it set. Never raises."""
    name = os.getenv("WAIFU_ENV_FILE", ".env") if path is None else str(path)
    if name.strip().lower() in {"0", "false", "no", "off", ""}:
        return []
    try:
        text = Path(name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    added = []
    for key, value in parse(text).items():
        if key not in os.environ:
            os.environ[key] = value
            added.append(key)
    return added
