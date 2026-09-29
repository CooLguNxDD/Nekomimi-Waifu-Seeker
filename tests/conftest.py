"""Suite-wide setup: never read a developer's ``.env`` (it may hold real keys
and switch on billed services); every test sets the variables it needs."""

import os
import importlib

import pytest

os.environ["WAIFU_ENV_FILE"] = "0"
# Guess-time portrait lookup opens AniList and Wikipedia. The suite stays
# offline; tests that exercise the lookup turn this back on and stub HTTP.
os.environ["WAIFU_PORTRAITS"] = "0"


@pytest.fixture(autouse=True)
def isolate_persisted_test_data(tmp_path, monkeypatch):
    """Keep session and entity-cache writes inside each test's temporary directory."""
    from waifu_engine.nekomimi import session

    with session._STORE_LOCK:
        previous_store = dict(session._STORE)
        session._STORE.clear()
    monkeypatch.setattr(session, "SESSION_DIR", str(tmp_path / "sessions"))
    cache = importlib.import_module("waifu_engine.sources.cache")
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "entity-cache"))
    try:
        yield
    finally:
        with session._STORE_LOCK:
            session._STORE.clear()
            session._STORE.update(previous_store)
