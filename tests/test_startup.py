"""Laya preloads at app startup. Offline: the agent is a stub, no weights."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from waifu_engine import web
from waifu_engine.nekomimi import laya_client


class FakeAgent:
    backend = "torch"

    def __init__(self, fail_warmup: bool = False):
        self.calls = []
        self.fail_warmup = fail_warmup

    def predict(self, state, questions):
        self.calls.append(set(questions))
        if self.fail_warmup:
            raise RuntimeError("boom")
        return {"answers": {"warmup": {"noul": 0.9}}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("WAIFU_FORCE_FALLBACK", "WAIFU_LAYA_PRELOAD", "WAIFU_LAYA_REQUIRED"):
        monkeypatch.delenv(var, raising=False)
    laya_client.reset()
    yield
    laya_client.reset()


def _install(monkeypatch, agent):
    loads = []

    def get_agent():
        loads.append(1)
        laya_client._AGENT = agent
        return agent

    monkeypatch.setattr(laya_client, "get_agent", get_agent)
    return loads


def test_preload_loads_once_and_warms_up(monkeypatch):
    agent = FakeAgent()
    loads = _install(monkeypatch, agent)
    info = laya_client.preload()
    assert loads == [1]
    assert agent.calls == [{"warmup"}]
    assert info["loaded"] and info["backend"] == "torch" and info["error"] is None
    assert info["load_seconds"] is not None and info["warmup_seconds"] is not None


def test_preload_without_model_degrades(monkeypatch):
    monkeypatch.setattr(laya_client, "get_agent", lambda: None)
    info = laya_client.preload()
    assert info["loaded"] is False
    assert info["error"]
    assert laya_client.main() == 1


def test_warmup_failure_keeps_the_loaded_model(monkeypatch):
    _install(monkeypatch, FakeAgent(fail_warmup=True))
    info = laya_client.preload()
    assert info["loaded"] is True
    assert "warm-up failed" in info["error"]


def test_status_never_loads(monkeypatch):
    monkeypatch.setattr(laya_client, "get_agent",
                        lambda: pytest.fail("status() must not load the model"))
    assert laya_client.status() == {"preloaded": False, "loaded": False, "backend": "none"}


def test_app_startup_preloads_before_first_request(monkeypatch):
    agent = FakeAgent()
    loads = _install(monkeypatch, agent)
    with TestClient(web.app) as client:
        assert loads == [1]  # before any request was made
        health = client.get("/healthz").json()
        assert health["status"] == "ok"
        assert health["laya"]["loaded"] is True
        assert health["laya"]["preloaded"] is True
    assert loads == [1]


def test_preload_can_be_disabled(monkeypatch):
    monkeypatch.setenv("WAIFU_LAYA_PRELOAD", "0")
    loads = _install(monkeypatch, FakeAgent())
    with TestClient(web.app) as client:
        assert client.get("/healthz").json()["laya"]["preloaded"] is False
    assert loads == []


def test_force_fallback_skips_preload(monkeypatch):
    monkeypatch.setenv("WAIFU_FORCE_FALLBACK", "1")
    loads = _install(monkeypatch, FakeAgent())
    with TestClient(web.app):
        pass
    assert loads == []


def test_required_laya_fails_startup(monkeypatch):
    monkeypatch.setenv("WAIFU_LAYA_REQUIRED", "1")
    monkeypatch.setattr(laya_client, "get_agent", lambda: None)
    with pytest.raises(RuntimeError, match="WAIFU_LAYA_REQUIRED"):
        with TestClient(web.app):
            pass


def test_required_laya_loads_even_with_preload_off(monkeypatch):
    monkeypatch.setenv("WAIFU_LAYA_PRELOAD", "0")
    monkeypatch.setenv("WAIFU_LAYA_REQUIRED", "1")
    loads = _install(monkeypatch, FakeAgent())
    with TestClient(web.app):
        pass
    assert loads == [1]
    laya_client.reset()
    monkeypatch.setattr(laya_client, "get_agent", lambda: None)
    with pytest.raises(RuntimeError, match="WAIFU_LAYA_REQUIRED"):
        with TestClient(web.app):
            pass
