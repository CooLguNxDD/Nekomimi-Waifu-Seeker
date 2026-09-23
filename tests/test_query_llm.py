"""Offline tests for the optional search-query rewriter. No network."""

from __future__ import annotations

import io
import json

import pytest

from waifu_engine import query_llm, sources


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("WAIFU_QUERY_LLM", "WAIFU_QUERY_LLM_BASE_URL", "WAIFU_QUERY_LLM_MODEL",
                "WAIFU_QUERY_LLM_API_KEY", "OPENAI_API_KEY", "WAIFU_ONLINE_SEARCH"):
        monkeypatch.delenv(var, raising=False)
    query_llm._rewrite_cached.cache_clear()
    yield
    query_llm._rewrite_cached.cache_clear()


def _serve(monkeypatch, content, sent=None):
    def urlopen(req, timeout):
        if sent is not None:
            sent.append((req, json.loads(req.data)))
        if isinstance(content, Exception):
            raise content
        body = {"choices": [{"message": {"content": content}}]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(query_llm.urllib.request, "urlopen", urlopen)


def test_off_by_default_and_never_calls_out(monkeypatch):
    _serve(monkeypatch, AssertionError("must not be called"))
    assert query_llm.rewrite(["silver hair", "mage"]) is None


def test_defaults_to_local_qwen():
    assert query_llm.model() == "Qwen/Qwen3.6-35B-A3B"
    assert query_llm.base_url() == "http://localhost:8000/v1"


def test_parses_capped_queries_and_strips_thinking(monkeypatch):
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    sent = []
    reply = '<think>["ignored"]</think>Sure: ["silver hair mage", "silver hair mage", 7, "' \
            + "x" * 300 + '", "fourth", "fifth"]'
    _serve(monkeypatch, reply, sent)
    got = query_llm.rewrite(["silver hair", "mage"], "game")
    assert got == ["silver hair mage", "x" * query_llm.MAX_QUERY_CHARS, "fourth"]
    req, body = sent[0]
    assert req.full_url == "http://localhost:8000/v1/chat/completions"
    assert body["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "silver hair" in body["messages"][1]["content"]
    # Cached: the same facts do not hit the endpoint twice.
    query_llm.rewrite(["silver hair", "mage"], "game")
    assert len(sent) == 1


def test_openai_gets_no_template_kwargs_and_a_bearer_key(monkeypatch):
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    monkeypatch.setenv("WAIFU_QUERY_LLM_BASE_URL", "https://api.openai.com/v1/")
    monkeypatch.setenv("WAIFU_QUERY_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = []
    _serve(monkeypatch, '["a query"]', sent)
    assert query_llm.rewrite(["tsundere"]) == ["a query"]
    req, body = sent[0]
    assert "chat_template_kwargs" not in body
    assert req.get_header("Authorization") == "Bearer sk-test"


@pytest.mark.parametrize("content", ["no json here", "[1, 2]", '{"q": "x"}', "[broken",
                                     OSError("connection refused"), TimeoutError()])
def test_failures_return_none(monkeypatch, content):
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    _serve(monkeypatch, content)
    assert query_llm.rewrite(["silver hair"]) is None


def test_disabled_when_offline(monkeypatch):
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    monkeypatch.setenv("WAIFU_ONLINE_SEARCH", "0")
    assert query_llm.rewrite(["silver hair"]) is None


def test_rewritten_and_focus_queries_lead_the_templates():
    base = sources._queries(["Nintendo", "male", "Italian plumber"], "game")
    assert sources._queries(["Nintendo", "male", "Italian plumber"], "game",
                            focus=None, rewritten=None) == base
    queries = sources._queries(["Nintendo", "male", "Italian plumber"], "game",
                               focus=["Italian plumber", "Nintendo"],
                               rewritten=["mario nintendo plumber"])
    assert queries[0] == "mario nintendo plumber"
    assert queries[1] == "Italian plumber Nintendo video game character"
    assert len(queries) <= sources.MAX_QUERIES
