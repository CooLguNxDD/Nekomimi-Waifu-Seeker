"""Local-LLM name hypotheses: parsing, resolution, gating. Offline, no network."""

from __future__ import annotations

import io
import json
import time

import pytest

from waifu_engine import query_llm, sources
from waifu_engine.nekomimi import engine, session as sess_mod
from waifu_engine.nekomimi.session import Candidate
from waifu_engine.sources import anilist, llm_names, wikipedia


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("WAIFU_QUERY_LLM", "WAIFU_QUERY_LLM_BASE_URL", "WAIFU_QUERY_LLM_MODEL",
                "WAIFU_QUERY_LLM_API_KEY", "OPENAI_API_KEY", "WAIFU_ONLINE_SEARCH",
                "WAIFU_LLM_NAMES", "WAIFU_QUERY_LLM_THINKING", "WAIFU_QUERY_LLM_MAX_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    query_llm.clear()
    sources._LLM_DONE.clear()
    yield
    query_llm.clear()
    sources._LLM_DONE.clear()


def _serve(monkeypatch, content, sent=None):
    def urlopen(req, timeout):
        if sent is not None:
            sent.append(json.loads(req.data))
        body = {"choices": [{"message": {"content": content}}]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(query_llm.urllib.request, "urlopen", urlopen)


def test_parse_characters_keeps_clean_rows_only():
    reply = ('<think>[{"name": "ignored"}]</think>Here: [{"name": " Makima ", "series": '
             '"Chainsaw Man"}, {"name": "makima"}, {"name": 7}, "Power", {"series": "x"}, '
             '{"name": "' + "y" * 200 + '", "series": ["bad"]}]')
    rows = query_llm.parse_characters(reply, n=5)
    assert rows == [
        {"name": "Makima", "series": "Chainsaw Man"},
        {"name": "Power", "series": ""},
        {"name": "y" * query_llm.MAX_NAME_CHARS, "series": ""},
    ]
    assert query_llm.parse_characters("no json") is None
    assert query_llm.parse_characters("[1, 2]") is None


def test_names_are_off_with_the_query_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(query_llm.urllib.request, "urlopen", boom)
    assert query_llm.names_enabled() is False
    assert query_llm.propose_characters(["female", "demon"]) is None
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    monkeypatch.setenv("WAIFU_LLM_NAMES", "0")
    assert query_llm.names_enabled() is False


def test_propose_sends_player_facts_only_and_caches(monkeypatch):
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    sent = []
    _serve(monkeypatch, '[{"name": "Makima", "series": "Chainsaw Man"}]', sent)
    facts = ["The character is female", "Not true: Is your character human"]
    got = query_llm.propose_characters(facts, "anime")
    assert got == [{"name": "Makima", "series": "Chainsaw Man"}]
    body = sent[0]
    assert body["messages"][0]["content"] == query_llm.NAMES_PROMPT.format(n=12)
    user = body["messages"][1]["content"]
    assert user == "Facts:\n- " + "\n- ".join(facts) + "\n- medium: anime"
    assert body["max_tokens"] >= query_llm.NAMES_MAX_TOKENS
    assert body["reasoning_effort"] == "none"
    query_llm.propose_characters(facts, "anime")
    assert len(sent) == 1
    # A names request and a rewrite of the same facts are separate entries.
    _serve(monkeypatch, '["demon woman red hair"]', sent)
    assert query_llm.rewrite(facts, "anime") == ["demon woman red hair"]


def _hit(name, series, source="anilist"):
    return {"id": f"x_{name}", "name": name, "series": series, "medium": "anime",
            "blurb": f"{name} from {series}.", "tags": [], "source": source}


def test_resolve_keeps_real_pages_and_drops_inventions(monkeypatch):
    pages = {
        "Makima": [_hit("Makima", "Chainsaw Man")],
        "2B": [_hit("YoRHa No.2 Type B", "NieR:Automata")],
        "Invented Girl": [_hit("Someone Else", "Other Show")],
    }
    monkeypatch.setattr(anilist, "search_characters", lambda q, limit=10: pages.get(q, []))
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit=8: [])
    rows = [
        {"name": "Makima", "series": "Chainsaw Man"},
        {"name": "2B", "series": "NieR: Automata"},
        {"name": "Invented Girl", "series": "Nowhere"},
        {"name": "Makima", "series": ""},
    ]
    out = llm_names.resolve(rows, "anime")
    assert [c["name"] for c in out] == ["Makima", "YoRHa No.2 Type B"]
    assert all(c["via"] == "llm_names" and c["source"] == "anilist" for c in out)


def test_resolve_survives_a_failing_lookup(monkeypatch):
    def flaky(q, limit=10):
        if q == "Bad":
            raise OSError("down")
        return [_hit(q, "Show")]

    monkeypatch.setattr(anilist, "search_characters", flaky)
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit=8: [])
    errors: list[str] = []
    out = llm_names.resolve([{"name": "Bad"}, {"name": "Good"}], errors=errors)
    assert [c["name"] for c in out] == ["Good"]
    assert errors and "down" in errors[0]


def test_background_hits_join_the_next_search_once_per_facts(monkeypatch):
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    runs = []

    def fake_search(facts, medium_hint=None, errors=None, stats=None):
        runs.append(tuple(facts))
        return [_hit("Makima", "Chainsaw Man")]

    monkeypatch.setattr(llm_names, "search", fake_search)
    assert sources._llm_names_start("sess-llm", ["female", "demon"], "anime")
    for _ in range(200):
        if not sources._LLM_BG.is_pending("sess-llm"):
            break
        time.sleep(0.01)
    assert [c["name"] for c in sources._LLM_BG.take("sess-llm")] == ["Makima"]
    assert not sources._llm_names_start("sess-llm", ["female", "demon"], "anime")
    assert runs == [("female", "demon")]


def test_hypotheses_skip_rejected_names_and_wait_for_a_flat_pool(monkeypatch):
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    sess = sess_mod.new_session()
    sess.constraints = ["The character is female", "The character is not Arue"]
    # Empty pool: ask, and never send the rejected (scraped) name.
    assert engine._hypothesis_facts(sess, lambda: False) == ["The character is female"]

    sess.turn = 4
    sess.candidates = [Candidate(id="a", name="A", logodds=5.0),
                       Candidate(id="b", name="B", logodds=0.0)]
    # A clear leader and search not stuck: no call.
    assert engine._hypothesis_facts(sess, lambda: False) == []
    assert engine._hypothesis_facts(sess, lambda: True) == ["The character is female"]
    sess.candidates = [Candidate(id=f"c{i}", name=f"C{i}", logodds=0.0) for i in range(6)]
    assert engine._hypothesis_facts(sess, lambda: False) == ["The character is female"]
