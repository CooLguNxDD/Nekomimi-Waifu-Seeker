"""Local-LLM name hypotheses: parsing, resolution, gating. Offline, no network."""

from __future__ import annotations

import io
import hashlib
import json
import time

import pytest

from waifu_engine import query_llm, sources
from waifu_engine.nekomimi import engine, session as sess_mod
from waifu_engine.nekomimi.session import Candidate
from waifu_engine.sources import anilist, cache, llm_names, wikipedia


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


def test_entity_cache_separates_unicode_names_and_rejects_wrong_series(monkeypatch):
    assert cache._make_slug("東京") != cache._make_slug("大阪")
    cache.save_entity(_hit("Aqua", "Kingdom Hearts"))
    assert cache.get_entity("Aqua", "KonoSuba") is None
    assert not llm_names._matches(_hit("Aqua", "Kingdom Hearts"), "Aqua", "KonoSuba")

    monkeypatch.setattr(anilist, "search_characters", lambda q, limit=10: [
        _hit("Aqua", "KonoSuba"),
    ])
    monkeypatch.setattr(wikipedia, "search_characters", lambda q, limit=8: [])
    resolved = llm_names._lookup("Aqua", "KonoSuba", None)
    assert resolved is not None and resolved["series"] == "KonoSuba"


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
    """Carry proposal revision metadata with each background result."""
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
    hits, metadata = sources.take_hypothesis_candidates("sess-llm")
    assert [c["name"] for c in hits] == ["Makima"]
    assert metadata == {
        "evidence_revision": 0,
        "fingerprint": hashlib.sha256(b"female\ndemon").hexdigest(),
    }
    assert not sources._llm_names_start("sess-llm", ["female", "demon"], "anime")
    assert runs == [("female", "demon")]


def test_hypotheses_skip_rejected_names_and_wait_for_a_flat_pool(monkeypatch):
    """Avoid repeating unchanged proposals and never send a rejected identity."""
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    sess = sess_mod.new_session()
    sess.seed = "female character"
    sess.asked = [{
        "turn": 1, "qid": "gender_female", "text": "Is your character female?",
        "answer": "yes", "detail": None,
    }]
    sess.exclusion_log.append({
        "candidate_id": "arue", "name": "Arue", "series": "KonoSuba",
        "reason": "wrong_guess_identity", "revision": 1, "turn": 1,
    })
    # Empty pool: ask, and never send the rejected (scraped) name.
    facts = engine._hypothesis_facts(sess, lambda: False)
    assert facts == ["female character", "Yes: Is your character female"]
    assert "Arue" not in "\n".join(facts)
    sess.proposal_fingerprint = hashlib.sha256("\n".join(facts).encode()).hexdigest()
    sess.proposal_revision = 1
    sess.proposal_turn = 1

    sess.turn = 4
    sess.candidates = [Candidate(id="a", name="A", logodds=5.0),
                       Candidate(id="b", name="B", logodds=0.0)]
    # A clear leader and unchanged facts: no duplicate, even when stuck.
    assert engine._hypothesis_facts(sess, lambda: False) == []
    assert engine._hypothesis_facts(sess, lambda: True) == []


def test_name_proposal_cooldown_new_fact_and_per_session_cap(monkeypatch):
    """Respect cooldown and request limits while letting a rejection retrigger search."""
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    sess = sess_mod.new_session("pink hair")
    first = engine.proposal_facts(sess)
    sess.proposal_fingerprint = hashlib.sha256("\n".join(first).encode()).hexdigest()
    sess.proposal_revision = sess.evidence_revision
    sess.proposal_turn = 2
    sess.proposal_calls = 1
    sess.asked = [{
        "turn": 3, "qid": "gender_female", "text": "Is your character female?",
        "answer": "yes", "detail": "wears a black ribbon",
    }]
    sess.turn = 3
    assert engine._hypothesis_facts(sess, lambda: False) == []  # cooldown
    sess.turn = 4
    updated = engine._hypothesis_facts(sess, lambda: False)
    assert "wears a black ribbon" in updated

    sess.proposal_calls = engine._LLM_NAMES_MAX_PER_SESSION
    assert engine._hypothesis_facts(sess, lambda: True) == []

    sess.proposal_calls = 1
    sess.proposal_fingerprint = hashlib.sha256("\n".join(updated).encode()).hexdigest()
    sess.proposal_turn = sess.turn
    sess.proposal_revision = sess.evidence_revision
    sess.exclusion_log.append({
        "candidate_id": "rejected", "name": "Wrong Pick", "reason": "wrong_guess_identity",
        "revision": sess.evidence_revision + 1,
    })
    assert engine._hypothesis_facts(sess, lambda: False) == updated


def test_name_proposals_use_button_memory_and_retry_only_changed_facts(monkeypatch):
    """Allow normalized button evidence to seed one nonblocking proposal job."""
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    runs = []

    def fake_search(facts, medium_hint=None, errors=None, stats=None):
        runs.append(tuple(facts))
        return []

    monkeypatch.setattr(llm_names, "search", fake_search)
    monkeypatch.setattr(engine, "ONLINE", True)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    sess = sess_mod.new_session()
    sess.asked = [{
        "turn": 1, "qid": "gender_female", "text": "Is your character female?",
        "answer": "yes", "detail": None,
    }]
    sess.turn = 1

    engine.refresh_candidates(sess)  # empty pool is broad; no Laya wait is needed
    for _ in range(200):
        if not sources.background_status(sess.id)["llm_names"]:
            break
        time.sleep(0.01)
    assert len(runs) == 1
    assert any("female" in fact.lower() for fact in runs[0])

    engine.refresh_candidates(sess)
    assert len(runs) == 1  # unchanged normalized facts do not start another job


def test_seed_proposal_records_the_revision_it_used(monkeypatch):
    """Tag initial seed leads with the revision already applied to the session."""
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    monkeypatch.setattr(query_llm, "prefetch", lambda *a, **k: False)
    monkeypatch.setattr(engine, "ONLINE", True)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    monkeypatch.setattr(engine.laya_client, "ask", lambda *a, **k: None)
    monkeypatch.setattr(engine.laya_client, "available", lambda: False)
    monkeypatch.setattr(llm_names, "search", lambda *a, **k: [])

    state = engine.start("pink hair")
    sess = sess_mod.load_session(state["session_id"])
    assert sess is not None
    for _ in range(200):
        if not sources.background_status(sess.id)["llm_names"]:
            break
        time.sleep(0.01)
    _hits, metadata = sources.take_hypothesis_candidates(sess.id)
    assert metadata["evidence_revision"] == sess.evidence_revision == 1


def test_stale_proposal_is_rescored_and_rejected_identity_stays_excluded(monkeypatch):
    """Rescore old leads under current evidence and keep a rejected identity excluded."""
    monkeypatch.setenv("WAIFU_LLM_NAMES", "0")
    monkeypatch.setattr(engine, "ONLINE", True)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    sess = sess_mod.new_session()
    sess.candidates = [Candidate(id="old-makima", name="Makima", series="Chainsaw Man")]
    engine._reject_identity(sess, sess.candidates[0])
    sess.evidence_revision = 4
    question = dict(engine.QUESTIONS_BY_ID["gender_female"])
    sess.evidence["gender_female"] = (question, "yes")
    sess.asked = [{
        "turn": 1, "qid": "gender_female", "text": "Is your character female?",
        "answer": "yes", "detail": None,
    }]
    calls = []

    def ask(state, questions):
        calls.append((state, questions))
        return {"match": {"noul": 0.97}}

    monkeypatch.setattr(engine.laya_client, "ask", ask)
    stale_hits = [
        _hit("New Hero", "New Work"),
        _hit("Makima", "Chainsaw Man").copy(),
    ]
    stale_hits[1]["id"] = "new-makima-url"
    sources._LLM_BG.finish(
        sess.id, stale_hits, time.perf_counter(), [],
        metadata={"evidence_revision": 1, "fingerprint": "old-facts"},
    )
    events = []
    monkeypatch.setattr(engine.timing, "note", lambda **fields: events.append(fields))

    assert engine.refresh_candidates(sess) == 1
    assert calls and calls[0][1]["match"]["type"] == "noul"
    assert calls[0][1]["match"]["instructions"] == question["instructions"]
    assert sess.by_id("x_New Hero") is not None
    assert ("x_New Hero", "gender_female") in sess.match_cache
    assert ("x_New Hero", "gender_female") in sess.match_fingerprints
    assert all(c.name != "Makima" for c in sess.alive_candidates())
    assert any(row.get("proposal_stale") is True for row in events)
    trace = engine.trace_payload(sess, "Makima")
    assert trace["target"]["alive"] is False
    assert any(row["reason"] == "wrong_guess_identity" for row in trace["target"]["exclusions"])
