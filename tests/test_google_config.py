"""Google GenAI settings: switches, model selection, thinking, .env loading."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from waifu_engine import envfile, google_config, query_llm

GOOGLE_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "WAIFU_GEMINI_SEARCH", "WAIFU_GEMINI_LLM",
               "WAIFU_GEMINI_MODEL", "WAIFU_GEMINI_SEARCH_MODEL", "WAIFU_GEMINI_LLM_MODEL",
               "WAIFU_QUERY_LLM", "WAIFU_ONLINE_SEARCH")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in GOOGLE_VARS:
        monkeypatch.delenv(var, raising=False)
    google_config.clear()
    query_llm.clear()
    yield
    google_config.clear()
    query_llm.clear()


# --- switches and model selection -------------------------------------------


def test_each_feature_needs_its_switch_a_key_and_network(monkeypatch):
    monkeypatch.setenv("WAIFU_GEMINI_SEARCH", "1")
    monkeypatch.setenv("WAIFU_GEMINI_LLM", "1")
    assert not google_config.search_enabled() and not google_config.llm_enabled()  # no key
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert google_config.search_enabled() and google_config.llm_enabled()
    monkeypatch.setenv("WAIFU_GEMINI_LLM", "0")
    assert google_config.search_enabled() and not google_config.llm_enabled()
    monkeypatch.setenv("WAIFU_ONLINE_SEARCH", "0")
    assert not google_config.search_enabled()


def test_gemini_key_wins_over_google_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini")
    assert google_config.api_key() == "gemini"


def test_one_model_for_both_with_per_feature_overrides(monkeypatch):
    assert google_config.search_model() == google_config.llm_model() == "gemini-2.5-flash"
    monkeypatch.setenv("WAIFU_GEMINI_MODEL", "gemini-3-flash")
    assert google_config.search_model() == google_config.llm_model() == "gemini-3-flash"
    monkeypatch.setenv("WAIFU_GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
    assert google_config.llm_model() == "gemini-2.5-flash-lite"
    assert google_config.search_model() == "gemini-3-flash"


def test_thinking_is_kept_to_the_minimum_each_model_allows():
    assert google_config.thinking_config("gemini-2.5-flash").thinking_budget == 0
    assert google_config.thinking_config("gemini-2.5-flash-lite").thinking_budget == 0
    assert str(google_config.thinking_config("gemini-3-pro").thinking_level).endswith("LOW")
    assert google_config.thinking_config("gemini-2.5-pro") is None  # cannot turn it off


def test_status_reports_both_features(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("WAIFU_GEMINI_LLM", "1")
    monkeypatch.setenv("WAIFU_GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
    st = google_config.status()
    assert st["key"] is True
    assert st["search"] == {"enabled": False, "model": "gemini-2.5-flash"}
    assert st["llm"] == {"enabled": True, "model": "gemini-2.5-flash-lite"}


# --- Gemini as the query LLM -------------------------------------------------


class FakeModels:
    def __init__(self, text='["mech pilot desert war character", "desert mech anime"]'):
        self.text = text
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text=self.text)


@pytest.fixture
def gemini_llm(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("WAIFU_GEMINI_LLM", "1")
    models = FakeModels()
    monkeypatch.setattr(google_config, "client", lambda: SimpleNamespace(models=models))

    def no_http(*a, **k):
        raise AssertionError("the Gemini backend must not use the OpenAI-compatible path")

    monkeypatch.setattr(query_llm.urllib.request, "urlopen", no_http)
    return models


def test_gemini_llm_switch_alone_enables_the_query_llm(gemini_llm):
    assert query_llm.enabled() and query_llm.provider() == "gemini"
    assert query_llm.base_url() == "gemini"
    assert query_llm.status()["provider"] == "gemini"


def test_gemini_llm_rewrites_typed_text_with_the_selected_model(gemini_llm, monkeypatch):
    monkeypatch.setenv("WAIFU_GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
    got = query_llm.rewrite(["pilots a mech in a desert war"], "anime")
    assert got == ["mech pilot desert war character", "desert mech anime"]
    call = gemini_llm.calls[0]
    assert call["model"] == "gemini-2.5-flash-lite"
    assert "pilots a mech in a desert war" in call["contents"]
    assert "medium: anime" in call["contents"]
    cfg = call["config"]
    assert "JSON array" in cfg.system_instruction
    assert cfg.response_mime_type == "application/json"
    assert cfg.thinking_config.thinking_budget == 0
    assert cfg.max_output_tokens == 256  # zero budget: the reply is the whole cap
    assert "movies" in cfg.system_instruction and "TV series" in cfg.system_instruction
    assert cfg.tools is None  # no search tool: this call only writes queries


def test_gemini_llm_leaves_room_for_thinking_tokens(gemini_llm, monkeypatch):
    monkeypatch.setenv("WAIFU_GEMINI_LLM_MODEL", "gemini-3-flash")
    query_llm.rewrite(["pilots a mech"], "movie")
    assert gemini_llm.calls[0]["config"].max_output_tokens == 2048


def test_gemini_llm_runs_through_the_same_background_prefetch(gemini_llm):
    assert query_llm.prefetch(["mech pilot"]) is True
    assert query_llm.wait(["mech pilot"], timeout=5) == [
        "mech pilot desert war character", "desert mech anime"]
    assert query_llm.status()["calls"] == 1


def test_gemini_llm_failure_returns_none(gemini_llm, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(google_config, "client",
                        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=boom)))
    assert query_llm.rewrite(["x"]) is None


def test_without_a_key_the_openai_compatible_path_is_used(monkeypatch):
    monkeypatch.setenv("WAIFU_GEMINI_LLM", "1")  # no key: Gemini stays off
    monkeypatch.setenv("WAIFU_QUERY_LLM", "1")
    assert query_llm.provider() == "openai"
    assert query_llm.base_url() == query_llm.DEFAULT_BASE_URL


# --- .env loading -------------------------------------------------------------


def test_env_parse_handles_comments_quotes_and_export():
    text = """
# comment
GOOGLE_API_KEY=abc123  # trailing comment
export WAIFU_GEMINI_SEARCH=1
WAIFU_GEMINI_MODEL="gemini-2.5-flash"
QUOTED='keep # this'
GOOGLE_API_KEY_QUOTED="abc123"  # prod key
not a line
EMPTY=
"""
    assert envfile.parse(text) == {
        "GOOGLE_API_KEY": "abc123", "WAIFU_GEMINI_SEARCH": "1",
        "WAIFU_GEMINI_MODEL": "gemini-2.5-flash", "QUOTED": "keep # this",
        "GOOGLE_API_KEY_QUOTED": "abc123", "EMPTY": ""}


def test_env_file_fills_only_unset_variables(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("WAIFU_GEMINI_MODEL=from-file\nWAIFU_TEST_ONLY_VAR=file\n")
    monkeypatch.setenv("WAIFU_GEMINI_MODEL", "from-shell")
    monkeypatch.delenv("WAIFU_TEST_ONLY_VAR", raising=False)
    try:
        assert envfile.load(f) == ["WAIFU_TEST_ONLY_VAR"]
        assert os.environ["WAIFU_GEMINI_MODEL"] == "from-shell"
        assert os.environ["WAIFU_TEST_ONLY_VAR"] == "file"
    finally:
        os.environ.pop("WAIFU_TEST_ONLY_VAR", None)


def test_env_loading_can_be_off_and_missing_files_are_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("WAIFU_ENV_FILE", "0")
    assert envfile.load() == []
    assert envfile.load(tmp_path / "missing.env") == []


def test_example_env_file_documents_every_google_setting():
    example = envfile.parse(open(".env.example", encoding="utf-8").read())
    for var in ("GOOGLE_API_KEY", "WAIFU_GEMINI_MODEL", "WAIFU_GEMINI_SEARCH", "WAIFU_GEMINI_LLM"):
        assert var in example
    assert example["WAIFU_GEMINI_SEARCH"] == "0" and example["WAIFU_GEMINI_LLM"] == "0"
    assert example["GOOGLE_API_KEY"] == ""  # never ship a key
