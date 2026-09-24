"""One character, many spellings: dedup must not split a character in two."""

from __future__ import annotations

import pytest

from waifu_engine import sources, web_search
from waifu_engine.names import name_keys, same_character
from waifu_engine.nekomimi import session as sess_mod
from waifu_engine.sources import anilist, wikipedia


@pytest.mark.parametrize("a, b", [
    ("Koharu Shimoe", "Shimoe Koharu"),          # given-first vs family-first
    ("Shimoe, Koharu", "koharu shimoe"),
    ("Hatsune・Miku", "Miku Hatsune"),            # Japanese middle dot
    ("Spider-Man", "Spider Man"),                  # still matches without sorting
    ("Spider-Man", "spiderman"),
])
def test_spellings_of_one_character_match(a, b):
    assert same_character(a, b)


@pytest.mark.parametrize("a, b", [
    ("Rem", "Ram"),
    ("Asuka Langley Soryu", "Asuka Kazama"),
    ("Mario", "Luigi"),
])
def test_different_characters_stay_apart(a, b):
    assert not same_character(a, b)


def test_names_without_letters_have_no_key():
    assert name_keys("") == frozenset() and name_keys(" - ・ ") == frozenset()


@pytest.mark.parametrize("a, b", [
    ("Link", "Link (The Legend of Zelda)"),
    ("Link", "Link (The Legend of Zelda) (video game)"),
    ("Black Cat", "Black Cat (Marvel Comics)"),
    ("Hatsune Miku", "Hatsune Miku (anime)"),
    ("Koharu Shimoe", "Shimoe Koharu (Blue Archive)"),
])
def test_trailing_series_title_is_the_same_character(a, b):
    assert name_keys(a) & name_keys(b)
    assert same_character(a, b)


@pytest.mark.parametrize("a, b", [
    ("Young Link", "Link"),
    ("Toon Link", "Link"),
    ("Young Link", "Toon Link"),
    ("Young Link", "Link (The Legend of Zelda)"),
    ("Toon Link", "Link (The Legend of Zelda)"),
    ("Aqua (Kingdom Hearts)", "Aqua (KonoSuba)"),
])
def test_qualifiers_and_different_titles_stay_apart(a, b):
    assert not same_character(a, b)


def test_session_pool_merges_a_titled_form_and_keeps_young_link():
    s = sess_mod.new_session()
    added = s.add_candidates([
        {"id": "bare", "name": "Link", "series": "The Legend of Zelda"},
        {"id": "titled", "name": "Link (The Legend of Zelda)", "series": "The Legend of Zelda"},
        {"id": "young", "name": "Young Link", "series": "The Legend of Zelda"},
        {"id": "toon", "name": "Toon Link", "series": "The Legend of Zelda"},
    ])
    assert added == 3
    assert {c.name for c in s.candidates} == {"Link", "Young Link", "Toon Link"}
    assert s.add_candidates([
        {"id": "kh", "name": "Aqua (Kingdom Hearts)"},
        {"id": "ks", "name": "Aqua (KonoSuba)"},
    ]) == 2


def test_session_pool_rejects_the_other_name_order():
    s = sess_mod.new_session()
    added = s.add_candidates([
        {"id": "gem_1", "name": "Koharu Shimoe", "series": "Blue Archive"},
        {"id": "al_1", "name": "Shimoe Koharu", "series": "Blue Archive"},
    ])
    assert added == 1
    assert s.add_candidates([{"id": "wiki_1", "name": "Shimoe Koharu"}]) == 0
    assert [c.name for c in s.candidates] == ["Koharu Shimoe"]


def test_search_merge_keeps_one_of_two_name_orders(monkeypatch):
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    monkeypatch.setattr(web_search, "_want_ddg_fill", lambda *a, **k: False)
    monkeypatch.setattr(wikipedia, "search_characters",
                        lambda q, limit: [{"id": "w", "name": "Koharu Shimoe"}])
    monkeypatch.setattr(anilist, "search_characters",
                        lambda q, limit: [{"id": "a", "name": "Shimoe Koharu"},
                                          {"id": "b", "name": "Hifumi Ajitani"}])
    out = sources.find_candidates(["Blue Archive"], medium_hint="anime", limit=5)
    assert [c["name"] for c in out] == ["Koharu Shimoe", "Hifumi Ajitani"]
    # A name already in play is excluded in either order.
    again = sources.find_candidates(["Blue Archive"], medium_hint="anime", limit=5,
                                    exclude_names={"Shimoe Koharu"})
    assert [c["name"] for c in again] == ["Hifumi Ajitani"]


def test_search_merge_folds_a_titled_name_and_keeps_variants(monkeypatch):
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    monkeypatch.setattr(web_search, "_want_ddg_fill", lambda *a, **k: False)
    monkeypatch.setattr(wikipedia, "search_characters",
                        lambda q, limit: [{"id": "w", "name": "Link"}])
    monkeypatch.setattr(anilist, "search_characters",
                        lambda q, limit: [
                            {"id": "a", "name": "Link (The Legend of Zelda)"},
                            {"id": "b", "name": "Young Link"},
                            {"id": "c", "name": "Toon Link"},
                            {"id": "d", "name": "Aqua (Kingdom Hearts)"},
                            {"id": "e", "name": "Aqua (KonoSuba)"},
                        ])
    out = sources.find_candidates(["Zelda"], medium_hint="game", limit=8)
    assert [c["name"] for c in out] == [
        "Link", "Young Link", "Toon Link", "Aqua (Kingdom Hearts)", "Aqua (KonoSuba)",
    ]


def test_ddg_merge_uses_the_same_identity():
    found, seen = {}, set()
    web_search._take_candidate(found, seen, {"id": "1", "name": "Koharu Shimoe"}, set(), 5)
    web_search._take_candidate(found, seen, {"id": "2", "name": "Shimoe Koharu"}, set(), 5)
    assert list(found) == ["1"]
