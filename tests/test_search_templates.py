"""Search strings keep a resolved work, and name search uses character names.

Empty-seed rebench: after a series was only the first template fact, the
narrow retry was "female character" / "long-haired character". That filled
the pool with other franchises (Zero Two) and later mantle-holders (Yara
Flor). AniList's name search on the same sentence returned unrelated leads.
"""

from waifu_engine import sources, web_search
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.sources import anilist, wikipedia


def _series_detail(detail: str) -> dict:
    """A series=other answer whose free text is ``detail``."""
    return {
        "qid": "series", "text": "Which series is your character from?",
        "answer": "other", "kind": "choice", "detail": detail, "category": "series",
        "options": {"other": {"label": "Another series", "fact": "", "prior": 0.3, "tags": []}},
    }


def _yes(qid: str) -> dict:
    """A yes answer stored the way ``submit_answer`` leaves it."""
    question = traits.QUESTIONS_BY_ID[qid]
    return {
        "qid": question["id"], "text": question["text"], "category": question["category"],
        "prior": question["prior"], "kind": question.get("kind", "yesno"),
        "answer": "yes", "detail": None, "options": question.get("options"),
    }


def _choice(qid: str, answer: str, detail: str | None = None) -> dict:
    """A choice answer, including a colour typed on the hair question."""
    question = traits.QUESTIONS_BY_ID[qid]
    return {
        "qid": question["id"], "text": question["text"], "category": question["category"],
        "prior": question["prior"], "kind": "choice", "answer": answer,
        "detail": detail, "options": question["options"],
    }


def test_unpinned_narrow_retry_is_still_the_newest_fact():
    """Without a resolved work, the newest clue is still the short retry."""
    queries = sources._queries(
        ["Nintendo", "male", "red cap", "Italian plumber"], "game")
    assert queries[-1] == "Italian plumber video game character"


def test_resolved_work_stays_in_every_template_group():
    """The narrow group is the work, not the newest trait."""
    pin = "Neon Genesis Evangelion"
    early = sources._template_queries([pin, "red hair", "female"], "anime", pin=pin)
    assert early[-1] == "Neon Genesis Evangelion anime character"
    assert any("red hair" in query and "female" in query for query in early)
    later = sources._template_queries(
        [pin, "red hair", "female", "long-haired", "a pilot of a mech, plane or spacecraft"],
        "anime", pin=pin)
    assert later[-1] == "Neon Genesis Evangelion anime character"
    for query in (*early, *later):
        assert "Neon Genesis Evangelion" in query
        assert not query.lower().startswith("female")
        assert not query.lower().startswith("long-haired")


def test_personal_alias_stays_on_a_mantle_title():
    """A disjoint personal name is part of the pin, so later holders are not the only hit."""
    assert web_search.disambiguating_alias("Wonder Woman", "Wonder Woman") == "Diana Prince"
    assert web_search.disambiguating_alias("Asuka Langley Soryu", "Neon Genesis Evangelion") == ""
    parts = web_search.fulltext_pin_parts(["wonder woman", "long-haired"])
    assert parts[0] == "Wonder Woman"
    assert "Diana Prince" in parts
    queries = sources._template_queries(
        ["Wonder Woman", "black hair", "a soldier or fighter", "long-haired"],
        "comic", pin=" ".join(parts))
    for query in queries:
        assert "Wonder Woman" in query
        assert "Diana Prince" in query
    assert queries[-1] == "Wonder Woman Diana Prince comic book character"


def test_one_title_name_is_pinned_and_co_leads_are_name_queries():
    """A single headliner joins the full-text pin. Several co-leads are separate name searches."""
    cloud = web_search.fulltext_pin_parts(["final fantasy vii"])
    assert cloud == ["Final Fantasy VII", "Cloud Strife"]
    eva = web_search.fulltext_pin_parts(["neon genesis evangelion"])
    assert eva == ["Neon Genesis Evangelion"]
    assert web_search.title_character_queries(["evangelion"]) == [
        "Shinji Ikari", "Asuka Langley Soryu", "Rei Ayanami",
    ]
    assert "Diana Prince" in web_search.title_character_queries(["wonder woman"])


def test_anilist_plan_uses_title_names_not_the_trait_sentence():
    """AniList must be asked for Asuka by name, not for the whole fact string."""
    names, per, stop = sources._anilist_plan(
        ["Neon Genesis Evangelion", "red hair", "female"],
        "Neon Genesis Evangelion",
        ["Neon Genesis Evangelion red hair female anime character"],
        12,
    )
    assert names == ["Shinji Ikari", "Asuka Langley Soryu", "Rei Ayanami"]
    assert per == 3
    assert stop is False
    plain, _, stop_plain = sources._anilist_plan(
        ["Italian plumber"], None, ["Italian plumber video game character"], 12)
    assert plain == ["Italian plumber video game character"]
    assert stop_plain is True


def test_anilist_requests_each_title_name(monkeypatch):
    """A full pool from the first name must not skip the later title names."""
    monkeypatch.setattr(web_search, "_want_playwright", lambda: False)
    monkeypatch.setattr(web_search, "_enrich_on", lambda: False)
    monkeypatch.setattr(wikipedia, "search_characters", lambda *a, **k: [])
    seen = []

    def search(query, limit=10):
        seen.append((query, limit))
        return [{"id": query, "name": query, "popularity": 1}]

    monkeypatch.setattr(anilist, "search_characters", search)
    sources.find_candidates(
        ["Neon Genesis Evangelion", "red hair", "female"],
        medium_hint="anime", limit=2, use_ddg=False,
        pin="Neon Genesis Evangelion",
    )
    # limit=2 would stop after two hits if a full page ended the loop.
    assert [query for query, _limit in seen] == [
        "Shinji Ikari", "Asuka Langley Soryu", "Rei Ayanami",
    ]
    assert {limit for _query, limit in seen} == {2}


def test_empty_seed_turns_keep_the_work_in_the_engine_pin(monkeypatch):
    """The Asuka and Wonder Woman answer sequences never emit a series-free query."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    seen = {}

    def capture(terms, **kwargs):
        seen["terms"] = terms
        seen["pin"] = kwargs.get("pin")
        seen["medium"] = kwargs.get("medium_hint")
        seen["focus"] = kwargs.get("focus")
        return []

    monkeypatch.setattr(engine.sources, "find_candidates", capture)

    asuka = sess_mod.new_session("")
    asuka.asked = [
        _series_detail("neon genesis evangelion"),
        _choice("hair_color", "red", "orange"),
        _choice("medium", "anime"),
        _yes("gender_female"),
    ]
    engine.refresh_candidates(asuka)
    assert seen["pin"] == "Neon Genesis Evangelion"
    assert seen["medium"] == "anime"
    asuka_queries = sources._queries(
        seen["terms"], seen["medium"], focus=seen["focus"], pin=seen["pin"])
    assert asuka_queries
    for query in asuka_queries:
        assert query.startswith("Neon Genesis Evangelion")
    assert any("red hair" in query for query in asuka_queries)
    assert "Asuka Langley Soryu" in sources._anilist_plan(
        seen["terms"], seen["pin"], asuka_queries, 12)[0]

    wonder = sess_mod.new_session("")
    wonder.asked = [
        _series_detail("wonder woman"),
        _choice("hair_color", "black"),
        _choice("medium", "comic"),
        _yes("job_soldier"),
        _yes("gender_female"),
        _yes("hair_long"),
    ]
    engine.refresh_candidates(wonder)
    assert seen["pin"] == "Wonder Woman Diana Prince"
    wonder_queries = sources._queries(
        seen["terms"], seen["medium"], focus=seen["focus"], pin=seen["pin"])
    for query in wonder_queries:
        assert "Wonder Woman" in query and "Diana Prince" in query
    assert all(not query.lower().startswith("long-haired") for query in wonder_queries)


def test_legacy_constraint_query_puts_a_dropped_work_back():
    """The last-six DDG window used to forget a series answered first."""
    facts = ["wonder woman", *(f"trait {i}" for i in range(8)), "long-haired"]
    text = web_search._constraint_query_text(facts)
    assert text.lower().startswith("diana prince") or "diana prince" in text.lower()
    assert "wonder woman" in text.lower()
    assert "long-haired" in text
