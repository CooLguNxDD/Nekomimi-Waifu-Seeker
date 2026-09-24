"""Alias collapse, non-character quarantine, and headliner prior.

Honest-answer empty-seed bench (Asuka / Cloud / Wonder Woman) scored 1/3.
Asuka's mass was split across spellings while Kyoko and the Evangelion page
stayed whole. Wonder Woman lost to Nubia after royalty and black hair.
These tests lock the offline ranking that those misses exposed.
"""

from __future__ import annotations

import pytest

from waifu_engine import web_search
from waifu_engine.names import canonical_name
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits
from waifu_engine.nekomimi.session import Candidate


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Heuristic scoring only, and no search."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    monkeypatch.setattr(engine, "ONLINE", False)


def _series_detail(detail: str) -> dict:
    """A series=other answer whose free text is ``detail``."""
    return {
        "qid": "series", "text": "Which series is your character from?",
        "answer": "other", "kind": "choice", "detail": detail,
        "options": {"other": {"label": "Another series"}},
    }


def _rank(rows, asked=None, detail=""):
    """Posterior after replaying ``asked`` with an optional series detail."""
    sess = sess_mod.new_session()
    sess.add_candidates(rows)
    if detail:
        sess.asked.append(_series_detail(detail))
    for qid, answer in asked or []:
        engine.score_candidates(sess, traits.QUESTIONS_BY_ID[qid], answer)
    if not asked:
        engine._rescore_candidates(sess)
    return [(c.name, p) for c, p in sess.posterior()], sess


def test_asuka_spellings_collapse_to_one_canonical_row():
    """Soryu, Sohryu, Shikinami and bare Langley are one probability mass."""
    sess = sess_mod.new_session()
    added = sess.add_candidates([
        {"id": "soryu", "name": "Asuka Langley Soryu", "series": "Neon Genesis Evangelion",
         "popularity": 1000, "tags": ["female"]},
        {"id": "sohryu", "name": "Asuka Langley Sohryu", "series": "Neon Genesis Evangelion",
         "popularity": 800, "tags": ["red"]},
        {"id": "langley", "name": "Asuka Langley", "series": "Neon Genesis Evangelion",
         "tags": ["twintails"]},
        {"id": "shikinami", "name": "Asuka Shikinami Langley", "popularity": 50},
        {"id": "bare", "name": "Soryu", "popularity": 10},
        {"id": "kyoko", "name": "Kyoko Zeppelin Soryu", "series": "Neon Genesis Evangelion",
         "popularity": 5000, "tags": ["female", "red"]},
    ])
    assert added == 2
    names = [c.name for c in sess.alive_candidates()]
    assert names.count("Asuka Langley Soryu") == 1
    assert "Kyoko Zeppelin Soryu" in names
    keeper = sess.by_id("soryu")
    assert keeper is not None and keeper.alive
    assert {"female", "red", "twintails"} <= set(keeper.tags)
    assert canonical_name("Sohryu") == "Asuka Langley Soryu"


def test_asuka_alias_mass_beats_kyoko_and_pages_are_not_guessable():
    """After female and red hair, merged Asuka leads Kyoko. Evangelion is not a guess."""
    rows = [
        {"id": "shinji", "name": "Shinji Ikari", "series": "Neon Genesis Evangelion",
         "medium": "anime", "popularity": 8000, "tags": ["male", "brown"],
         "blurb": "A boy who pilots an Eva."},
        {"id": "kyoko", "name": "Kyoko Zeppelin Soryu", "series": "Neon Genesis Evangelion",
         "medium": "anime", "popularity": 9000, "tags": ["female", "red"],
         "blurb": "A woman with red hair."},
        {"id": "soryu", "name": "Asuka Langley Soryu", "series": "Neon Genesis Evangelion",
         "medium": "anime", "popularity": 4000, "tags": ["female", "red"],
         "blurb": "A pilot with red hair."},
        {"id": "sohryu", "name": "Asuka Langley Sohryu", "series": "Neon Genesis Evangelion",
         "medium": "anime", "popularity": 3000, "tags": ["female", "red"],
         "blurb": "Romanisation of Asuka Langley Soryu."},
        {"id": "langley", "name": "Asuka Langley", "series": "Neon Genesis Evangelion",
         "medium": "anime", "popularity": 2000, "tags": ["female"]},
        {"id": "shiki", "name": "Asuka Shikinami Langley", "series": "Neon Genesis Evangelion",
         "medium": "anime", "popularity": 2000, "tags": ["female", "red"]},
        {"id": "eva", "name": "Evangelion", "series": "Neon Genesis Evangelion",
         "medium": "anime", "popularity": 10**9,
         "blurb": "Neon Genesis Evangelion is a Japanese media franchise."},
        {"id": "angels", "name": "Angels", "series": "Neon Genesis Evangelion",
         "medium": "anime", "popularity": 10**8,
         "blurb": "The Angels are a fictional species."},
    ]
    ranked, sess = _rank(rows, [("gender_female", "yes"), ("hair_color", "red")], detail="evangelion")
    names = [name for name, _p in ranked]
    assert names[0] == "Asuka Langley Soryu"
    assert "Kyoko Zeppelin Soryu" in names
    assert names.index("Asuka Langley Soryu") < names.index("Kyoko Zeppelin Soryu")
    assert "Evangelion" not in names and "Angels" not in names
    assert sess.by_id("eva") is None or not any(c.id == "eva" for c, _ in sess.posterior())
    assert sum(1 for name in names if name == "Asuka Langley Soryu") == 1


def test_unresolved_alias_split_blocks_the_guess(monkeypatch):
    """Two high-mass spellings of one person are not a commit."""
    monkeypatch.setattr(sess_mod.GuessSession, "collapse_identities", lambda self: 0)
    sess = sess_mod.new_session()
    sess.turn = 8
    sess.leader_id = "soryu"
    sess.leader_streak = 5
    q_gender = traits.QUESTIONS_BY_ID["gender_female"]
    q_pilot = traits.QUESTIONS_BY_ID["job_pilot"]
    sess.evidence = {
        "gender_female": (q_gender, "yes"),
        "job_pilot": (q_pilot, "yes"),
    }
    for cid, odds in (("soryu", 1.4), ("sohryu", 1.0), ("kyoko", 0.0)):
        sess.match_cache[(cid, "gender_female")] = 0.9
        sess.match_cache[(cid, "job_pilot")] = 0.9
    sess.candidates = [
        Candidate(id="soryu", name="Asuka Langley Soryu", series="Neon Genesis Evangelion",
                  logodds=1.4, tags=["female"]),
        Candidate(id="sohryu", name="Asuka Langley Sohryu", series="Neon Genesis Evangelion",
                  logodds=1.0, tags=["female"]),
        Candidate(id="kyoko", name="Kyoko Zeppelin Soryu", series="Neon Genesis Evangelion",
                  logodds=0.0, tags=["female"]),
    ]
    assert engine._alias_split_blocks_guess(sess)
    assert engine._should_guess(sess, None) is False


def test_collapsed_asuka_can_still_be_guessed():
    """Once the spellings are one row, a strong leader is allowed to commit."""
    sess = sess_mod.new_session()
    sess.candidates = [
        Candidate(id="soryu", name="Asuka Langley Soryu", series="Neon Genesis Evangelion",
                  logodds=2.0, popularity=4000, tags=["female", "red"]),
        Candidate(id="sohryu", name="Asuka Langley Sohryu", series="Neon Genesis Evangelion",
                  logodds=1.5, popularity=1000, tags=["female"]),
        Candidate(id="kyoko", name="Kyoko Zeppelin Soryu", series="Neon Genesis Evangelion",
                  logodds=0.0, popularity=1000, tags=["female"]),
    ]
    assert sess.collapse_identities() == 1
    assert [c.name for c in sess.alive_candidates()] == [
        "Asuka Langley Soryu", "Kyoko Zeppelin Soryu",
    ]
    q_gender = traits.QUESTIONS_BY_ID["gender_female"]
    q_pilot = traits.QUESTIONS_BY_ID["job_pilot"]
    sess.evidence = {
        "gender_female": (q_gender, "yes"),
        "job_pilot": (q_pilot, "yes"),
    }
    sess.match_cache[("soryu", "gender_female")] = 0.9
    sess.match_cache[("soryu", "job_pilot")] = 0.9
    sess.match_cache[("kyoko", "gender_female")] = 0.55
    sess.match_cache[("kyoko", "job_pilot")] = 0.55
    sess.turn = 8
    assert engine._should_guess(sess, None) is True


def test_wrong_guess_rejects_the_identity_and_merges_the_rest():
    """Rejecting one spelling rejects the person. Other aliases re-merge."""
    sess = sess_mod.new_session()
    sess.turn = 6
    sess.stage = "guessing"
    sess.pending_guess = "kyoko"
    sess.guesses_made = 1
    sess.candidates = [
        Candidate(id="soryu", name="Soryu", series="Neon Genesis Evangelion",
                  logodds=1.0, tags=["female", "twintails"], popularity=1000),
        Candidate(id="sohryu", name="Asuka Langley Sohryu", series="Neon Genesis Evangelion",
                  logodds=0.8, tags=["red"], popularity=800),
        Candidate(id="kyoko", name="Kyoko Zeppelin Soryu", series="Neon Genesis Evangelion",
                  logodds=1.2, tags=["female", "brown"], popularity=2000),
    ]
    state = engine.submit_guess_result(sess, correct=False)
    assert "kyoko" in sess.rejected
    assert sess.by_id("kyoko").alive is False
    alive = [c.name for c in sess.alive_candidates()]
    assert alive == ["Asuka Langley Soryu"]
    assert {"female", "twintails", "red"} <= set(sess.by_id("soryu").tags)
    assert state["stage"] == "asking"
    assert state["question"]["qid"] == "hair_color"


def test_rejecting_one_asuka_spelling_rejects_the_rest():
    """A wrong guess of one spelling must not leave another spelling to commit."""
    sess = sess_mod.new_session()
    sess.turn = 6
    sess.stage = "guessing"
    sess.pending_guess = "soryu"
    sess.guesses_made = 1
    sess.candidates = [
        Candidate(id="soryu", name="Asuka Langley Soryu", series="Neon Genesis Evangelion",
                  logodds=1.0, tags=["female", "red"], popularity=1000),
        Candidate(id="sohryu", name="Asuka Langley Sohryu", series="Neon Genesis Evangelion",
                  logodds=0.9, tags=["female", "red"], popularity=900),
        Candidate(id="rei", name="Rei Ayanami", series="Neon Genesis Evangelion",
                  logodds=0.2, tags=["female", "blue"], popularity=3000),
    ]
    engine.submit_guess_result(sess, correct=False)
    assert {"soryu", "sohryu"} <= sess.rejected
    assert sess.by_id("rei").alive is True
    assert all(not c.alive for c in sess.candidates if c.id in {"soryu", "sohryu"})


def test_wonder_woman_and_diana_are_one_row():
    """Wonder Woman and Diana Prince do not split their own posterior."""
    sess = sess_mod.new_session()
    added = sess.add_candidates([
        {"id": "ww", "name": "Wonder Woman", "series": "DC Comics", "medium": "comic",
         "popularity": 8000, "tags": ["female"]},
        {"id": "diana", "name": "Diana Prince", "series": "Wonder Woman", "medium": "comic",
         "popularity": 6000, "tags": ["royalty"]},
        {"id": "themy", "name": "Diana of Themyscira", "popularity": 100},
    ])
    assert added == 1
    assert [c.name for c in sess.alive_candidates()] == ["Wonder Woman"]
    assert "royalty" in sess.by_id("ww").tags


def test_wonder_woman_beats_nubia_after_royalty_and_black_hair():
    """Royalty plus black hair must not hand the book to Nubia."""
    rows = [
        {"id": "ww", "name": "Wonder Woman", "series": "DC Comics", "medium": "comic",
         "popularity": 8000, "tags": ["female"],
         "blurb": "A superhero. Her civilian name is Diana Prince."},
        {"id": "diana", "name": "Diana Prince", "series": "Wonder Woman", "medium": "comic",
         "popularity": 5000, "tags": ["female"],
         "blurb": "Diana Prince is Wonder Woman."},
        {"id": "nubia", "name": "Nubia", "series": "Wonder Woman", "medium": "comic",
         "popularity": 8000, "tags": ["female", "royalty", "black"],
         "blurb": "An Amazon princess with black hair in Wonder Woman."},
        {"id": "orana", "name": "Orana", "series": "Wonder Woman", "medium": "comic",
         "popularity": 3000, "tags": ["female", "royalty"],
         "blurb": "An Amazon who briefly claimed the Wonder Woman title."},
        {"id": "fury", "name": "Fury", "series": "Wonder Woman", "medium": "comic",
         "popularity": 4000, "tags": ["female", "black"],
         "blurb": "An Amazon with black hair in Wonder Woman."},
    ]
    ranked, _sess = _rank(
        rows,
        [("job_royalty", "yes"), ("hair_color", "black")],
        detail="wonder woman",
    )
    by = dict(ranked)
    assert "Wonder Woman" in by and "Diana Prince" not in by
    assert by["Wonder Woman"] >= by["Nubia"]
    assert ranked[0][0] == "Wonder Woman"


def test_cloud_stays_ahead_of_the_final_fantasy_vii_cast():
    """Naming Final Fantasy VII prefers Cloud. A hard antagonist miss can still overrule."""
    rows = [
        {"id": "cloud", "name": "Cloud Strife", "series": "Final Fantasy VII",
         "medium": "game", "popularity": 20000, "tags": ["male"],
         "blurb": "A mercenary in Final Fantasy VII."},
        {"id": "zack", "name": "Zack Fair", "series": "Final Fantasy VII",
         "medium": "game", "popularity": 20000, "tags": ["male"],
         "blurb": "A SOLDIER in Final Fantasy VII."},
        {"id": "seph", "name": "Sephiroth", "series": "Final Fantasy VII",
         "medium": "game", "popularity": 20000, "tags": ["male", "antagonist"],
         "blurb": "The antagonist of Final Fantasy VII."},
    ]
    ranked, _sess = _rank(rows, [("gender_male", "yes")], detail="final fantasy vii")
    assert ranked[0][0] == "Cloud Strife"
    sess = sess_mod.new_session()
    sess.add_candidates(rows)
    sess.asked.append(_series_detail("ff7"))
    for cid, p in (("cloud", 0.08), ("zack", 0.2), ("seph", 0.93)):
        sess.match_cache[(cid, "role_antagonist")] = p
    engine.score_candidates(sess, traits.QUESTIONS_BY_ID["role_antagonist"], "yes")
    assert sess.posterior()[0][0].name == "Sephiroth"


def test_franchise_page_cannot_be_the_guess():
    """A franchise article with huge fame is invisible to the guess."""
    sess = sess_mod.new_session()
    added = sess.add_candidates([
        {"id": "eva", "name": "Evangelion", "popularity": 10**9,
         "blurb": "Neon Genesis Evangelion is a Japanese media franchise."},
        {"id": "asuka", "name": "Asuka Langley Soryu", "series": "Neon Genesis Evangelion",
         "popularity": 10, "tags": ["female"]},
    ])
    assert added == 1
    sess.candidates.append(Candidate(
        id="stuck", name="Angels", logodds=80, popularity=10**9,
        blurb="The Angels are a fictional species.",
    ))
    payload = engine._guess_payload(sess)
    assert payload["guess"]["name"] == "Asuka Langley Soryu"
    assert web_search.is_non_character("Evangelion")
    assert web_search.is_non_character("Angels", "", "")
    assert web_search.is_non_character(
        "Lilith", "", "Lilith is a fictional species in that series.",
    )
    assert not web_search.is_non_character(
        "Shinji Ikari", "",
        "Shinji Ikari is the protagonist of the anime series Neon Genesis Evangelion.",
    )
    assert not web_search.is_non_character(
        "Lisa", "", "Lisa is a series regular on the show.",
    )
    assert not web_search.is_non_character("Hatsune Miku")
    assert not web_search.is_non_character("Cloud Strife")


def test_protagonist_of_a_franchise_stays_a_character():
    """A person who belongs to a franchise is not the franchise page.

    The word gap before "franchise" used to swallow "of", so "protagonist of
    the Metroid franchise" and "member of the Saiyan species" were quarantined.
    "species of" is a creature, not that page.
    """
    assert not web_search.is_non_character(
        "Samus Aran", "",
        "Samus Aran is the protagonist of the Metroid franchise.",
    )
    assert not web_search.is_non_character(
        "Vegeta", "",
        "Vegeta is a member of the Saiyan species.",
    )
    assert not web_search.is_non_character(
        "Pikachu", "",
        "Pikachu is a species of Pokémon.",
    )
    assert web_search.is_non_character(
        "Metroid", "", "Metroid is a Japanese media franchise.",
    )
    assert web_search.is_non_character(
        "Saiyans", "", "The Saiyans are a fictional species.",
    )
    assert web_search.is_non_character(
        "Metroid", "", "Metroid is a series of video games.",
    )


def test_namesake_from_another_work_is_not_absorbed():
    """Kingdom Hearts Aqua must not take a KonoSuba blurb or tags.

    ``same_character`` is true for bare "Aqua" and "Aqua (KonoSuba)". Absorb
    still requires one lexicon id, or series fields that do not name two works.
    """
    sess = sess_mod.new_session()
    added = sess.add_candidates([
        {"id": "kh", "name": "Aqua", "series": "Kingdom Hearts", "medium": "game",
         "tags": ["blue"], "blurb": "A Keyblade wielder.", "popularity": 100},
        {"id": "ks", "name": "Aqua (KonoSuba)", "series": "KonoSuba", "medium": "anime",
         "tags": ["mage"],
         "blurb": "A goddess with a much longer blurb from the other work.",
         "popularity": 9000},
    ])
    assert added == 1
    row = sess.by_id("kh")
    assert row.tags == ["blue"]
    assert row.blurb == "A Keyblade wielder."
    assert row.popularity == 100
    assert sess.add_candidates([
        {"id": "ks2", "name": "Aqua", "series": "KonoSuba", "tags": ["mage"],
         "blurb": "Another longer KonoSuba blurb that must not replace the first."},
    ]) == 0
    assert row.tags == ["blue"]
    assert row.blurb == "A Keyblade wielder."


def test_wonder_woman_and_ff7_resolve_without_breaking_mario():
    """Series-other text names those works. "mario" is still the whole clue only."""
    eva = sess_mod.new_session()
    eva.asked.append(_series_detail("neon genesis evangelion"))
    assert engine._confirmed_series(eva) == "Neon Genesis Evangelion"
    ff = sess_mod.new_session()
    ff.asked.append(_series_detail("final fantasy vii"))
    assert engine._confirmed_series(ff) == "Final Fantasy VII"
    ww = sess_mod.new_session("wonder woman")
    assert engine._typed_franchise(ww) == "Wonder Woman"
    assert engine._typed_franchise(sess_mod.new_session("mario")) == "Super Mario"
    assert engine._typed_franchise(sess_mod.new_session("Mario Rossi")) == ""
