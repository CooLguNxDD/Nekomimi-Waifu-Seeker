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


def test_spice_and_wolf_aliases_confirm_the_series():
    """Typed Spice and Wolf aliases pin the work the way Fullmetal Alchemist does."""
    for detail in (
        "spice and wolf",
        "spice & wolf",
        "spice wolf",
        "ookami to koushinryou",
        "ookami to koshinryo",
        "ookami to ningen",
    ):
        sess = sess_mod.new_session()
        sess.asked.append(_series_detail(detail))
        assert engine._confirmed_series(sess) == "Spice and Wolf"
        assert engine._typed_franchise(sess) == "Spice and Wolf"
    assert web_search.is_non_character("Spice and Wolf")
    assert not web_search.is_non_character("Holo")


def test_batman_spellings_collapse_and_lead_the_cast():
    """Batman, Bruce Wayne and Absolute Batman are one row, ahead of mantle-holders."""
    sess = sess_mod.new_session()
    added = sess.add_candidates([
        {"id": "bats", "name": "Batman", "series": "Batman", "medium": "comic",
         "popularity": 8000, "tags": ["male"], "blurb": "The Dark Knight of Gotham."},
        {"id": "bruce", "name": "Bruce Wayne", "series": "DC Comics", "medium": "comic",
         "popularity": 6000, "tags": ["vigilante"], "blurb": "Bruce Wayne is Batman."},
        {"id": "abs", "name": "Absolute Batman", "series": "Batman", "medium": "comic",
         "popularity": 1000, "tags": ["cape"], "blurb": "Absolute Batman in Gotham."},
    ])
    assert added == 1
    row = sess.by_id("bats")
    assert row is not None and row.name == "Batman"
    assert {"male", "vigilante", "cape"} <= set(row.tags)
    ranked, _sess = _rank(
        [
            {"id": "bats", "name": "Batman", "series": "Batman", "medium": "comic",
             "popularity": 4000, "tags": ["male", "vigilante", "cape"],
             "blurb": "Bruce Wayne fights crime in Gotham as Batman."},
            {"id": "dick", "name": "Dick Grayson", "series": "Batman", "medium": "comic",
             "popularity": 9000, "tags": ["male", "vigilante", "cape"],
             "blurb": "A vigilante who wore the Batman mantle."},
            {"id": "jp", "name": "Jean-Paul Valley", "series": "Batman", "medium": "comic",
             "popularity": 9000, "tags": ["male", "vigilante", "cape", "armor"],
             "blurb": "Jean-Paul Valley took the Batman mantle and wears armor."},
        ],
        [("role_vigilante", "yes"), ("look_cape", "yes")],
        detail="batman",
    )
    names = [name for name, _p in ranked]
    assert names[0] == "Batman"
    assert "Dick Grayson" in names and "Jean-Paul Valley" in names
    assert web_search.fulltext_pin_parts(["batman"])[:2] == ["Batman", "Bruce Wayne"]


def test_trafalgar_law_spellings_collapse_and_luffy_is_the_one_piece_lead():
    """Law's full name merges. The One Piece prior is Luffy's, not a Warlord's."""
    sess = sess_mod.new_session()
    added = sess.add_candidates([
        {"id": "law", "name": "Trafalgar Law", "series": "One Piece", "medium": "anime",
         "popularity": 5000, "tags": ["male"]},
        {"id": "full", "name": "Trafalgar D. Water Law", "series": "One Piece",
         "medium": "anime", "popularity": 1000, "tags": ["surgeon"]},
    ])
    assert added == 1
    assert sess.by_id("law").name == "Trafalgar Law"
    assert "surgeon" in sess.by_id("law").tags
    luffy = sess_mod.new_session()
    assert luffy.add_candidates([
        {"id": "short", "name": "Luffy", "series": "One Piece", "popularity": 10},
        {"id": "full", "name": "Monkey D. Luffy", "series": "One Piece", "popularity": 10},
    ]) == 1
    assert [c.name for c in luffy.alive_candidates()] == ["Monkey D. Luffy"]
    ranked, _sess = _rank(
        [
            {"id": "luffy", "name": "Monkey D. Luffy", "series": "One Piece",
             "medium": "anime", "popularity": 4000, "tags": ["male", "protagonist"],
             "blurb": "The captain of the Straw Hat Pirates in One Piece."},
            {"id": "law", "name": "Trafalgar Law", "series": "One Piece",
             "medium": "anime", "popularity": 9000, "tags": ["male"],
             "blurb": "Captain of the Heart Pirates in One Piece."},
            {"id": "mihawk", "name": "Dracule Mihawk", "series": "One Piece",
             "medium": "anime", "popularity": 9000, "tags": ["male", "eyepatch"],
             "blurb": "A swordsman with an eyepatch in One Piece."},
        ],
        [("gender_male", "yes")],
        detail="one piece",
    )
    assert ranked[0][0] == "Monkey D. Luffy"
    assert "Trafalgar Law" in [name for name, _p in ranked]
    assert web_search.fulltext_pin_parts(["one piece"]) == ["One Piece", "Monkey D. Luffy"]


def test_edward_and_holo_outrank_same_work_lookalikes():
    """Naming the work lifts the title character over a tagged side character."""
    edward, _sess = _rank(
        [
            {"id": "ed", "name": "Edward Elric", "series": "Fullmetal Alchemist",
             "medium": "anime", "popularity": 4000, "tags": ["male", "sibling", "military"],
             "blurb": "The Fullmetal Alchemist, a boy with an automail arm."},
            {"id": "heinkel", "name": "Heinkel", "series": "Fullmetal Alchemist",
             "medium": "anime", "popularity": 9000, "tags": ["male", "sibling", "military"],
             "blurb": "A chimera soldier in the Amestrian military."},
        ],
        [("bond_sibling", "yes"), ("team_military", "yes")],
        detail="fullmetal alchemist",
    )
    assert edward[0][0] == "Edward Elric"
    assert "Heinkel" in [name for name, _p in edward]
    assert web_search.fulltext_pin_parts(["fullmetal alchemist"]) == [
        "Fullmetal Alchemist", "Edward Elric",
    ]
    holo, _sess = _rank(
        [
            {"id": "holo", "name": "Holo", "series": "Spice and Wolf",
             "medium": "anime", "popularity": 3000, "tags": ["female", "animal-ears", "tail"],
             "blurb": "The wolf harvest goddess of Spice and Wolf."},
            {"id": "myuri", "name": "Myuri", "series": "Spice and Wolf",
             "medium": "anime", "popularity": 9000, "tags": ["female", "animal-ears", "tail"],
             "blurb": "Holo's daughter, a girl with wolf ears and a tail."},
        ],
        [("gender_female", "yes"), ("look_tail", "yes")],
        detail="spice and wolf",
    )
    assert holo[0][0] == "Holo"
    assert web_search.fulltext_pin_parts(["ookami to koushinryou"]) == ["Spice and Wolf", "Holo"]


def test_series_lock_pins_a_splitting_look_instead_of_the_angel_kit():
    """After the work is known, a prosthetic that splits the cast leads halo."""
    sess = sess_mod.new_session()
    sess.add_candidates([
        {"id": "ed", "name": "Edward Elric", "series": "Fullmetal Alchemist",
         "medium": "anime", "tags": ["male", "prosthetic"],
         "blurb": "A short alchemist."},
        {"id": "heinkel", "name": "Heinkel", "series": "Fullmetal Alchemist",
         "medium": "anime", "tags": ["male"], "blurb": "A chimera soldier."},
        {"id": "roy", "name": "Roy Mustang", "series": "Fullmetal Alchemist",
         "medium": "anime", "tags": ["male"], "blurb": "A flame alchemist."},
        {"id": "mika", "name": "Misono Mika", "series": "Blue Archive",
         "medium": "game", "tags": ["animal-ears"], "blurb": "A student with animal ears."},
    ])
    sess.asked.append(_series_detail("fullmetal alchemist"))
    pins = engine._pinned_question_ids(sess)
    assert pins[0] == "hair_color"
    assert "look_prosthetic" in pins
    assert "look_animal_ears" not in pins
    assert "look_halo" not in pins and "look_wings" not in pins and "look_horns" not in pins
    ids = [q["id"] for q in engine.candidate_questions(sess)]
    assert ids[0] == "hair_color"
    assert ids[1] == "look_prosthetic"

    ears = sess_mod.new_session()
    ears.add_candidates([
        {"id": "holo", "name": "Holo", "series": "Spice and Wolf", "medium": "anime",
         "tags": ["female"], "blurb": "She keeps her wolf ears when she travels."},
        {"id": "lawrence", "name": "Kraft Lawrence", "series": "Spice and Wolf",
         "medium": "anime", "tags": ["male"], "blurb": "A traveling merchant."},
    ])
    ears.asked.append(_series_detail("spice & wolf"))
    ear_pins = engine._pinned_question_ids(ears)
    assert "look_animal_ears" in ear_pins
    assert "look_halo" not in ear_pins


def _arm_guess(sess, leader_id: str) -> None:
    """Two strong model judgments so an early guess is otherwise allowed."""
    sess.turn = 8
    for qid, p in (("gender_male", 0.9), ("age_adult", 0.8)):
        question = traits.QUESTIONS_BY_ID[qid]
        sess.evidence[qid] = (question, "yes")
        sess.match_cache[(leader_id, qid)] = p


def test_fairy_tail_title_is_not_a_physical_tail_split():
    """The guild name is not a body part, and a real tail still splits the cast."""
    question = traits.QUESTIONS_BY_ID["look_tail"]
    guild = Candidate(
        id="lucy", name="Lucy Heartfilia", series="Fairy Tail",
        blurb="A celestial mage in the Fairy Tail guild.", tags=["tail"],
    )
    quiet = Candidate(
        id="gray", name="Gray Fullbuster", series="Fairy Tail",
        blurb="An ice mage.",
    )
    tailed = Candidate(
        id="happy", name="Happy", series="Fairy Tail",
        blurb="A blue cat with a tail who travels with Fairy Tail.",
    )
    assert "tail" in web_search.mine_trait_slugs(guild.blurb)
    assert engine._candidate_has_trait(question, guild) is False
    assert engine._candidate_has_trait(question, quiet) is False
    assert engine._candidate_has_trait(question, tailed) is True

    sess = sess_mod.new_session()
    sess.candidates = [guild, quiet]
    sess.asked.append({
        "qid": "series", "text": "Which series?", "answer": "fairytail",
        "kind": "choice", "detail": None,
        "options": {"fairytail": {
            "label": "Fairy Tail", "series_key": "fairytail", "fact": "Fairy Tail",
        }},
    })
    assert "look_tail" not in engine._split_look_ids(sess)
    _arm_guess(sess, "lucy")
    guild.logodds = 3.0
    quiet.logodds = 1.6
    assert engine._advance(sess)["stage"] == "guessing"

    split = sess_mod.new_session()
    happy = Candidate(
        id="happy", name="Happy", series="Fairy Tail", logodds=1.6,
        tags=["male"], blurb="A blue cat with a tail.",
    )
    natsu = Candidate(
        id="natsu", name="Natsu Dragneel", series="Fairy Tail", logodds=3.0,
        tags=["male"], blurb="A fire mage in the Fairy Tail guild.",
    )
    split.candidates = [natsu, happy]
    _arm_guess(split, "natsu")
    state = engine._advance(split)
    assert state["stage"] == "asking"
    assert state["question"]["qid"] == "look_tail"


def test_near_twin_defers_once_per_pair():
    """The same leader and runner are not deferred twice; another pair still is."""
    sess = sess_mod.new_session()
    sess.candidates = [
        Candidate(id="heinkel", name="Heinkel", series="Fullmetal Alchemist",
                  logodds=3.0, popularity=3000, tags=["male"]),
        Candidate(id="ed", name="Edward Elric", series="Fullmetal Alchemist",
                  logodds=1.6, popularity=5000, tags=["male", "prosthetic", "eyepatch"],
                  blurb="An automail arm and an eyepatch."),
    ]
    _arm_guess(sess, "heinkel")
    first = engine._advance(sess)
    assert first["stage"] == "asking"
    assert first["question"]["qid"] == "look_prosthetic"
    assert tuple(sorted(("heinkel", "ed"))) in sess.near_twin_pairs
    sess.asked[-1]["answer"] = "no"
    second = engine._advance(sess)
    assert second["stage"] == "guessing"
    assert "look_eyepatch" not in {row["qid"] for row in sess.asked}

    other = sess_mod.new_session()
    other.near_twin_pairs.add(tuple(sorted(("heinkel", "ed"))))
    other.candidates = [
        Candidate(id="mihawk", name="Dracule Mihawk", series="One Piece",
                  logodds=3.0, popularity=4000, tags=["male", "eyepatch"]),
        Candidate(id="law", name="Trafalgar Law", series="One Piece",
                  logodds=1.6, popularity=5000, tags=["male"]),
    ]
    _arm_guess(other, "mihawk")
    state = engine._advance(other)
    assert state["stage"] == "asking"
    assert state["question"]["qid"] == "look_eyepatch"


def test_near_twin_rare_look_defers_the_guess():
    """A same-work pair that disagrees on a prosthetic is asked that, not guessed."""
    sess = sess_mod.new_session()
    sess.candidates = [
        Candidate(id="heinkel", name="Heinkel", series="Fullmetal Alchemist",
                  logodds=3.0, popularity=3000, tags=["male"]),
        Candidate(id="ed", name="Edward Elric", series="Fullmetal Alchemist",
                  logodds=1.6, popularity=5000, tags=["male", "prosthetic"],
                  blurb="An automail arm."),
    ]
    _arm_guess(sess, "heinkel")
    assert engine._should_guess(sess, None) is True
    state = engine._advance(sess)
    assert state["stage"] == "asking"
    assert state["question"]["qid"] == "look_prosthetic"


def test_near_twin_deferral_does_not_block_a_settled_or_capped_guess():
    """No rare split, a different work, or the turn cap still commits."""
    same = sess_mod.new_session()
    same.candidates = [
        Candidate(id="asuka", name="Asuka Langley Soryu", series="Neon Genesis Evangelion",
                  logodds=3.0, popularity=4000, tags=["female"]),
        Candidate(id="kyoko", name="Kyoko Zeppelin Soryu", series="Neon Genesis Evangelion",
                  logodds=1.6, popularity=2000, tags=["female"]),
    ]
    _arm_guess(same, "asuka")
    assert engine._advance(same)["stage"] == "guessing"

    other = sess_mod.new_session()
    other.candidates = [
        Candidate(id="heinkel", name="Heinkel", series="Fullmetal Alchemist",
                  logodds=3.0, popularity=3000, tags=["male"]),
        Candidate(id="ed", name="Edward Elric", series="Cowboy Bebop",
                  logodds=1.6, popularity=5000, tags=["male", "prosthetic"]),
    ]
    _arm_guess(other, "heinkel")
    assert engine._advance(other)["stage"] == "guessing"

    capped = sess_mod.new_session()
    capped.candidates = [
        Candidate(id="mihawk", name="Dracule Mihawk", series="One Piece",
                  logodds=3.0, popularity=4000, tags=["male", "eyepatch"]),
        Candidate(id="law", name="Trafalgar Law", series="One Piece",
                  logodds=1.6, popularity=5000, tags=["male"]),
    ]
    _arm_guess(capped, "mihawk")
    capped.turn = engine.MAX_TURNS
    assert engine._advance(capped)["stage"] == "guessing"
