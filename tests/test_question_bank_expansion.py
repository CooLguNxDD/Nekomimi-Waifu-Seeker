"""New bank rows stay unique, narrow, and wired to mined trait slugs."""

from __future__ import annotations

import json

from importlib.resources import files

from waifu_engine import web_search
from waifu_engine.nekomimi import engine, session as sess_mod, traits
from waifu_engine.nekomimi.lexicon import TRAIT_PATTERNS

NEW_QIDS = (
    "look_eyepatch",
    "look_cape",
    "look_animal_ears",
    "look_tail",
    "look_prosthetic",
    "weapon_staff",
    "weapon_spear",
    "bond_sibling",
    "bond_orphan",
    "bond_twin",
    "team_hero",
    "team_military",
    "species_elf",
    "species_dragon",
    "trope_isekai",
    "trope_amnesia",
    "trope_chosen",
    "voice_silent",
    "voice_formal",
    "prop_instrument",
    "role_vigilante",
)


def test_new_questions_are_unique_yesno_rows():
    """Each new qid expands once, with a short instruction and a prior."""
    ids = [q["id"] for q in traits.QUESTION_BANK]
    assert len(ids) == len(set(ids))
    raw = json.loads(
        files("waifu_engine.nekomimi").joinpath("question_bank.json").read_text(encoding="utf-8")
    )
    raw_ids = [row.get("id") for row in raw["questions"] if row.get("id")]
    assert len(raw_ids) == len(set(raw_ids))
    for qid in NEW_QIDS:
        question = traits.QUESTIONS_BY_ID[qid]
        assert question["kind"] == "yesno"
        assert question["id"] not in {"hair_color", "eye_color", "medium"}
        assert "`candidate`" in question["instructions"]
        assert len(question["instructions"]) < 120
        assert question["criteria"]["true"].startswith("yes, this is true")
        assert 0 < question["prior"] < 0.5


def test_new_tags_are_mined_from_blurbs():
    """A lexicon marker for a new tag has to be the bank's tags_true slug."""
    mined = dict(TRAIT_PATTERNS)
    for qid in NEW_QIDS:
        for tag in traits.QUESTIONS_BY_ID[qid]["tags_true"]:
            assert tag in mined
    assert "elf" in web_search.mine_trait_slugs("She is an elf archer.")
    assert "animal-ears" in web_search.mine_trait_slugs("A girl with cat ears.")
    assert "twin" not in web_search.mine_trait_slugs("She wears twintails.")
    assert "tail" not in web_search.mine_trait_slugs("She wears a ponytail.")
    assert "isekai-travel" in web_search.mine_trait_slugs("An isekai hero.")


def test_elf_tag_moves_the_pool():
    """A yes on the elf question ranks the tagged elf above a human peer."""
    sess = sess_mod.new_session()
    sess.add_candidates([
        {"id": "elf", "name": "Lyra", "series": "Grove", "medium": "game",
         "tags": ["elf", "female"], "blurb": "An elf archer of the grove.", "popularity": 10},
        {"id": "human", "name": "Mara", "series": "Grove", "medium": "game",
         "tags": ["human", "female"], "blurb": "A human knight of the grove.", "popularity": 10},
    ])
    engine.score_candidates(sess, traits.QUESTIONS_BY_ID["species_elf"], "yes")
    ranked = [c.id for c, _p in sess.posterior()]
    assert ranked[0] == "elf"
