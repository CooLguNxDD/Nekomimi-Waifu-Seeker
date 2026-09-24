"""Mika Misono should beat Trinity lookalikes on pink hair + halo + wings.

Laya is stubbed to a flat noul, which is what the live demo did: every
Blue Archive blurb mentions a halo, and several mention the word "wings"
because the halo is wing-shaped. The profile text has to break that tie.
"""

from __future__ import annotations

import pytest

from waifu_engine import web_search
from waifu_engine.nekomimi import engine, laya_client, session as sess_mod, traits

SEED = "pink hair + halo + wings"

MIKA = (
    "Misono Mika is a president of Trinity General School's Tea Party. "
    "Mika has long pink hair that turns into a pale blue near the bottom. She has golden eyes. "
    "On her back are a pair of angel wings that sprout from around waist level. "
    "Her halo is comprised of two pink spirals around a central sphere."
)
HANAKO = (
    "Urawa Hanako is a student at Trinity General School in the Supplementary Lessons Department. "
    "Hanako has thigh-length pink hair that fades into a lighter tone. She has green eyes and an ahoge. "
    "Hanako has a light pink halo which resembles two circumferences with four petal-shaped figures."
)
HANAE = (
    "Asagao Hanae is a member of the Knights Hospitaller at Trinity General School. "
    "Hanae has lavender hair tied into two long pigtails reaching her legs. She also has blue eyes. "
    "Hanae's halo is a pink heart with a hollowed cross inside another heart with wings."
)
MINE = (
    "Aomori Mine is the president of the Knights Hospitaller at Trinity General School. "
    "Mine has waist length, light blue hair. Several pink strands of hair can be seen in her hair. "
    "She has a pair of light blue, feathered, angel wings, sprouting just below her shoulders. "
    "Her yellow halo consists of a small circle with a cross and two hearts."
)
KOHARU = (
    "Shimoe Koharu studies at Trinity General School in the Supplementary Lessons Department. "
    "She has short pink hair, green eyes, and a small round halo above her head. "
    "She wears the school's uniform and carries a machine gun."
)
SERINA = (
    "Sumi Serina is a healer with the Knights Hospitaller at Trinity General School. "
    "She has light pink hair, blue eyes, and a halo above a nurse's cap. "
    "Her uniform is white and pink. She treats the wounded after battles."
)

POOL = [
    {"id": "mika", "name": "Mika Misono", "series": "Blue Archive", "medium": "game",
     "blurb": MIKA, "popularity": 100, "tags": ["game", "female", "pink"]},
    {"id": "hanako", "name": "Hanako Urawa", "series": "Blue Archive", "medium": "game",
     "blurb": HANAKO, "popularity": 80000, "tags": ["game", "female", "pink", "halo"]},
    {"id": "hanae", "name": "Hanae Asagao", "series": "Blue Archive", "medium": "game",
     "blurb": HANAE, "popularity": 50000, "tags": ["game", "female"]},
    {"id": "mine", "name": "Mine Aomori", "series": "Blue Archive", "medium": "game",
     "blurb": MINE, "popularity": 1_000_000, "tags": ["game", "female", "wings", "halo"]},
    {"id": "koharu", "name": "Koharu Shimoe", "series": "Blue Archive", "medium": "game",
     "blurb": KOHARU, "popularity": 200000, "tags": ["game", "female", "pink", "halo"]},
    {"id": "serina", "name": "Serina Sumi", "series": "Blue Archive", "medium": "game",
     "blurb": SERINA, "popularity": 60000, "tags": ["game", "female", "pink", "halo"]},
]


@pytest.fixture(autouse=True)
def _flat_laya(monkeypatch):
    """Every candidate gets the same noul. Lexical evidence has to separate them."""
    def ask(state, questions):
        if "match" not in questions:
            return None
        return {"match": {"noul": 0.62, "action": {"act_probability": 0.2}}}

    monkeypatch.setattr(laya_client, "ask", ask)
    monkeypatch.setattr(engine.sources, "find_candidates", lambda *a, **k: [])
    monkeypatch.setattr(engine, "ONLINE", True)


def _pool_session() -> sess_mod.GuessSession:
    s = sess_mod.new_session(SEED)
    s.add_candidates(POOL)
    return s


def test_halo_wings_and_horns_are_separate_questions():
    halo = traits.QUESTIONS_BY_ID["look_halo"]
    wings = traits.QUESTIONS_BY_ID["look_wings"]
    horns = traits.QUESTIONS_BY_ID["look_horns"]
    assert halo["text"] == "Does your character have a halo?"
    assert wings["text"] == "Does your character have wings?"
    assert horns["text"] == "Does your character have horns?"
    assert halo["tags_true"] == ["halo"]
    assert wings["tags_true"] == ["wings"]
    assert horns["tags_true"] == ["horns"]
    assert len({halo["category"], wings["category"], horns["category"], "look"}) == 4
    assert "wing-shaped halo" in wings["criteria"]["true"]
    mined = {slug for slug, _ in web_search.TRAIT_PATTERNS}
    assert {"halo", "wings", "horns"} <= mined
    for gone in ("horns or wings", "horns, wings or a halo"):
        assert all(gone not in q["text"] and gone not in q["instructions"]
                   for q in traits.QUESTION_BANK)


def test_a_yes_to_halo_does_not_settle_wings():
    s = _pool_session()
    s.asked.append({"qid": "look_halo", "text": "?", "category": "look_halo", "answer": "yes"})
    assert "look_wings" not in s.settled_categories()
    ids = {q["id"] for q in engine.candidate_questions(s)}
    assert "look_wings" in ids
    # Nobody in the Trinity pool has horns, so that question has nothing to
    # split until a horned character shows up. Halo's yes must not hide it.
    s.add_candidates([{
        "id": "albedo",
        "name": "Albedo",
        "series": "Overlord",
        "medium": "anime",
        "blurb": (
            "She has long black hair, golden eyes, black horns, and fallen angel wings. "
            "The horns and wings are part of her body, not of a halo."
        ),
        "tags": ["horns", "wings"],
    }])
    ids = {q["id"] for q in engine.candidate_questions(s)}
    assert "look_horns" in ids


def test_body_wings_are_not_a_wing_shaped_halo():
    assert traits.has_body_wings(MIKA)
    assert traits.has_body_wings(MINE)
    assert not traits.has_body_wings(HANAE)
    assert not traits.has_body_wings(HANAKO)
    assert traits.has_body_wings("She has large white wings on her back.")
    assert traits.has_body_wings("She has a halo and broad white wings on her back.")
    assert not traits.has_body_wings("Her halo has small white wings around a ring.")
    assert "wings" in web_search.mine_trait_slugs(MIKA)
    assert "wings" not in web_search.mine_trait_slugs(HANAE)
    assert "halo" in web_search.mine_trait_slugs(HANAE)
    assert "pink" in web_search.mine_trait_slugs(HANAKO)
    assert "pink" not in web_search.mine_trait_slugs(MINE)


def test_clue_requires_the_full_set_and_ignores_a_thin_blurb():
    full = traits.clue_likelihood(SEED, MIKA, ["pink"])
    partial = traits.clue_likelihood(SEED, HANAKO, ["pink", "halo"])
    decoy = traits.clue_likelihood(SEED, HANAE, [])
    winged = traits.clue_likelihood(SEED, MINE, ["wings", "halo"])
    assert full == pytest.approx(0.96)
    assert partial < 0.2
    assert decoy < partial
    assert winged < 0.2
    assert traits.clue_likelihood(SEED, "A student at Trinity.", []) is None
    assert traits.clue_likelihood("Italian plumber", MIKA, []) is None


def test_flat_noul_still_ranks_mika_over_richer_lookalikes():
    s = _pool_session()
    engine.score_candidates(s, traits.clue_question("clue_seed", SEED), "yes")
    ranked = [c.id for c, _ in s.posterior()]
    assert ranked[0] == "mika"
    mika_p = s.posterior()[0][1]
    second = s.posterior()[1][1]
    assert mika_p > second * 2
    assert all(c.alive for c in s.candidates)
    # Mine is the famous one with real wings and a wings tag. Fame must not win.
    assert s.by_id("mika").logodds > s.by_id("mine").logodds + 1.0


def test_wings_answer_demotes_pink_hair_without_body_wings():
    s = _pool_session()
    engine.score_candidates(s, traits.QUESTIONS_BY_ID["look_wings"], "yes")
    order = [c.id for c, _ in s.posterior()]
    assert order.index("mika") < order.index("hanako")
    assert order.index("mine") < order.index("hanae")
    assert s.by_id("hanako").alive


def test_seed_traits_are_asked_ahead_of_shared_school_questions():
    s = sess_mod.new_session(SEED)
    s.add_candidates([
        {"id": f"c{i}", "name": f"Student {i}", "series": "Blue Archive" if i < 4 else "Other Game",
         "medium": "game",
         "blurb": "A student at the academy." if i % 2 == 0 else "A hero of the story.",
         "tags": ["student", "uniform", "teen"] if i % 2 == 0 else ["teen"]}
        for i in range(6)
    ])
    ids = [q["id"] for q in engine.candidate_questions(s)]
    for qid in ("hair_color", "look_halo", "look_wings"):
        assert qid in ids
        assert ids.index(qid) < ids.index("look_uniform")
    s.asked.append({
        "qid": "series", "text": "Which series is your character from?", "category": "series",
        "answer": "s1", "kind": "choice",
        "options": {"s1": {"label": "Blue Archive", "fact": "Blue Archive",
                           "series_key": "bluearchive", "prior": 0.0, "tags": []}},
    })
    ids = [q["id"] for q in engine.candidate_questions(s)]
    assert ids[0] in {"hair_color", "look_halo", "look_wings", "look_horns"}
    assert "look_horns" in ids


def test_initial_trait_seed_asks_for_inline_gemini(monkeypatch):
    seen = []
    monkeypatch.setattr(engine.sources, "find_candidates",
                        lambda terms, **k: seen.append(k) or [])
    engine.start(SEED)
    assert seen[0]["gemini_inline"] is True
    s = _pool_session()
    seen.clear()
    engine.refresh_candidates(s)
    assert seen[0]["gemini_inline"] is False


def test_a_negated_trait_is_required_to_be_absent():
    mika = traits.clue_likelihood("pink hair and no wings", MIKA, ["pink"])
    koharu = traits.clue_likelihood("pink hair and no wings", KOHARU, ["pink"])
    assert mika is not None and mika < 0.2
    assert koharu is not None and koharu >= 0.9
    assert traits.clue_likelihood("without a halo", HANAKO, ["halo"]) <= 0.22
    plain = (
        "She has long black hair and dark eyes, and she wears a school uniform "
        "with a ribbon. Her outfit is described down to the shoes and the bag."
    )
    assert traits.clue_likelihood("without a halo", plain, []) >= 0.8
    assert traits.clue_likelihood(SEED, MIKA, ["pink"]) >= 0.9
    assert traits.visual_search_phrases("pink hair and no wings") == ["pink hair"]


def test_seed_plus_signs_become_search_words():
    s = sess_mod.new_session(SEED)
    assert engine._search_terms(s) == ["pink hair halo wings"]


def test_confirmed_series_stays_with_the_visual_traits(monkeypatch):
    s = _pool_session()
    s.asked.append({
        "qid": "series",
        "text": "Which series is your character from?",
        "category": "series",
        "answer": "s1",
        "kind": "choice",
        "options": {
            "s1": {"label": "Blue Archive", "fact": "Blue Archive", "series_key": "bluearchive",
                   "prior": 0.0, "tags": []},
            "other": {"label": "Another series", "fact": "", "prior": 0.3, "tags": []},
        },
        "detail": None,
    })
    # Enough later answers that a middle "Blue Archive" would fall out of the
    # template window (first fact + last three) without the anchor.
    for i, text in enumerate(
        ["Is your character female?", "Is your character a student?",
         "Is your character notably tall?", "Does your character wear a school uniform?"],
        start=1,
    ):
        s.asked.append({"qid": f"q{i}", "text": text, "category": "filler", "prior": 0.4,
                        "answer": "yes", "detail": None})
    seen = {}

    def search(terms, **kwargs):
        seen["terms"] = terms
        seen["focus"] = kwargs.get("focus")
        return []

    monkeypatch.setattr(engine.sources, "find_candidates", search)
    engine.refresh_candidates(s)
    anchor = seen["terms"][0]
    assert anchor.startswith("Blue Archive")
    for phrase in ("pink hair", "halo", "wings"):
        assert phrase in anchor
    assert seen["focus"][0] == anchor


def test_series_fact_stays_inside_the_focus_window():
    entries = [("pink hair halo wings", 0.0), ("Blue Archive", 0.0)]
    entries += [(f"broad fact {i}", 0.5 + i / 100) for i in range(10)]
    offered = [text for text, _rank in engine._focus_offer(entries)]
    assert "Blue Archive" in offered
    assert "pink hair halo wings" in offered
