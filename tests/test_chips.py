"""Soft-chip table: polarity, residual text, and the snapshot label."""

from waifu_engine.nekomimi import chips, engine, session as sess_mod
from waifu_engine.nekomimi.traits import CHIP_HIT, chip_likelihood, free_text_trait_hits


def test_negated_royalty_is_a_no_and_leaves_the_dress():
    """"not a princess in a white dress" must not score royalty as a yes."""
    hits = dict(chips.free_text_trait_hits("not a princess in a white dress"))
    assert hits["job_royalty"] == "no"
    assert "princess" not in chips.chip_residual("not a princess in a white dress")
    assert "white dress" in chips.chip_residual("not a princess in a white dress")


def test_angel_chip_stays_soft_when_the_blurb_never_says_it():
    """A miss stays at one half. The old noul of 0 floored the rest of the pool."""
    assert chips.free_text_trait_ids("she is an angel") == ["species_angel"]
    assert chip_likelihood("species_angel", "a knight with a sword", []) == 0.5
    assert chip_likelihood("species_angel", "an angel with a halo", []) == CHIP_HIT
    assert CHIP_HIT >= 0.90


def test_traits_reexports_the_chip_table():
    """Call sites that still import the hits helper from traits see the same pairs."""
    assert free_text_trait_hits("demon king") == chips.free_text_trait_hits("demon king")


def test_state_snapshot_uses_the_choice_label():
    """A refresh must rebuild the chip from the option label, not the key."""
    sess = sess_mod.new_session("silver hair")
    sess.stage = "done"
    sess.asked.append(
        {
            "qid": "series",
            "text": "Which series is your character from?",
            "answer": "eva",
            "detail": None,
            "options": {"eva": {"label": "Neon Genesis Evangelion"}},
        }
    )
    snap = engine.state_payload(sess)
    assert snap["asked"][0]["label"] == "Neon Genesis Evangelion"
    assert snap["asked"][0]["answer"] == "eva"
