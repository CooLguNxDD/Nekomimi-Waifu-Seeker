"""A profile that never mentions a trait is scored near its base rate, not by Laya's drift."""

from __future__ import annotations

import math

import pytest

from waifu_engine.nekomimi import engine, laya_client, session as sess_mod
from waifu_engine.nekomimi.session import Candidate
from waifu_engine.nekomimi.traits import QUESTIONS_BY_ID, series_question

HALO = QUESTIONS_BY_ID["look_halo"]
STOIC = QUESTIONS_BY_ID["pers_stoic"]
HAIR = QUESTIONS_BY_ID["hair_color"]


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """Pin the documented defaults so an env override cannot move the numbers."""
    monkeypatch.setattr(engine, "SILENT_WEIGHT", 0.3)
    monkeypatch.setattr(engine, "CHOICE_SILENT_MAX", 0.5)


def _cand(cid: str, blurb: str = "", tags: list[str] | None = None, **kw) -> Candidate:
    return Candidate(id=cid, name=cid.title(), blurb=blurb, tags=tags or [], **kw)


def test_silent_noul_is_pulled_toward_the_prior_and_a_grounded_one_is_kept():
    silent = _cand("tifa", "A martial artist who runs a bar.")
    stated = _cand("kuu", "A stoic, quiet swordswoman.")
    tagged = _cand("rei", "A pilot.", tags=["stoic"])
    # Measured: Laya put 0.31 on "stoic" for Tifa's real page (prior 0.2).
    assert engine._calibrated_noul(STOIC, silent, 0.31) == pytest.approx(0.2 + 0.3 * 0.11)
    assert engine._calibrated_noul(STOIC, stated, 0.75) == 0.75
    assert engine._calibrated_noul(STOIC, tagged, 0.75) == 0.75


def test_a_no_halo_answer_no_longer_sinks_a_silent_target(monkeypatch):
    """Halo scored 0.63-0.73 on real pages that never mention one."""
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: {"match": {"noul": 0.7}})
    sess = sess_mod.new_session()
    sess.add_candidates([{"id": "tifa", "name": "Tifa", "blurb": "A martial artist."}])
    engine.score_candidates(sess, HALO, "no")
    halo = next(r for r in engine.evidence_breakdown(sess, "tifa") if r["qid"] == "look_halo")
    assert halo["logodds"] == pytest.approx(math.log(1 - (0.05 + 0.3 * 0.65)), abs=1e-3)
    assert halo["logodds"] > -0.3  # raw 0.7 would have cost log(0.3) = -1.2


def test_a_flat_choice_uses_base_rates_and_a_confident_one_is_kept(monkeypatch):
    flat = {"blonde": 0.1, "black": 0.06, "brown": 0.18, "white": 0.16,
            "red": 0.1, "blue": 0.1, "pink": 0.19, "other": 0.11}
    sharp = {k: 0.01 for k in flat} | {"red": 0.93}
    sess = sess_mod.new_session()
    sess.add_candidates([
        {"id": "tifa", "name": "Tifa", "blurb": "A martial artist."},
        {"id": "asuka", "name": "Asuka", "blurb": "A pilot with long red hair."},
    ])
    sess.choice_cache[("tifa", "hair_color")] = flat
    sess.choice_cache[("asuka", "hair_color")] = sharp
    monkeypatch.setattr(laya_client, "ask", lambda state, questions: None)
    engine.score_candidates(sess, HAIR, "black")
    rows = {cid: next(r["logodds"] for r in engine.evidence_breakdown(sess, cid)
                      if r["qid"] == "hair_color") for cid in ("tifa", "asuka")}
    base = engine._silent_choice(HAIR, sess.by_id("tifa"))
    assert base["black"] == pytest.approx(0.2)  # the bank's base rate for black hair
    assert rows["tifa"] == pytest.approx(math.log(0.2), abs=1e-3)
    assert rows["asuka"] == pytest.approx(math.log(0.02), abs=1e-3)  # sharp, kept (clamped)
    assert sess.choice_cache[("tifa", "hair_color")] == flat  # cache stays raw


def test_silent_choice_keeps_the_tag_nudge():
    blue = engine._silent_choice(HAIR, _cand("rem", tags=["blue"]))
    plain = engine._silent_choice(HAIR, _cand("x"))
    assert blue["blue"] > plain["blue"]
    assert sum(blue.values()) == pytest.approx(1.0)


def _series() -> dict:
    return series_question("series", [("rezero", "Re:Zero"), ("hunterxhunter", "Hunter x Hunter")])


@pytest.mark.parametrize("series", ["", "Unknown", "Web result"])
def test_unknown_series_leans_to_another_series(series):
    dist = engine._known_choice(_series(), _cand("megumin", "An archwizard.", series=series))
    assert dist["other"] == pytest.approx(0.6)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_unknown_series_still_takes_a_listed_work_its_page_names():
    cand = _cand("ram", "A maid at the Roswaal mansion in Re:Zero.", series="Web result")
    dist = engine._known_choice(_series(), cand)
    assert dist["s1"] == pytest.approx(0.9)


def test_known_series_is_unchanged():
    dist = engine._known_choice(_series(), _cand("gon", series="Hunter x Hunter"))
    assert dist["s2"] == pytest.approx(0.9)


def test_lookahead_uses_the_same_calibration():
    sess = sess_mod.new_session()
    cand = _cand("tifa", "A martial artist.")
    rows = ()
    sess.lookahead[("tifa", "pers_stoic")] = (rows, 0.8)
    assert engine._lookahead_likelihood(sess, STOIC, cand, rows) == pytest.approx(
        engine._calibrated_noul(STOIC, cand, 0.8))
