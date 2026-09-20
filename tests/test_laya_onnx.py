"""Offline tests for the DirectML/ONNX Laya path.

No weights, no ORT session, no network. Decoding and padding are pure functions.
"""

from __future__ import annotations

import numpy as np

from waifu_engine.nekomimi.laya_onnx import KMAX, SEQ_LEN, decode_answers, pad_collate


def test_decode_noul_yes():
    questions = {"match": {"type": "noul", "instructions": "Is the candidate female?"}}
    items = [{"markers": [1, 2], "qtype": 2}]
    logits = np.array([[0.0, 4.0] + [-1e4] * (KMAX - 2)], dtype=np.float32)
    act = np.array([[2.0, 0.1]], dtype=np.float32)
    answers = decode_answers(questions, items, logits, act, [1.0, 1.0, 1.0], {})
    assert answers["match"]["type"] == "noul"
    assert answers["match"]["noul"] > 0.9
    assert "act_probability" in answers["match"]["action"]


def test_decode_choice_uses_probabilities_key():
    questions = {
        "next_question": {
            "type": "choice",
            "instructions": "Which trait?",
            "criteria": {"gender_female": "female", "gender_male": "male"},
        }
    }
    items = [{"markers": [1, 2], "qtype": 0}]
    logits = np.array([[3.0, 0.1]], dtype=np.float32)
    act = np.array([[1.0, 0.0]], dtype=np.float32)
    answers = decode_answers(questions, items, logits, act, [1.0, 1.0, 1.0], {})
    out = answers["next_question"]
    assert out["type"] == "choice"
    assert out["choice"] == "gender_female"
    assert "probabilities" in out
    assert "probs" not in out
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-3


def test_decode_score_expected_ordinal():
    questions = {
        "fit": {
            "type": "score",
            "instructions": "How strong is the match?",
            "criteria": ["weak", "decent", "excellent"],
        }
    }
    items = [{"markers": [1, 2, 3], "qtype": 1}]
    logits = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    act = np.array([[0.5, 0.5]], dtype=np.float32)
    answers = decode_answers(questions, items, logits, act, [1.0, 1.0, 1.0], {})
    assert answers["fit"]["score"] > 1.5
    assert set(answers["fit"]["probabilities"]) == {"0", "1", "2"}


def test_pad_collate_fixed_ranks():
    n, l, k = 3, 40, 4
    batch = {
        "input_ids": np.ones((n, l), dtype=np.int64),
        "attention_mask": np.ones((n, l), dtype=np.int64),
        "marker_pos": np.arange(k, dtype=np.int64)[None, :].repeat(n, axis=0),
        "marker_mask": np.ones((n, k), dtype=np.int64),
        "qtype": np.array([2, 0, 1], dtype=np.int64),
    }
    padded = pad_collate(batch, pad_id=0, seq_len=SEQ_LEN, kmax=KMAX)
    assert padded["input_ids"].shape == (n, SEQ_LEN)
    assert padded["attention_mask"].shape == (n, SEQ_LEN)
    assert padded["marker_pos"].shape == (n, KMAX)
    assert padded["marker_mask"].shape == (n, KMAX)
    assert padded["qtype"].shape == (n,)
    assert padded["input_ids"][:, :l].sum() == n * l
    assert padded["input_ids"][:, l:].sum() == 0
    assert padded["marker_mask"][:, :k].sum() == n * k
    assert padded["marker_mask"][:, k:].sum() == 0
    auto = pad_collate(batch, pad_id=0)
    assert auto["input_ids"].shape[1] % 8 == 0
    assert auto["input_ids"].shape[1] >= l
    assert auto["marker_pos"].shape[1] >= k


def test_provider_torch_skips_onnx(monkeypatch):
    from waifu_engine.nekomimi import laya_client

    monkeypatch.setenv("WAIFU_LAYA_PROVIDER", "torch")
    assert laya_client._try_onnx() is None


def test_backend_none_after_reset():
    from waifu_engine.nekomimi import laya_client

    laya_client.reset()
    assert laya_client.backend() == "none"
