"""ONNX Runtime + DirectML backend for Laya.

Laya's PyTorch path only auto-selects CUDA/MPS. On Windows AMD the workable
GPU bridge is: export ``DecisionModel`` once, then run it with ONNX Runtime's
DirectML execution provider. Tokenization and answer decoding stay on CPU.

Shapes are padded to a fixed (seq=512, kmax=16) so DirectML compiles one graph.
Batch stays dynamic (1-10 questions per turn).

This module never raises out of ``load_onnx_agent`` / ``export_onnx`` callers
that catch Exception; the CLI prints failures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional extra
    ort = None

SEQ_LEN = 512
KMAX = 16
ONNX_NAME = "laya-decision.onnx"
META_NAME = "laya-decision.meta.json"

_YES = {"1", "true", "yes"}


def onnx_path() -> Path:
    raw = os.getenv("WAIFU_LAYA_ONNX_PATH")
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2] / "data" / ONNX_NAME


def meta_path(model_path: Path | None = None) -> Path:
    p = model_path or onnx_path()
    if p.name == ONNX_NAME:
        return p.with_name(META_NAME)
    return p.with_suffix(".meta.json")


def _dml_device_id() -> int:
    return int(os.getenv("WAIFU_LAYA_DML_DEVICE", "0"))


def ort_available() -> bool:
    return ort is not None


def dml_available() -> bool:
    return bool(ort) and "DmlExecutionProvider" in ort.get_available_providers()


def decode_answers(
    questions: dict[str, dict[str, Any]],
    items: list[dict[str, Any]],
    logits: np.ndarray,
    act_logits: np.ndarray,
    temperature: list[float] | tuple[float, ...],
    temperature_by_options: dict[str, float],
) -> dict[str, Any]:
    """Pack Laya-shaped answers from a batched forward pass.

    Mirrors ``laya.agent.Agent.system_one`` decoding so DirectML and torch
    produce the same keys (``probabilities``, not ``probs``).
    """
    from laya.agent import Agent
    from laya.common import QTYPES, confidence_from_probs, temp_bucket

    ids = list(questions.keys())
    act = _softmax(act_logits.astype(np.float64))
    answers: dict[str, Any] = {}
    for r, qid in enumerate(ids):
        q = Agent._to_internal(questions[qid])
        k = len(items[r]["markers"])
        qt = QTYPES[q["t"]]
        t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        z = logits[r, :k].astype(np.float64) / max(1e-3, float(t_scale))
        p = np.exp(z - z.max())
        p = p / p.sum()
        conf_score = round(confidence_from_probs(p, k), 4)
        ext = {"act_probability": round(float(act[r, 0]), 4)}
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            answers[qid] = {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": conf_score,
                "action": ext,
            }
        elif q["t"] == "score":
            exp_score = float((np.arange(k) * p).sum())
            answers[qid] = {
                "type": "score",
                "score": round(exp_score, 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf_score,
                "action": ext,
            }
        else:
            answers[qid] = {
                "type": "noul",
                "noul": round(float(p[1]), 4),
                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                "action": ext,
            }
    return answers


def _round_up(n: int, multiple: int) -> int:
    return max(multiple, ((int(n) + multiple - 1) // multiple) * multiple)


def pad_collate(
    batch: dict[str, Any],
    pad_id: int,
    seq_len: int | None = None,
    kmax: int | None = None,
) -> dict[str, np.ndarray]:
    """Pad a ``collate_items`` batch to ONNX ranks.

    ``seq_len``/``kmax`` None pads to the batch max (seq rounded up to 8).
    """
    n = int(batch["input_ids"].shape[0])
    src_ids = _as_numpy(batch["input_ids"])
    src_att = _as_numpy(batch["attention_mask"])
    src_pos = _as_numpy(batch["marker_pos"])
    src_mask = _as_numpy(batch["marker_mask"])
    actual_l = int(src_ids.shape[1])
    actual_k = int(src_pos.shape[1])
    if seq_len is None:
        seq_len = _round_up(actual_l, 8)
    if kmax is None:
        kmax = max(actual_k, 2)
    l = min(actual_l, seq_len)
    k = min(actual_k, kmax)
    ids = np.full((n, seq_len), pad_id, dtype=np.int64)
    att = np.zeros((n, seq_len), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=np.int64)
    ids[:, :l] = src_ids[:, :l]
    att[:, :l] = src_att[:, :l]
    mpos[:, :k] = src_pos[:, :k]
    mmask[:, :k] = src_mask[:, :k].astype(np.int64)
    qtype = _as_numpy(batch["qtype"]).astype(np.int64).reshape(n)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": qtype,
    }


class OnnxAgent:
    """Duck-typed Laya Agent: ``predict`` + ``cfg``, runs on ORT."""

    def __init__(
        self,
        tok: Any,
        cfg: dict[str, Any],
        session: Any,
        provider: str,
        seq_len: int,
        kmax: int,
        dynamic_seq: bool = False,
    ):
        self.tok = tok
        self.cfg = cfg
        self.session = session
        self.provider = provider
        self.backend = "dml" if provider == "dml" else "onnx-cpu"
        self.device = provider
        self.seq_len = seq_len
        self.kmax = kmax
        self.dynamic_seq = dynamic_seq
        self.temperature = cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = cfg.get("temperature_by_options", {})
        self._input_names = [i.name for i in session.get_inputs()]
        self._output_names = [o.name for o in session.get_outputs()]

    def predict(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from laya.agent import Agent
        from laya.common import QTYPES, build_sequence, collate_items, render_options

        ids = list(questions.keys())
        items: list[dict[str, Any]] = []
        max_len = int(self.cfg.get("max_len", self.seq_len))
        head_max_len = int(self.cfg.get("head_max_len", 192))
        for qid in ids:
            q = Agent._to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            if len(markers) > self.kmax:
                raise ValueError("question %r has %d options; onnx kmax is %d" % (qid, len(markers), self.kmax))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        b = collate_items([items], self.tok.pad_token_id)
        feeds = pad_collate(b, self.tok.pad_token_id, self.seq_len, self.kmax)
        ort_inputs = {name: feeds[name] for name in self._input_names}
        logits, act_logits = self.session.run(self._output_names, ort_inputs)
        answers = decode_answers(
            questions, items, logits, act_logits, self.temperature, self.temperature_by_options
        )
        n_tokens = int(feeds["attention_mask"].sum())
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": n_tokens, "output_tokens": 0},
        }

    system_one = predict


def load_onnx_agent(path: Path | None = None, prefer_dml: bool = True) -> OnnxAgent | None:
    """Load tokenizer + ORT session. Returns None if the file or runtime is missing."""
    if ort is None:
        return None
    path = path or onnx_path()
    if not path.is_file():
        return None
    meta = _read_meta(path)
    # A graph exported from a different checkpoint has different weights and
    # different temperatures; silently reusing it would be wrong, not just slow.
    if not _meta_matches_checkpoint(meta):
        return None
    session, provider = _make_session(path, prefer_dml=prefer_dml)
    if session is None:
        return None
    tok, cfg = _load_tokenizer_cfg()
    return OnnxAgent(
        tok,
        cfg,
        session,
        provider,
        seq_len=int(meta.get("seq_len", SEQ_LEN)),
        kmax=int(meta.get("kmax", KMAX)),
        dynamic_seq=bool(meta.get("dynamic_seq", False)),
    )


def export_onnx(path: Path | None = None, agent: Any | None = None) -> Path:
    """Load the torch agent if needed, export DecisionModel, write ``path``."""
    import torch

    path = path or onnx_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if agent is None:
        import laya
        from .laya_client import HEAD_MAX_LEN, MAX_LEN, MODEL_ID, SUBFOLDER

        agent = laya.load(MODEL_ID, subfolder=SUBFOLDER or None)
        agent.cfg["head_max_len"] = HEAD_MAX_LEN
        agent.cfg["max_len"] = MAX_LEN

    seq_len = int(agent.cfg.get("max_len", SEQ_LEN))
    _prepare_encoder(agent.model)
    wrapper = _ExportWrapper(agent.model)
    wrapper.eval()

    pad_id = int(getattr(agent.tok, "pad_token_id", 0) or 0)
    dummy = _dummy_inputs(pad_id, seq_len=seq_len, kmax=KMAX, vocab=int(getattr(agent.tok, "vocab_size", 1000) or 1000))

    with torch.no_grad():
        _ = wrapper(*dummy)

    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    _torch_onnx_export(wrapper, dummy, tmp)
    if path.exists():
        path.unlink()
    tmp.replace(path)
    _write_meta(path, seq_len=seq_len, kmax=KMAX)
    return path


def verify_onnx(path: Path | None = None, agent: Any | None = None, prefer_dml: bool = True) -> dict[str, Any]:
    """Compare one torch forward to ORT. Used by the CLI, not by request handlers."""
    import time

    import torch

    path = path or onnx_path()
    if agent is None:
        import laya
        from .laya_client import HEAD_MAX_LEN, MAX_LEN, MODEL_ID, SUBFOLDER

        agent = laya.load(MODEL_ID, subfolder=SUBFOLDER or None)
        agent.cfg["head_max_len"] = HEAD_MAX_LEN
        agent.cfg["max_len"] = MAX_LEN

    _prepare_encoder(agent.model)
    seq_len = int(agent.cfg.get("max_len", SEQ_LEN))
    pad_id = int(getattr(agent.tok, "pad_token_id", 0) or 0)
    dummy = _dummy_inputs(pad_id, seq_len=seq_len, kmax=KMAX, vocab=int(getattr(agent.tok, "vocab_size", 1000) or 1000))
    wrapper = _ExportWrapper(agent.model).eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        torch_logits, torch_act = wrapper(*dummy)
        torch_ms = (time.perf_counter() - t0) * 1000
    session, provider = _make_session(path, prefer_dml=prefer_dml)
    if session is None:
        raise RuntimeError("could not create ORT session for %s" % path)
    feeds = {
        "input_ids": dummy[0].numpy(),
        "attention_mask": dummy[1].numpy(),
        "marker_pos": dummy[2].numpy(),
        "marker_mask": dummy[3].numpy(),
        "qtype": dummy[4].numpy(),
    }
    out_names = [o.name for o in session.get_outputs()]
    session.run(out_names, feeds)  # warmup (DML graph compile)
    t0 = time.perf_counter()
    ort_logits, ort_act = session.run(out_names, feeds)
    ort_ms = (time.perf_counter() - t0) * 1000
    logits_max = float(np.max(np.abs(torch_logits.numpy() - ort_logits)))
    act_max = float(np.max(np.abs(torch_act.numpy() - ort_act)))
    return {
        "provider": provider,
        "ort_providers": session.get_providers(),
        "torch_ms": round(torch_ms, 2),
        "ort_ms": round(ort_ms, 2),
        "logits_max_abs": logits_max,
        "act_max_abs": act_max,
        "path": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
    }


# --- internals -------------------------------------------------------


class _ExportWrapper:
    """Keep DecisionModel.forward but accept int64 marker_mask for ORT/DML."""

    def __init__(self, model: Any):
        self._mod = _module(model)

    def eval(self):
        self._mod.eval()
        return self

    def __call__(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self._mod(input_ids, attention_mask, marker_pos, marker_mask, qtype)


def _module(model: Any):
    import torch
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
            mm = marker_mask.bool() if marker_mask.dtype != torch.bool else marker_mask
            return self.inner(input_ids, attention_mask, marker_pos, mm, qtype)

    return Wrapper(model)


def _prepare_encoder(model: Any) -> None:
    enc = getattr(model, "encoder", None)
    if enc is None:
        return
    cfg = getattr(enc, "config", None)
    if cfg is not None:
        try:
            cfg.reference_compile = False
        except Exception:
            pass
        try:
            cfg._attn_implementation = "eager"
        except Exception:
            pass
    if hasattr(enc, "set_attn_implementation"):
        try:
            enc.set_attn_implementation("eager")
        except Exception:
            pass


def _dummy_inputs(pad_id: int, seq_len: int, kmax: int, vocab: int, batch: int = 2):
    import torch

    vocab = max(vocab, pad_id + 2)
    ids = torch.randint(0, vocab, (batch, seq_len), dtype=torch.long)
    ids[:, seq_len // 2 :] = pad_id
    att = torch.ones(batch, seq_len, dtype=torch.long)
    att[:, seq_len // 2 :] = 0
    mpos = torch.zeros(batch, kmax, dtype=torch.long)
    mpos[:, 0] = 1
    mpos[:, 1] = 5
    mmask = torch.zeros(batch, kmax, dtype=torch.long)
    mmask[:, :2] = 1
    qtype = torch.tensor([2, 0], dtype=torch.long)
    return ids, att, mpos, mmask, qtype


def _torch_onnx_export(wrapper: Any, dummy: tuple, tmp: Path) -> None:
    import inspect

    import torch

    names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    # Seq and k are fixed (512 / 16). DirectML Reshape breaks if those ranks
    # change; only batch stays dynamic (1-10 questions per turn).
    kwargs: dict[str, Any] = {
        "input_names": names,
        "output_names": ["logits", "act_logits"],
        "dynamic_axes": {n: {0: "batch"} for n in names + ["logits", "act_logits"]},
        "opset_version": 18,
        "do_constant_folding": True,
        "export_params": True,
    }
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        try:
            torch.onnx.export(wrapper._mod, dummy, str(tmp), dynamo=False, **kwargs)
            return
        except TypeError:
            pass
        except Exception:
            if tmp.exists():
                tmp.unlink()
            torch.onnx.export(wrapper._mod, dummy, str(tmp), dynamo=True, **kwargs)
            _materialize_dynamo(tmp)
            return
    torch.onnx.export(wrapper._mod, dummy, str(tmp), **kwargs)


def _materialize_dynamo(tmp: Path) -> None:
    """torch.onnx.export(dynamo=True) may return/write an in-memory program."""
    if tmp.is_file() and tmp.stat().st_size > 0:
        return
    raise RuntimeError("dynamo ONNX export did not write %s" % tmp)


def _make_session(path: Path, prefer_dml: bool) -> tuple[Any, str] | tuple[None, None]:
    if ort is None:
        return None, None
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # DirectML requires these off; they are harmless for CPU.
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    device_id = _dml_device_id()
    attempts: list[list[Any]] = []
    if prefer_dml and dml_available():
        attempts.append(
            [("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        )
    attempts.append(["CPUExecutionProvider"])
    last_err: Exception | None = None
    for providers in attempts:
        try:
            sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
            used = sess.get_providers()[0]
            label = "dml" if used == "DmlExecutionProvider" else "cpu"
            return sess, label
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err is not None:
        print("[waifu] ORT session failed: %s" % last_err, flush=True)
    return None, None


def _load_tokenizer_cfg() -> tuple[Any, dict[str, Any]]:
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    from .laya_client import HEAD_MAX_LEN, MAX_LEN, MODEL_ID, SUBFOLDER

    model_dir = Path(snapshot_download(MODEL_ID))
    if SUBFOLDER:
        model_dir = model_dir / SUBFOLDER
    _fix_tokenizer_config(str(model_dir))
    cfg_path = Path(model_dir) / "rl_agent_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["head_max_len"] = HEAD_MAX_LEN
    cfg["max_len"] = MAX_LEN
    tok_dir = Path(model_dir) / "tokenizer"
    tok = AutoTokenizer.from_pretrained(str(tok_dir if tok_dir.is_dir() else cfg.get("encoder")))
    return tok, cfg


def _write_meta(path: Path, seq_len: int, kmax: int) -> None:
    from .laya_client import MODEL_ID, SUBFOLDER

    payload = {
        "model": MODEL_ID,
        "subfolder": SUBFOLDER,
        "seq_len": seq_len,
        "kmax": kmax,
        "dynamic_seq": False,
        "opset": 18,
        "inputs": ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        "outputs": ["logits", "act_logits"],
    }
    meta_path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _meta_matches_checkpoint(meta: dict[str, Any]) -> bool:
    from .laya_client import MODEL_ID, SUBFOLDER

    if "model" not in meta and "subfolder" not in meta:
        return False  # pre-checkpoint-tracking export: re-export rather than guess
    return meta.get("model") == MODEL_ID and (meta.get("subfolder") or "") == SUBFOLDER


def _read_meta(path: Path) -> dict[str, Any]:
    mp = meta_path(path)
    if not mp.is_file():
        return {"seq_len": SEQ_LEN, "kmax": KMAX}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {"seq_len": SEQ_LEN, "kmax": KMAX}


def _as_numpy(t: Any) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Laya DecisionModel to ONNX for DirectML.")
    parser.add_argument("--out", type=Path, default=None, help="ONNX output path")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Verify on CPU EP only")
    args = parser.parse_args(argv)

    os.environ.setdefault("USE_TF", "0")
    out = args.out or onnx_path()
    print("[waifu] loading torch Laya (CPU, ~26s)...", flush=True)
    import laya

    from .laya_client import HEAD_MAX_LEN, MAX_LEN, MODEL_ID

    agent = laya.load(MODEL_ID)
    agent.cfg["head_max_len"] = HEAD_MAX_LEN
    agent.cfg["max_len"] = MAX_LEN
    print("[waifu] exporting %s ..." % out, flush=True)
    export_onnx(out, agent=agent)
    print("[waifu] wrote %s (%.1f MB)" % (out, out.stat().st_size / (1024 * 1024)), flush=True)
    if args.skip_verify:
        return 0
    print("[waifu] verifying against torch...", flush=True)
    info = verify_onnx(out, agent=agent, prefer_dml=not args.cpu)
    print(
        "[waifu] provider=%s torch=%.0fms onnx=%.0fms logits_err=%.4g act_err=%.4g"
        % (info["provider"], info["torch_ms"], info["ort_ms"], info["logits_max_abs"], info["act_max_abs"]),
        flush=True,
    )
    print("[waifu] ort providers: %s" % info["ort_providers"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
