"""FinBERT sentiment (yiyanghkust/finbert-tone, the notebook's model) on onnxruntime.

Deterministic, no LLM, no quota. The model is an int8-quantized ONNX export produced by
``scripts/export_finbert_onnx.py`` (build time); at runtime only onnxruntime + tokenizers are needed.
"""
from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

MODEL_DIR = Path(os.environ.get("FINBERT_DIR") or Path(__file__).resolve().parent / "model")
MAX_LEN = 512

_lock = threading.Lock()
_state: dict[str, Any] = {}


class SentimentUnavailable(RuntimeError):
    """The ONNX model bundle is missing or failed to load."""


def available() -> bool:
    return (MODEL_DIR / "model.int8.onnx").exists()


def _load() -> dict[str, Any]:
    with _lock:
        if _state:
            return _state
        if not available():
            raise SentimentUnavailable("The FinBERT model is not installed on this server.")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
            tok.enable_truncation(max_length=MAX_LEN)
            tok.enable_padding()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.environ.get("FINBERT_THREADS", "1"))
            sess = ort.InferenceSession(str(MODEL_DIR / "model.int8.onnx"), opts, providers=["CPUExecutionProvider"])
            labels = json.loads((MODEL_DIR / "labels.json").read_text())
        except Exception as exc:  # noqa: BLE001
            raise SentimentUnavailable(f"FinBERT failed to load ({type(exc).__name__}).") from exc
        _state.update(tok=tok, sess=sess, labels=labels, inputs={i.name for i in sess.get_inputs()})
        return _state


def _softmax(row: list[float]) -> list[float]:
    m = max(row)
    e = [math.exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]


def classify(sentences: list[str]) -> list[dict[str, Any]]:
    import numpy as np

    st = _load()
    enc = st["tok"].encode_batch(sentences)
    feeds = {
        "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
    }
    logits = st["sess"].run(["logits"], {k: v for k, v in feeds.items() if k in st["inputs"]})[0]
    out = []
    for sentence, row in zip(sentences, logits.tolist()):
        probs = _softmax(row)
        i = max(range(len(probs)), key=probs.__getitem__)
        out.append(
            {
                "sentence": sentence,
                "label": st["labels"][i],
                "confidence": round(probs[i], 4),
                "scores": {lab: round(p, 4) for lab, p in zip(st["labels"], probs)},
            }
        )
    return out
