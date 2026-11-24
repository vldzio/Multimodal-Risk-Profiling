"""Export the notebook's sentiment model (yiyanghkust/finbert-tone) to a quantized ONNX bundle.

Needs torch + transformers + onnx + onnxruntime (build-time only; the API itself only needs
onnxruntime + tokenizers). Output: demo_api/model/{model.int8.onnx, tokenizer.json, labels.json}.

    python scripts/export_finbert_onnx.py [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import BertForSequenceClassification, BertTokenizerFast

MODEL_NAME = "yiyanghkust/finbert-tone"
# From the model's own config.json (id2label). NOTE: the notebook's hard-coded map (0 Negative, 1 Neutral,
# 2 Positive) does not match this checkpoint and mislabels its predictions.
LABELS = ["neutral", "positive", "negative"]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tok = BertTokenizerFast.from_pretrained(MODEL_NAME)  # the notebook uses BertTokenizer (lowercasing)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    assert [model.config.id2label[i].lower() for i in range(3)] == LABELS, model.config.id2label

    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    (out / "labels.json").write_text(json.dumps(LABELS))

    enc = tok(["export example sentence", "another"], return_tensors="pt", padding=True)
    fp32 = out / "model.fp32.onnx"
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
        str(fp32),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "token_type_ids", "logits")}
        | {"logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8)
    fp32.unlink()
    for extra in out.glob("*.onnx.data"):
        extra.unlink()
    print("wrote", out, f"({(out / 'model.int8.onnx').stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "demo_api" / "model")
