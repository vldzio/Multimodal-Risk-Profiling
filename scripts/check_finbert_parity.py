"""One-off parity check: int8 ONNX bundle vs the notebook's full-precision model (needs the export deps)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_api import finbert  # noqa: E402

SENTENCES = [
    "The company reported a 15% rise in quarterly net profit.",
    "Management warned of significant liquidity pressure over the next twelve months.",
    "The board will hold its annual general meeting in May.",
    "Operating profit fell to EUR 3.1 million from EUR 8.9 million a year earlier.",
    "Revenue grew 22% year over year, beating analyst expectations.",
    "The firm filed for Chapter 11 bankruptcy protection after missing debt payments.",
    "The company completed the sale of its Nordic subsidiary for EUR 40 million.",
    "Net sales increased by 4%, and the order backlog remains strong.",
    "The auditor raised substantial doubt about the company's ability to continue as a going concern.",
    "The shares are listed on the Helsinki stock exchange.",
]
tok = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").eval()
with torch.no_grad():
    ref = model(**tok(SENTENCES, return_tensors="pt", padding=True, truncation=True, max_length=512)).logits.argmax(1).tolist()
ref = [model.config.id2label[i].lower() for i in ref]
got = [r["label"] for r in finbert.classify(SENTENCES)]
for s, a, b in zip(SENTENCES, ref, got):
    print("OK  " if a == b else "DIFF", f"{a:9} {b:9} {s[:70]}")
agree = sum(a == b for a, b in zip(ref, got))
print(f"agreement {agree}/{len(SENTENCES)}")
sys.exit(0 if agree == len(SENTENCES) else 1)
