"""/sentiment uses the notebook's FinBERT via onnxruntime: no LLM, no daily credits."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from demo_api import finbert, guard, llm  # noqa: E402
from demo_api.app import app  # noqa: E402


@pytest.fixture()
def client():
    guard._mem.clear()
    return TestClient(app)


def test_sentiment_does_not_call_llm_or_spend_credits(client, monkeypatch):
    monkeypatch.setattr(guard, "PER_IP_DAILY", 1)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: pytest.fail("LLM must not be used for sentiment"))
    monkeypatch.setattr(
        finbert, "classify", lambda s: [{"sentence": x, "label": "neutral", "confidence": 0.9, "scores": {}} for x in s]
    )
    for _ in range(3):  # more calls than the per-IP credit budget
        r = client.post("/sentiment", json={"sentences": ["a", "b"]})
        assert r.status_code == 200
    body = r.json()
    assert body["model"] == "yiyanghkust/finbert-tone" and len(body["results"]) == 2


def test_sentiment_caps_at_eight_sentences(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(finbert, "classify", lambda s: seen.setdefault("n", len(s)) and [])
    client.post("/sentiment", json={"sentences": [str(i) for i in range(20)]})
    assert seen["n"] == 8


def test_sentiment_503_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(finbert, "MODEL_DIR", Path("does-not-exist"))
    finbert._state.clear()
    r = client.post("/sentiment", json={"sentences": ["x"]})
    assert r.status_code == 503 and "FinBERT" in r.json()["error"]


@pytest.mark.skipif(not finbert.available(), reason="run scripts/export_finbert_onnx.py to build the model bundle")
def test_real_model_labels_obvious_sentences():
    out = finbert.classify(
        [
            "The company reported a 15% rise in quarterly net profit.",
            "Management warned of significant liquidity pressure and potential default.",
        ]
    )
    assert [o["label"] for o in out] == ["positive", "negative"]
    assert all(0 < o["confidence"] <= 1 and abs(sum(o["scores"].values()) - 1) < 0.01 for o in out)
