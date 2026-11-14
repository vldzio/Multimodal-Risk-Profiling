"""Tests for the risk-scoring maths and the demo API (LLM calls are stubbed at the llm.chat_json seam)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from demo_api import guard, llm, risk_core  # noqa: E402

HEALTHY = dict(
    Current_Assets=500, Total_Current_Liabilities=200, Total_Assets=1000, Retained_Earnings=400,
    EBIT=150, Market_Value=1200, Net_Sales=1300, Net_Income=100, Total_Liabilities=400,
)
DISTRESSED = dict(
    Current_Assets=100, Total_Current_Liabilities=300, Total_Assets=800, Retained_Earnings=-250,
    EBIT=-60, Market_Value=50, Net_Sales=400, Net_Income=-120, Total_Liabilities=900,
)


def test_altman_matches_hand_calculation():
    r = risk_core.score(HEALTHY)
    expected = 1.2 * 0.3 + 1.4 * 0.4 + 3.3 * 0.15 + 0.6 * 3.0 + 1.0 * 1.3
    assert r["altman"]["z"] == pytest.approx(expected, abs=1e-3)
    assert r["altman"]["zone"] == "Safe"


def test_healthy_is_low_risk_and_distressed_is_risk():
    assert risk_core.score(HEALTHY)["overall"] == "Low Risk"
    d = risk_core.score(DISTRESSED)
    assert d["altman"]["zone"] == "Distress"
    assert d["zmijewski"]["risk"] == "High Risk" and d["ohlson"]["risk"] == "High Risk"
    assert d["overall"] == "Risk"


def test_probabilities_are_valid_and_match_formulae():
    r = risk_core.score(DISTRESSED)
    f = risk_core.parse_financials(DISTRESSED)
    x = risk_core.features(f)
    zs = -4.3 - 4.5 * x["ROA"] + 5.7 * x["Leverage"] - 0.004 * x["Current_Ratio_Calc"]
    assert r["zmijewski"]["probability"] == pytest.approx(0.5 * (1 + math.erf(zs / math.sqrt(2))), abs=1e-3)
    assert 0 <= r["ohlson"]["probability"] <= 1


@pytest.mark.parametrize("field", list(HEALTHY))
def test_missing_field_rejected(field):
    bad = dict(HEALTHY)
    del bad[field]
    with pytest.raises(ValueError):
        risk_core.score(bad)


def test_non_positive_denominators_rejected():
    with pytest.raises(ValueError):
        risk_core.score({**HEALTHY, "Total_Assets": 0})


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(guard, "UPSTASH_URL", "")
    guard._mem.clear()
    from demo_api.app import app

    return TestClient(app)


def test_score_endpoint_is_free_and_deterministic(client):
    for _ in range(30):  # far beyond any LLM quota
        r = client.post("/score", json={"financials": HEALTHY})
        assert r.status_code == 200 and r.json()["overall"] == "Low Risk"


def test_score_validation_error(client):
    r = client.post("/score", json={"financials": {"Total_Assets": 1}})
    assert r.status_code == 400 and "Missing value" in r.json()["error"]


def test_explain_passes_computed_numbers_to_llm(client, monkeypatch):
    seen = {}

    def fake(system, user, **kw):
        seen["user"] = user
        return {"headline": "h", "drivers": ["d"], "disagreements": "", "next_steps": ["n"]}

    monkeypatch.setattr(llm, "chat_json", fake)
    r = client.post("/explain", json={"company": "Acme", "financials": DISTRESSED})
    assert r.status_code == 200
    assert r.json()["narrative"]["headline"] == "h"
    assert '"zone": "Distress"' in seen["user"] and "Acme" in seen["user"]


def test_llm_unavailable_maps_to_503(client, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMUnavailable("pool busy")

    monkeypatch.setattr(llm, "chat_json", boom)
    r = client.post("/explain", json={"financials": HEALTHY})
    assert r.status_code == 503 and r.json()["error"] == "pool busy"


def test_rate_limit_returns_429_and_byo_bypasses(client, monkeypatch):
    monkeypatch.setattr(guard, "PER_IP_DAILY", 1)
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"headline": "h", "drivers": [], "next_steps": []})
    assert client.post("/explain", json={"financials": HEALTHY}).status_code == 200
    assert client.post("/explain", json={"financials": HEALTHY}).status_code == 429
    ok = client.post("/explain", json={"financials": HEALTHY}, headers={"X-LLM-Api-Key": "sk-test"})
    assert ok.status_code == 200


def test_byo_with_disallowed_host_rejected(client):
    r = client.post(
        "/explain", json={"financials": HEALTHY}, headers={"X-LLM-Api-Key": "k", "X-LLM-Base-Url": "https://evil.example/v1"}
    )
    assert r.status_code == 400


def test_sentiment_requires_sentences(client):
    assert client.post("/sentiment", json={"sentences": []}).status_code == 400
