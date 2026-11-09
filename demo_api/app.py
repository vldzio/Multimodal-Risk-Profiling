"""Financial Risk Profiling demo API.

* POST /score      - deterministic Altman / Zmijewski / Ohlson scoring (the notebook's maths), free
* POST /explain    - LLM analyst narrative grounded in the computed numbers
* POST /sentiment  - LLM sentiment of financial sentences (FinBERT-style positive/negative/neutral)
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("DEMO_PROJECT", "risk")
os.environ.setdefault("LANGSMITH_PROJECT", "risk-demo")

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from . import llm, risk_core  # noqa: E402
from .base import create_app, guarded  # noqa: E402

app = create_app("Financial Risk Profiling demo API")

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "drivers": {"type": "array", "items": {"type": "string"}},
        "disagreements": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "drivers", "next_steps"],
}

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "string"},
                    "label": {"enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["sentence", "label", "confidence"],
            },
        }
    },
    "required": ["results"],
}


async def _body(req: Request) -> dict:
    try:
        data = await req.json()
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


@app.post("/score")
async def score(req: Request):
    body = await _body(req)
    try:
        return risk_core.score(body.get("financials") or body)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/explain")
async def explain(req: Request):
    body = await _body(req)
    try:
        result = risk_core.score(body.get("financials") or body)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    company = str(body.get("company") or "the company")[:80]

    def run():
        system = (
            "You are a credit-risk analyst. Explain bankruptcy-risk model outputs to a non-specialist. "
            "Use ONLY the numbers provided; never invent figures. Be concise and concrete."
        )
        user = (
            f"Company: {company}\nModel results (JSON):\n{json.dumps(result)}\n\n"
            "Return JSON with: headline (one sentence verdict), drivers (3-5 bullet strings naming the specific ratios "
            "and values that drive the result), disagreements (note if the three models disagree and why, else empty string), "
            "next_steps (2-3 practical follow-ups for an analyst)."
        )
        narrative = llm.chat_json(system, user, schema=EXPLAIN_SCHEMA, metadata={"route": "explain"})
        return {"scores": result, "narrative": narrative}

    return await guarded(req, 1, run, name="risk.explain")


@app.post("/sentiment")
async def sentiment(req: Request):
    body = await _body(req)
    sentences = [str(s).strip()[:400] for s in (body.get("sentences") or []) if str(s).strip()][:8]
    if not sentences:
        return JSONResponse({"error": "Provide 1-8 sentences."}, status_code=400)

    def run():
        system = (
            "You label financial-news sentences like the Financial PhraseBank: positive, negative or neutral "
            "from an investor's point of view. Judge only what the sentence says."
        )
        user = "Classify each sentence. Return JSON {\"results\": [...]} in the same order.\n" + json.dumps(sentences)
        return llm.chat_json(system, user, schema=SENTIMENT_SCHEMA, metadata={"route": "sentiment"})

    return await guarded(req, 1, run, name="risk.sentiment")
