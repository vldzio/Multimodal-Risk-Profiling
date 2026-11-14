# Financial Risk Profiling (Multimodal-Risk-Profiling) - project context

## What this is
A research notebook (`US_bankrupt.ipynb`, Colab) on bankruptcy risk: Altman Z, Zmijewski and Ohlson scores plus classifier and FinBERT experiments. The `demo-ready` branch adds a hosted demo:

```text
Browser (Vercel, static `demo/`) -> Hugging Face Space `demo_api/` (limits, BYO key, LangSmith)
                                       -> private FreeLLMAPI gateway (127.0.0.1:3001, same container) -> free-tier LLMs
```

## Layout
- `demo_api/risk_core.py` - the notebook's feature definitions and coefficients as plain Python (cells 14-36: Altman 1.2/1.4/3.3/0.6/1.0 with zones 2.99/1.81; Zmijewski via normal CDF; Ohlson via sigmoid; `overall()` combiner). Deterministic, no LLM, free.
- `demo_api/app.py` - `/score` (no LLM), `/explain` (LLM narrative from computed numbers), `/sentiment` (LLM positive/negative/neutral labels). Shared plumbing: `llm.py`, `base.py`, `guard.py`, `gateway.py`.
- `demo/index.html` (+ `byo.js`, `base.css`, `config.js`, `write-config.mjs`) - static UI. `deploy/space/` - HF Space recipe.

## Run and test
- `python -m venv .venv && .venv\Scripts\pip install -r demo_api/requirements.txt pytest && .venv\Scripts\pytest tests` (20 tests; LLM stubbed at `llm.chat_json`).
- Local UI: set `API_URL` in `demo/config.js` to your local `uvicorn demo_api.app:app --port 8140`, serve `demo/` with `python -m http.server`. Restore `API_URL` to `""` before committing.

## Rules
- Scoring maths must keep matching the notebook; change both or neither. Never hard-code LLM text. LLM seam: `demo_api/llm.py`.
- Client IP for limits = LAST `X-Forwarded-For` entry. Never log API keys. Example company figures in the UI are illustrative, not real firms.

## Deploy facts
- Space `risk-demo-api`, LangSmith project `risk-demo`, Vercel dir `demo/` (static, `vercel.json` build runs `node demo/write-config.mjs`), env var `DEMO_API_URL`.
- Space secrets: `ENCRYPTION_KEY` (64 hex), `FREEAPI_CONFIG_JSON`, `LANGSMITH_API_KEY`, `UPSTASH_REDIS_REST_URL/TOKEN`; optional `CORS_ORIGINS`, `RL_PER_IP_DAILY`, `RL_GLOBAL_DAILY`. The Space clones branch `demo-ready` (build arg `REF`).
- Steps: `hf auth login`, `vercel login`, create the Space from `deploy/space/`, set secrets, then `vercel deploy --prod --yes --build-env DEMO_API_URL=<space url>` from the repo root.

## Status
Verified: 20 tests pass; UI checked in a browser. NOT verified: Space Docker build, real provider calls, live deploy. The notebook itself was not modified; the dataset (`american_bankruptcy.csv`) is not in the repo.
