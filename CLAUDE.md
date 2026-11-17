# Financial Risk Profiling (Multimodal-Risk-Profiling) - project context

## What this is
A research notebook (`US_bankrupt.ipynb`, Colab) on bankruptcy risk: Altman Z, Zmijewski and Ohlson scores plus classifier and FinBERT experiments. The `demo-ready` branch adds a hosted demo:

```text
Browser (Vercel, static `demo/`) -> Hugging Face Space `demo_api/` (limits, BYO key, LangSmith)
                                       -> private FreeLLMAPI gateway (127.0.0.1:3001, same container) -> free-tier LLMs
```

## Layout
- `demo_api/risk_core.py` - the notebook's feature definitions and coefficients as plain Python (cells 14-36: Altman 1.2/1.4/3.3/0.6/1.0 with zones 2.99/1.81; Zmijewski via normal CDF; Ohlson via sigmoid; `overall()` combiner). Deterministic, no LLM, free.
- `demo_api/app.py` - `/score` (no LLM), `/explain` (LLM narrative from computed numbers), `/sentiment` (FinBERT via `finbert.py`, int8 ONNX, no LLM, costs no credits; build the bundle with `python scripts/export_finbert_onnx.py`, output `demo_api/model/` is gitignored; Dockerfile does it at build). Shared plumbing: `llm.py`, `base.py`, `guard.py`, `gateway.py`.
- `demo/index.html` (+ `byo.js`, `base.css`, `config.js`, `write-config.mjs`) - static UI. `deploy/space/` - HF Space recipe.

## Run and test
- `python -m venv .venv && .venv\Scripts\pip install -r demo_api/requirements.txt pytest && .venv\Scripts\pytest tests` (24 tests; LLM stubbed at `llm.chat_json`; the real-model test skips if `demo_api/model/` is absent).
- Local UI: set `API_URL` in `demo/config.js` to your local `uvicorn demo_api.app:app --port 8140`, serve `demo/` with `python -m http.server`. Restore `API_URL` to `""` before committing.

## Rules
- Scoring maths must keep matching the notebook; change both or neither. Never hard-code LLM text. FinBERT label order is the checkpoint's own (0 neutral, 1 positive, 2 negative); the notebook's two zero-shot `sentiment_map` cells were corrected to match (the fine-tuning `label_map` is separate and self-consistent). LLM seam: `demo_api/llm.py`.
- Client IP for limits = LAST `X-Forwarded-For` entry, except `CF-Connecting-IP` when `TRUST_CF_CONNECTING_IP=true` (Render, behind Cloudflare only). Never log API keys. Example company figures in the UI are illustrative, not real firms.

## Deploy facts
- LIVE: UI https://risk-profiling-demo.vercel.app (Vercel project `risk-profiling-demo`, static `demo/`; `.vercelignore` keeps the upload tiny; `vercel.json` build runs `node demo/write-config.mjs`; env `DEMO_API_URL`). API https://risk-demo-api.onrender.com (Render free web service `risk-demo-api`, id `srv-dbu5h56k1f9s73acm0b0`, Oregon, Docker from `deploy/space/Dockerfile`, blueprint `render.yaml`). Created via the Render API using the key in `C:\Users\zack0\render.key`; auto-deploy on push is unreliable, trigger `POST /v1/services/<id>/deploys`.
- Render env: `ENCRYPTION_KEY`, `FREEAPI_CONFIG_JSON` (provider keys from `C:\Users\zack0\workspace-demo.config.json`, own admin password), `UPSTASH_REDIS_REST_URL/TOKEN` (shared DB, prefix `DEMO_PROJECT=risk`), `LANGSMITH_API_KEY` (`C:\Users\zack0\risk-profiling-langsmith.key`) + `LANGSMITH_ENDPOINT`/`LANGSMITH_BASE_URL=https://apac.api.smith.langchain.com` (workspace is APAC; a wrong region fails silently), `TRUST_CF_CONNECTING_IP=true`, `PORT=7860`, `CORS_ORIGINS` = the Vercel origin, `RL_PER_IP_DAILY=20`, `RL_GLOBAL_DAILY=300`. Key files are `NAME=value` lines: strip the prefix and quotes.
- The HF Space recipe (same Dockerfile) still works; the Space clones branch `demo-ready` (build arg `REF`).
- Known: `/explain` latency varies 2-110 s (gateway `auto` routing fails over across slow free providers; `FALLBACK_TIME_BUDGET_MS`, default 45000, could be lowered). Free Render sleeps when idle (~1 min cold start).

## Status
Verified: 24 tests pass; local UI; live Render API + Vercel UI (scores, FinBERT sentiment, LLM explain, Upstash counters, LangSmith APAC traces, CORS locked). NOT verified: the 20-per-visitor cap on the live service, cold-start behaviour. The notebook was modified only to fix the two zero-shot FinBERT `sentiment_map` cells; the dataset (`american_bankruptcy.csv`) is not in the repo.
