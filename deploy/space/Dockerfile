# Hugging Face Docker Space: private FreeLLMAPI gateway + Financial Risk Profiling demo API in one container.
#
#   public :7860  ->  demo_api (FastAPI: rate limits, BYO key, LangSmith tracing)
#   private :3001 ->  FreeLLMAPI (bound to 127.0.0.1, never exposed)
#
# The project code is cloned at build time, so the Space repo only needs this
# Dockerfile + start.sh + README.md. Rebuild the Space to pick up new commits.
FROM ghcr.io/tashfeenahmed/freellmapi:latest

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG REPO=https://github.com/vldzio/Multimodal-Risk-Profiling.git
ARG REF=demo-ready
ARG CACHEBUST=1
RUN git clone --depth 1 --branch "${REF}" "${REPO}" /srv/app \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /srv/app/demo_api/requirements.txt \
 && /opt/venv/bin/python -c "import onnxruntime, tokenizers"
# Build-time only: export the notebook's FinBERT (yiyanghkust/finbert-tone) to a ~110 MB int8 ONNX bundle,
# then throw the heavy export toolchain (torch etc.) away so the runtime image stays small.
RUN python3 -m venv /opt/export \
 && /opt/export/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      torch transformers onnx onnxruntime onnxscript tokenizers "numpy<2.3" \
 && /opt/export/bin/python /srv/app/scripts/export_finbert_onnx.py /srv/app/demo_api/model \
 && rm -rf /opt/export /root/.cache \
 && chown -R node:node /srv/app

COPY start.sh /srv/start.sh
RUN tr -d '\015' < /srv/start.sh > /tmp/start.sh && install -m 755 /tmp/start.sh /srv/start.sh && rm /tmp/start.sh

USER node
ENV HOME=/home/node \
    FREEAPI_DB_PATH=/app/server/data/freeapi.db \
    DEMO_PROJECT=risk \
    LANGSMITH_PROJECT=risk-demo
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:7860/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["/srv/start.sh"]
