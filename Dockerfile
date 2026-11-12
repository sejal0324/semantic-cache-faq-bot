# Production-style image for the FAQ bot API + demo page.
#   docker build -t semcache .                                  # full image: real embedding model
#   docker build --build-arg WITH_MODELS=0 -t semcache:lite .   # fake backends only (CI, smoke)
#   docker run -p 8000:8000 --env-file .env semcache
# Runs on Hugging Face Spaces (Docker SDK, set PORT=7860), Render, Fly.io or any container host.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG WITH_MODELS=1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/home/app/.cache/huggingface PORT=8000

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/src

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data

# Optional build secret "pipca": a CA bundle for builds behind a TLS-inspecting proxy.
# It is mounted for this step only and is never stored in the image.
RUN --mount=type=secret,id=pipca,required=false \
    if [ -f /run/secrets/pipca ]; then export PIP_CERT=/run/secrets/pipca; fi; \
    if [ "$WITH_MODELS" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install ".[models,otel]"; \
    else \
      pip install ".[otel]"; \
    fi

USER app
# Download the embedding model at build time so the first request isn't slow
RUN if [ "$WITH_MODELS" = "1" ]; then python -c \
      "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"; \
    fi

ENV DATA_DIR=/home/app/src/data
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "exec uvicorn --factory semcache.api:create_app --host 0.0.0.0 --port ${PORT} --workers 1"]
