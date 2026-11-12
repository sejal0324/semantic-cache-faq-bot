# Production-style image for the FAQ bot API + demo page.
# Works on Hugging Face Spaces (Docker SDK), Render, Fly.io, or any container host.
#   docker build -t semcache .
#   docker run -p 8000:8000 --env-file .env semcache
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface PORT=8000

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/src

# CPU-only torch first: ~5x smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data
RUN pip install ".[models,otel]"

USER app
# Download the embedding model at build time, so the first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"

ENV DATA_DIR=/home/app/src/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "uvicorn --factory semcache.api:create_app --host 0.0.0.0 --port ${PORT} --workers 1"]
