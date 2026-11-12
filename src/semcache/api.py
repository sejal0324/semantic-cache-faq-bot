"""HTTP API. Run with: uvicorn --factory semcache.api:create_app"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from semcache.bot import FAQBot
from semcache.config import Settings
from semcache.factory import build_bot
from semcache.llm import LLMError
from semcache.logging_utils import configure_logging
from semcache.text import normalize

HARD_MAX_CHARS = 2_000  # checked before normalisation, bounds the work per request
logger = logging.getLogger("semcache.api")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    question: str = Field(min_length=1, max_length=HARD_MAX_CHARS)


class AskResponse(BaseModel):
    answer: str
    cached: bool
    similarity: float
    reason: str
    latency_ms: float
    source: str | None


def create_app(bot: FAQBot | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="semantic-cache-faq-bot", version="0.1.0")
    app.state.bot = bot or build_bot(settings)
    max_chars = settings.max_question_chars

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", extra={"fields": {"error": type(exc).__name__}})
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, float | int]:
        snapshot: dict[str, float | int] = app.state.bot.metrics.snapshot()
        return snapshot

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        # Normalise first, then check length: NFKC can expand one char into many.
        question = normalize(req.question)
        if not question or len(question) > max_chars:
            raise HTTPException(status_code=422, detail=f"question must be 1-{max_chars} chars")
        try:
            result = await app.state.bot.ask(req.tenant_id, question)
        except LLMError as exc:
            raise HTTPException(
                status_code=503, detail="answer service temporarily unavailable"
            ) from exc
        return AskResponse(
            answer=result.answer,
            cached=result.cached,
            similarity=round(result.similarity, 4),
            reason=result.reason,
            latency_ms=round(result.latency_ms, 2),
            source=result.source_id,
        )

    return app
