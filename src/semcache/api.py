"""HTTP API. Run with: uvicorn --factory semcache.api:create_app"""

from __future__ import annotations

import hmac
import logging
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from semcache.bot import FAQBot
from semcache.config import Settings
from semcache.factory import build_bot
from semcache.llm import LLMError
from semcache.logging_utils import configure_logging
from semcache.text import normalize
from semcache.tracing import setup_tracing
from semcache.ui import PAGE, SECURITY_HEADERS

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
    setup_tracing(settings.tracing)
    app = FastAPI(title="semantic-cache-faq-bot", version="0.1.0")
    app.state.bot = bot or build_bot(settings)
    max_chars = settings.max_question_chars
    max_inflight = settings.max_inflight_requests
    app.state.inflight = 0

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", extra={"fields": {"error": type(exc).__name__}})
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    admin_token = settings.admin_token.get_secret_value()

    @app.delete("/cache/{tenant_id}")
    async def purge(
        tenant_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")],
        source_id: Annotated[str | None, Query(pattern=r"^f\d{2,}$")] = None,
        generated_only: bool = False,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, int]:
        if not admin_token:
            raise HTTPException(503, detail="cache admin is disabled (ADMIN_TOKEN not set)")
        if not x_admin_token or not hmac.compare_digest(
            x_admin_token.encode(), admin_token.encode()
        ):
            raise HTTPException(401, detail="invalid admin token")
        removed = app.state.bot.cache.purge(
            tenant_id, source_id=source_id, generated_only=generated_only
        )
        return {"removed": removed}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def demo_page() -> HTMLResponse:
        return HTMLResponse(PAGE, headers=SECURITY_HEADERS)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, float | int]:
        snapshot: dict[str, float | int] = app.state.bot.metrics.snapshot()
        return snapshot

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        # Load shedding: answer "busy" in 1 ms instead of queueing for a minute.
        if max_inflight and app.state.inflight >= max_inflight:
            app.state.bot.metrics.record_shed()
            raise HTTPException(503, detail="overloaded, retry", headers={"Retry-After": "1"})
        app.state.inflight += 1
        try:
            return await _answer(req)
        finally:
            app.state.inflight -= 1

    async def _answer(req: AskRequest) -> AskResponse:
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
