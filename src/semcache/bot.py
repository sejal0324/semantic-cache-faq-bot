"""The FAQ bot: embed -> cache lookup -> (hit) cached answer | (miss) LLM -> write back."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace

from semcache.cache import SemanticCache
from semcache.data import FAQ
from semcache.llm import LLMClient, LLMError, Message
from semcache.logging_utils import log_event, short_hash
from semcache.metrics import Metrics
from semcache.pii import PIIDetector
from semcache.store import Vector
from semcache.text import normalize

logger = logging.getLogger("semcache.bot")

SYSTEM_PROMPT = (
    "You are the customer support assistant for BazaarNow, an Indian online store. "
    "Answer in at most 3 short sentences, in the customer's style (English or Hinglish). "
    "If you are not sure about a policy, say so and point to help.bazaarnow.in. "
    "Never ask for, guess or repeat personal data such as phone numbers or order IDs."
)
GENERATED_CATEGORY = "generated"


@dataclass(frozen=True, slots=True)
class AskResult:
    answer: str
    cached: bool
    similarity: float
    reason: str
    latency_ms: float
    tokens: int
    stored_reason: str | None = None
    source_id: str | None = None


def seed_faqs(cache: SemanticCache, tenant: str, faqs: Iterable[FAQ]) -> int:
    """Pre-load curated FAQ answers. Returns how many were stored."""
    stored = 0
    for faq in faqs:
        decision = cache.put(
            tenant, faq.question, faq.answer, category=faq.category, source_id=faq.id
        )
        stored += int(decision.stored)
    return stored


class FAQBot:
    def __init__(
        self,
        cache: SemanticCache,
        llm: LLMClient,
        *,
        metrics: Metrics | None = None,
        writeback: bool = True,
        exact_fast_path: bool = True,
        single_flight: bool = True,
        max_tokens: int = 256,
    ) -> None:
        self.cache = cache
        self._llm = llm
        self.metrics = metrics or Metrics()
        self._writeback = writeback
        self._max_tokens = max_tokens
        self._pii = PIIDetector()
        self._exact = exact_fast_path
        self._single_flight = single_flight
        self._inflight: dict[tuple[str, str], asyncio.Future[str]] = {}

    async def ask(self, tenant: str, question: str) -> AskResult:
        start = time.perf_counter()
        q = normalize(question)
        try:
            exact = self.cache.lookup_exact(tenant, q) if self._exact else None
            if exact is not None and exact.entry is not None:
                # Fast path: an exact repeat needs no embedding at all.
                lookup = exact
                vector = None
            else:
                # Embedding is CPU-bound: keep it off the event loop.
                vector = await asyncio.to_thread(self.cache.embed, q)
                lookup = self.cache.lookup(tenant, q, vector=vector)
            if lookup.hit and lookup.entry is not None:
                result = AskResult(
                    answer=lookup.entry.answer,
                    cached=True,
                    similarity=lookup.similarity,
                    reason=lookup.reason,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    tokens=0,
                    source_id=lookup.entry.source_id,
                )
            else:
                assert vector is not None  # noqa: S101 - a miss always went through embedding
                result = await self._generate_once(
                    tenant, q, vector, lookup.similarity, lookup.reason
                )
                result = replace(result, latency_ms=(time.perf_counter() - start) * 1000)
        except Exception:
            self.metrics.record_error()
            log_event(logger, "ask_failed", tenant=tenant, q_hash=short_hash(q))
            raise
        self.metrics.record(
            cached=result.cached,
            latency_ms=result.latency_ms,
            tokens=result.tokens,
            coalesced=result.reason == "coalesced",
        )
        log_event(
            logger,
            "ask",
            tenant=tenant,
            q_hash=short_hash(q),
            q_len=len(q),
            pii=self._pii.contains_pii(q),
            cached=result.cached,
            reason=result.reason,
            similarity=round(result.similarity, 4),
            latency_ms=round(result.latency_ms, 2),
            tokens=result.tokens,
            stored=result.stored_reason,
        )
        return result

    async def _generate_once(
        self, tenant: str, q: str, vector: Vector, similarity: float, reason: str
    ) -> AskResult:
        """Single-flight: concurrent identical misses wait for one LLM call (no stampede)."""
        if not self._single_flight:
            return await self._generate(tenant, q, vector, similarity, reason)
        key = (tenant, q.lower())
        existing = self._inflight.get(key)
        if existing is not None:
            answer = await asyncio.shield(existing)
            return AskResult(answer, False, similarity, "coalesced", 0.0, 0)
        flight: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        flight.add_done_callback(_mark_retrieved)
        self._inflight[key] = flight
        try:
            result = await self._generate(tenant, q, vector, similarity, reason)
            flight.set_result(result.answer)
            return result
        except Exception as exc:
            flight.set_exception(exc)
            raise
        finally:
            if not flight.done():  # cancelled leader: release the waiters with an error
                flight.set_exception(LLMError("request cancelled"))
            self._inflight.pop(key, None)

    async def _generate(
        self, tenant: str, q: str, vector: Vector, similarity: float, reason: str
    ) -> AskResult:
        messages = [Message("system", SYSTEM_PROMPT), Message("user", q)]
        resp = await self._llm.complete(messages, temperature=0.0, max_tokens=self._max_tokens)
        stored_reason: str | None = None
        if self._writeback:
            decision = self.cache.put(
                tenant,
                q,
                resp.text,
                category=GENERATED_CATEGORY,
                vector=vector,
            )
            stored_reason = decision.reason
        return AskResult(
            answer=resp.text,
            cached=False,
            similarity=similarity,
            reason=reason,
            latency_ms=0.0,
            tokens=resp.input_tokens + resp.output_tokens,
            stored_reason=stored_reason,
        )


def _mark_retrieved(fut: asyncio.Future[str]) -> None:
    """Avoid 'exception was never retrieved' warnings when a failed flight had no waiters."""
    if not fut.cancelled():
        fut.exception()
