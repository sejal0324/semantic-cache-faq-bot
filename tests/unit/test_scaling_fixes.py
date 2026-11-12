"""Fixes found by the load test: exact-match fast path, single-flight, load shedding."""

import asyncio
from collections.abc import Sequence

import httpx
import pytest

from semcache.api import create_app
from semcache.bot import FAQBot
from semcache.cache import SemanticCache
from semcache.config import Settings
from semcache.embeddings import FakeEmbedder
from semcache.factory import build_bot
from semcache.llm import FakeLLM, LLMError


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed(self, texts: Sequence[str]):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().embed(texts)


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


# ---------- exact-match fast path ----------


def test_lookup_exact_is_case_and_space_insensitive() -> None:
    c = SemanticCache(FakeEmbedder(), store_kind="memory")
    c.put("t", "How do I track my order?", "My Orders.")
    r = c.lookup_exact("t", "  how do i TRACK my order? ")
    assert r is not None and r.hit and r.reason == "exact" and r.similarity == 1.0
    assert c.lookup_exact("t", "something else") is None
    assert c.lookup_exact("other-tenant", "How do I track my order?") is None


def test_lookup_exact_respects_ttl() -> None:
    clock = Clock()
    c = SemanticCache(FakeEmbedder(), store_kind="memory", default_ttl_s=10, clock=clock)
    c.put("t", "q one", "a")
    clock.now = 11
    assert c.lookup_exact("t", "q one") is None
    assert c.size("t") == 0


async def test_fast_path_skips_embedding_for_exact_repeats() -> None:
    emb = CountingEmbedder()
    bot = FAQBot(SemanticCache(emb, store_kind="memory"), FakeLLM(), exact_fast_path=True)
    await bot.ask("t", "Do you deliver on Sundays?")
    calls_after_first = emb.calls
    r = await bot.ask("t", "do you deliver on sundays?")
    assert r.cached and r.reason == "exact"
    assert emb.calls == calls_after_first


async def test_fast_path_can_be_disabled() -> None:
    emb = CountingEmbedder()
    bot = FAQBot(SemanticCache(emb, store_kind="memory"), FakeLLM(), exact_fast_path=False)
    await bot.ask("t", "Do you deliver on Sundays?")
    before = emb.calls
    r = await bot.ask("t", "Do you deliver on Sundays?")
    assert r.cached and r.reason == "hit"
    assert emb.calls == before + 1


# ---------- single-flight ----------


async def test_concurrent_identical_misses_call_llm_once() -> None:
    llm = FakeLLM(responder=lambda m: "answer", latency_s=(0.05, 0.05))
    bot = FAQBot(SemanticCache(FakeEmbedder(), store_kind="memory"), llm, single_flight=True)
    results = await asyncio.gather(*(bot.ask("t", "brand new question") for _ in range(10)))
    assert llm.calls == 1
    assert {r.answer for r in results} == {"answer"}
    assert sum(r.reason == "coalesced" for r in results) == 9
    snap = bot.metrics.snapshot()
    assert snap["llm_calls"] == 1 and snap["coalesced"] == 9


async def test_without_single_flight_each_miss_calls_llm() -> None:
    llm = FakeLLM(latency_s=(0.05, 0.05))
    bot = FAQBot(SemanticCache(FakeEmbedder(), store_kind="memory"), llm, single_flight=False)
    await asyncio.gather(*(bot.ask("t", "brand new question") for _ in range(5)))
    assert llm.calls == 5


async def test_single_flight_is_per_tenant() -> None:
    llm = FakeLLM(latency_s=(0.05, 0.05))
    bot = FAQBot(SemanticCache(FakeEmbedder(), store_kind="memory"), llm, single_flight=True)
    await asyncio.gather(bot.ask("a", "same question"), bot.ask("b", "same question"))
    assert llm.calls == 2


async def test_single_flight_error_reaches_all_waiters_and_clears() -> None:
    state = {"fail": True}

    def responder(_: object) -> str:
        if state["fail"]:
            raise LLMError("down")
        return "ok"

    llm = FakeLLM(responder=responder, latency_s=(0.05, 0.05))  # type: ignore[arg-type]
    bot = FAQBot(SemanticCache(FakeEmbedder(), store_kind="memory"), llm, single_flight=True)
    results = await asyncio.gather(*(bot.ask("t", "q?") for _ in range(4)), return_exceptions=True)
    assert all(isinstance(r, LLMError) for r in results)
    state["fail"] = False
    assert (await bot.ask("t", "q?")).answer == "ok", "a failed flight must not stick"


# ---------- load shedding ----------


async def test_api_sheds_load_beyond_max_inflight() -> None:
    settings = Settings(
        _env_file=None, llm_provider="fake", embedder="fake", max_inflight_requests=2
    )
    bot = build_bot(settings, llm=FakeLLM(latency_s=(0.2, 0.2)))
    app = create_app(bot=bot, settings=settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        responses = await asyncio.gather(
            *(
                client.post("/ask", json={"tenant_id": "x", "question": f"new question {i}"})
                for i in range(6)
            )
        )
    codes = sorted(r.status_code for r in responses)
    assert codes.count(200) == 2
    assert codes.count(503) == 4
    shed = next(r for r in responses if r.status_code == 503)
    assert shed.headers["retry-after"] == "1"
    assert bot.metrics.snapshot()["shed"] == 4


@pytest.mark.parametrize("value", [0])
def test_zero_means_no_limit(value: int) -> None:
    s = Settings(_env_file=None, max_inflight_requests=value)
    assert s.max_inflight_requests == 0
