import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from semcache import tracing
from semcache.api import create_app
from semcache.bot import FAQBot
from semcache.cache import SemanticCache
from semcache.config import Settings
from semcache.embeddings import FakeEmbedder
from semcache.factory import build_bot
from semcache.llm import FakeLLM

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def provider() -> None:
    p = TracerProvider()
    p.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    trace.set_tracer_provider(p)


@pytest.fixture
def spans() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER


def bot() -> FAQBot:
    return FAQBot(
        SemanticCache(FakeEmbedder(), threshold=0.8, store_kind="memory"),
        FakeLLM(responder=lambda m: "Refunds take 5-7 days."),
        price_in_per_1m=1.0,
        price_out_per_1m=2.0,
    )


async def test_miss_emits_stage_spans_with_tokens_and_cost(spans: InMemorySpanExporter) -> None:
    b = bot()
    r = await b.ask("t", "my phone 9876543210, when is my refund?")
    names = [s.name for s in spans.get_finished_spans()]
    assert {"faq.ask", "embed", "cache.lookup", "llm.generate"} <= set(names)
    root = next(s for s in spans.get_finished_spans() if s.name == "faq.ask")
    assert root.attributes["cached"] is False
    assert root.attributes["tokens"] > 0
    assert root.attributes["cost_usd"] == pytest.approx(r.cost_usd) and r.cost_usd > 0
    llm = next(s for s in spans.get_finished_spans() if s.name == "llm.generate")
    assert llm.attributes["model"] == "fake-llm"
    every_value = " ".join(
        str(v) for s in spans.get_finished_spans() for v in (s.attributes or {}).values()
    )
    assert "9876543210" not in every_value, "raw user text must never reach traces"
    assert b.metrics.snapshot()["cost_usd"] == pytest.approx(r.cost_usd)


async def test_exact_hit_skips_embedding_and_llm(spans: InMemorySpanExporter) -> None:
    b = bot()
    await b.ask("t", "Do you deliver on Sundays?")
    spans.clear()
    r = await b.ask("t", "do you deliver on sundays?")
    names = {s.name for s in spans.get_finished_spans()}
    assert r.cached and r.cost_usd == 0.0
    assert "cache.exact_lookup" in names
    assert "embed" not in names and "llm.generate" not in names


def test_setup_tracing_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", seen.append)
    monkeypatch.setattr(tracing, "_configured", False)
    assert tracing.setup_tracing("off") is False
    assert tracing.setup_tracing("console") is True
    assert len(seen) == 1
    assert tracing.setup_tracing("console") is False, "configure once"


def test_demo_page_is_safe_html() -> None:
    s = Settings(_env_file=None, llm_provider="fake", embedder="fake")
    c = TestClient(create_app(bot=build_bot(s), settings=s))
    r = c.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "textContent" in r.text and "innerHTML" not in r.text, "LLM output must not be HTML"
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
