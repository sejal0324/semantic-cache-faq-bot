"""End-to-end through HTTP with fake backends: request -> embed -> cache -> LLM -> response."""

import pytest
from fastapi.testclient import TestClient

from semcache.api import create_app
from semcache.config import Settings
from semcache.factory import build_bot
from semcache.llm import FakeLLM, LLMError


def fake_settings(**kw: object) -> Settings:
    return Settings(_env_file=None, llm_provider="fake", embedder="fake", **kw)  # type: ignore[arg-type]


@pytest.fixture
def client() -> TestClient:
    llm = FakeLLM(responder=lambda m: "We deliver in 2-5 days.")
    bot = build_bot(fake_settings(cache_threshold=0.9), llm=llm)
    return TestClient(create_app(bot=bot, settings=fake_settings()))


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_seeded_faq_is_a_cache_hit(client: TestClient) -> None:
    r = client.post("/ask", json={"tenant_id": "bazaarnow", "question": "How do I track my order?"})
    assert r.status_code == 200
    body = r.json()
    assert body["cached"] is True
    assert body["source"] == "f01"
    assert "My Orders" in body["answer"]


def test_miss_then_hit_for_new_tenant(client: TestClient) -> None:
    q = {"tenant_id": "shop-2", "question": "Do you ship to Nepal?"}
    first = client.post("/ask", json=q).json()
    second = client.post("/ask", json=q).json()
    assert (first["cached"], second["cached"]) == (False, True)
    assert first["answer"] == second["answer"] == "We deliver in 2-5 days."


def test_metrics_endpoint(client: TestClient) -> None:
    client.post("/ask", json={"tenant_id": "bazaarnow", "question": "How do I track my order?"})
    m = client.get("/metrics").json()
    assert m["requests"] == 1 and m["hits"] == 1 and m["hit_rate"] == 1.0


def test_llm_outage_returns_503_without_details() -> None:
    def down(_: object) -> str:
        raise LLMError("connection refused to 10.0.0.5:11434")

    bot = build_bot(fake_settings(), llm=FakeLLM(responder=down))  # type: ignore[arg-type]
    c = TestClient(create_app(bot=bot, settings=fake_settings()))
    r = c.post("/ask", json={"tenant_id": "x", "question": "anything new?"})
    assert r.status_code == 503
    assert "10.0.0.5" not in r.text


def test_create_app_builds_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDER", "fake")
    with TestClient(create_app()) as c:
        r = c.post("/ask", json={"tenant_id": "bazaarnow", "question": "How do I track my order?"})
        assert r.status_code == 200
        assert r.json()["cached"] is True


def test_build_bot_seeds_configured_tenant() -> None:
    bot = build_bot(fake_settings(seed_tenant="acme"))
    assert bot.cache.size("acme") == 30
    assert bot.cache.size("bazaarnow") == 0
