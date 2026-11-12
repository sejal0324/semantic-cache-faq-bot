import pytest
from fastapi.testclient import TestClient

from semcache.api import create_app
from semcache.cache import SemanticCache
from semcache.config import Settings
from semcache.embeddings import FakeEmbedder
from semcache.experiments import cost_usd
from semcache.factory import build_bot
from semcache.llm import FakeLLM

# ---------- cache purge ----------


def make() -> SemanticCache:
    c = SemanticCache(FakeEmbedder(), threshold=0.99, store_kind="memory")
    c.put("t", "refund timeline?", "5-7 days", category="policy", source_id="f11")
    c.put("t", "refund method?", "original method", category="policy", source_id="f12")
    c.put("t", "do you ship to nepal?", "no", category="generated")
    c.put("other", "refund timeline?", "5-7 days", source_id="f11")
    return c


def test_purge_by_source_id() -> None:
    c = make()
    assert c.purge("t", source_id="f11") == 1
    assert not c.lookup("t", "refund timeline?").hit
    assert c.lookup("t", "refund method?").hit
    assert c.lookup("other", "refund timeline?").hit, "other tenants untouched"


def test_purge_generated_only() -> None:
    c = make()
    assert c.purge("t", generated_only=True) == 1
    assert c.size("t") == 2


def test_purge_whole_tenant_and_unknown_tenant() -> None:
    c = make()
    assert c.purge("t") == 3 and c.size("t") == 0
    assert c.purge("nobody") == 0


# ---------- API ----------

TOKEN = "purge-token-123"


def client(**kw: object) -> TestClient:
    s = Settings(_env_file=None, llm_provider="fake", embedder="fake", **kw)  # type: ignore[arg-type]
    return TestClient(create_app(bot=build_bot(s, llm=FakeLLM()), settings=s))


def test_purge_endpoint_requires_token() -> None:
    c = client(admin_token=TOKEN)
    assert c.delete("/cache/bazaarnow").status_code == 401
    assert c.delete("/cache/bazaarnow", headers={"X-Admin-Token": "nope"}).status_code == 401
    r = c.delete("/cache/bazaarnow?source_id=f11", headers={"X-Admin-Token": TOKEN})
    assert r.status_code == 200 and r.json() == {"removed": 1}
    ask = c.post("/ask", json={"tenant_id": "bazaarnow", "question": "When will I get my refund?"})
    assert ask.json()["source"] != "f11"


def test_purge_disabled_without_token() -> None:
    assert client().delete("/cache/bazaarnow", headers={"X-Admin-Token": ""}).status_code == 503


@pytest.mark.parametrize("path", ["/cache/..%2Fetc", "/cache/a b", "/cache/" + "x" * 65])
def test_purge_validates_tenant(path: str) -> None:
    r = client(admin_token=TOKEN).delete(path, headers={"X-Admin-Token": TOKEN})
    assert r.status_code in (404, 422)


# ---------- cost ----------


def test_cost_usd() -> None:
    assert cost_usd(1_000_000, 0, price_in=0.30, price_out=2.50) == pytest.approx(0.30)
    assert cost_usd(0, 2_000_000, price_in=0.30, price_out=2.50) == pytest.approx(5.00)
    assert cost_usd(10, 10, price_in=0.0, price_out=0.0) == 0.0
