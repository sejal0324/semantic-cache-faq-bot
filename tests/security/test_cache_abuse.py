"""Attacks on the cache itself: poisoning via write-back, memory exhaustion, tenant leaks."""

import pytest

from semcache.cache import SemanticCache
from semcache.embeddings import FakeEmbedder


def make(**kw: object) -> SemanticCache:
    return SemanticCache(FakeEmbedder(), threshold=0.5, store_kind="memory", **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "answer",
    [
        "Track it at http://bazaarnow-support.evil.example/login",
        "See [your order](https://phish.example/track)",
        "Go to www.free-refunds.example now",
    ],
)
def test_poisoned_answer_with_foreign_link_is_not_cached(answer: str) -> None:
    """A miss writes the LLM answer back. If a crafted question makes the LLM emit an
    attacker link, that answer must not be stored and replayed to every later customer."""
    c = make(allowed_link_domains=frozenset({"bazaarnow.in"}))
    d = c.put("t", "How do I track my order?", answer)
    assert (d.stored, d.reason) == (False, "unsafe_link")
    assert c.size("t") == 0


def test_answer_with_allowed_link_is_cached() -> None:
    c = make(allowed_link_domains=frozenset({"bazaarnow.in"}))
    d = c.put("t", "track order", "Open https://www.bazaarnow.in/orders to track it.")
    assert d.stored


def test_lookalike_allowed_domain_is_rejected() -> None:
    c = make(allowed_link_domains=frozenset({"bazaarnow.in"}))
    for url in ("https://bazaarnow.in.evil.example/x", "https://evilbazaarnow.in/x"):
        assert c.put("t", "q " + url, f"go to {url}").reason == "unsafe_link"


def test_tenant_flood_is_capped() -> None:
    """Random tenant ids must not create unbounded per-tenant indexes (memory DoS)."""
    c = make(max_tenants=3)
    for i in range(3):
        assert c.put(f"t{i}", "track my order", "My Orders page").stored
    d = c.put("t-attacker-999", "track my order", "My Orders page")
    assert (d.stored, d.reason) == (False, "tenant_limit")
    assert c.tenant_count() == 3


def test_tenant_ids_are_exact_match() -> None:
    c = make()
    c.put("shop", "When will I get my refund?", "5 days")
    for other in ("Shop", "shop ", "shop​", "sh0p"):
        assert not c.lookup(other, "When will I get my refund?").hit, other


def test_per_tenant_entry_cap_bounds_memory() -> None:
    c = make(max_entries_per_tenant=50)
    for i in range(500):
        c.put("t", f"unique question number {i} zz{i}", f"answer {i}")
    assert c.size("t") == 50
