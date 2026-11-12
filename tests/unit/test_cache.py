import pytest

from semcache.cache import SemanticCache
from semcache.embeddings import FakeEmbedder
from semcache.guards import NegationGuard


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cache(clock: Clock) -> SemanticCache:
    return SemanticCache(
        FakeEmbedder(),
        threshold=0.5,
        store_kind="memory",
        ttl_s_by_category={"pricing": 100.0, "policy": 1_000.0},
        default_ttl_s=10_000.0,
        clock=clock,
    )


def test_miss_on_empty_cache(cache: SemanticCache) -> None:
    r = cache.lookup("t1", "When will I get my refund?")
    assert not r.hit
    assert r.reason == "empty"
    assert r.entry is None


def test_exact_repeat_hits(cache: SemanticCache) -> None:
    assert cache.put("t1", "When will I get my refund?", "5-7 days.", category="policy").stored
    r = cache.lookup("t1", "When will I get my refund?")
    assert r.hit
    assert r.entry is not None and r.entry.answer == "5-7 days."
    assert r.similarity == pytest.approx(1.0, abs=1e-5)
    assert r.reason == "hit"


def test_below_threshold_is_miss(cache: SemanticCache) -> None:
    cache.put("t1", "When will I get my refund?", "5-7 days.")
    r = cache.lookup("t1", "Do you sell gift cards?")
    assert not r.hit
    assert r.reason == "below_threshold"
    assert 0.0 <= r.similarity < 0.5


def test_threshold_is_inclusive(clock: Clock) -> None:
    c = SemanticCache(FakeEmbedder(), threshold=1.0, store_kind="memory", clock=clock)
    c.put("t", "track my order", "Use My Orders.")
    assert c.lookup("t", "track my order").hit


def test_negation_guard_blocks_false_hit(cache: SemanticCache) -> None:
    cache.put("t1", "refund milega?", "Yes, within 7 days.")
    r = cache.lookup("t1", "refund nahi milega?")
    assert not r.hit
    assert r.reason == "negation_mismatch"
    assert cache.negation_catches[-1] == ("refund nahi milega?", "refund milega?")


def test_guard_can_be_disabled(clock: Clock) -> None:
    c = SemanticCache(
        FakeEmbedder(), threshold=0.5, store_kind="memory", negation_guard=None, clock=clock
    )
    c.put("t1", "refund milega?", "Yes.")
    assert c.lookup("t1", "refund nahi milega?").hit


def test_guard_falls_through_to_next_candidate(clock: Clock) -> None:
    c = SemanticCache(
        FakeEmbedder(),
        threshold=0.3,
        store_kind="memory",
        negation_guard=NegationGuard.default(),
        clock=clock,
    )
    c.put("t", "refund milega kya", "Yes.")
    c.put("t", "refund nahi mila", "Raise a ticket.")
    r = c.lookup("t", "refund nahi milega kya")
    assert r.hit and r.entry is not None
    assert r.entry.answer == "Raise a ticket."


def test_ttl_expiry_by_category(cache: SemanticCache, clock: Clock) -> None:
    cache.put("t1", "What is the Plus membership price?", "Rs 999/year.", category="pricing")
    clock.now += 99
    assert cache.lookup("t1", "What is the Plus membership price?").hit
    clock.now += 2
    r = cache.lookup("t1", "What is the Plus membership price?")
    assert not r.hit
    assert r.reason == "expired"
    assert cache.size("t1") == 0, "expired entries are evicted on read"


def test_unknown_category_uses_default_ttl(cache: SemanticCache, clock: Clock) -> None:
    cache.put("t1", "How do I contact support?", "Chat with us.", category="weird")
    clock.now += 9_999
    assert cache.lookup("t1", "How do I contact support?").hit


def test_pii_in_answer_or_question_is_never_cached(cache: SemanticCache) -> None:
    d1 = cache.put("t1", "where is my order", "Order OD1234567890 ships today.")
    d2 = cache.put("t1", "my number is 9876543210, call me", "Sure, we will call.")
    assert (d1.stored, d1.reason) == (False, "pii")
    assert (d2.stored, d2.reason) == (False, "pii")
    assert cache.size("t1") == 0


def test_empty_answer_not_cached(cache: SemanticCache) -> None:
    assert cache.put("t1", "q", "   ").reason == "empty_answer"


def test_tenants_are_isolated(cache: SemanticCache) -> None:
    cache.put("shop-a", "When will I get my refund?", "A: 5 days.")
    assert not cache.lookup("shop-b", "When will I get my refund?").hit
    assert cache.lookup("shop-a", "When will I get my refund?").hit


def test_lru_eviction_per_tenant(clock: Clock) -> None:
    c = SemanticCache(
        FakeEmbedder(), threshold=0.99, store_kind="memory", max_entries_per_tenant=2, clock=clock
    )
    c.put("t", "alpha question", "a")
    c.put("t", "beta question", "b")
    assert c.lookup("t", "alpha question").hit  # touch alpha -> beta is now least recent
    c.put("t", "gamma question", "c")
    assert c.size("t") == 2
    assert not c.lookup("t", "beta question").hit
    assert c.lookup("t", "alpha question").hit


def test_same_question_put_twice_replaces(cache: SemanticCache) -> None:
    cache.put("t1", "How do I track my order?", "old")
    cache.put("t1", "  how do I track my ORDER? ", "new")
    assert cache.size("t1") == 1
    r = cache.lookup("t1", "How do I track my order?")
    assert r.entry is not None and r.entry.answer == "new"


def test_precomputed_vector_is_used(cache: SemanticCache) -> None:
    v = cache.embed("When will I get my refund?")
    cache.put("t1", "When will I get my refund?", "5 days", vector=v)
    assert cache.lookup("t1", "When will I get my refund?", vector=v).hit


def test_source_id_kept(cache: SemanticCache) -> None:
    cache.put("t1", "track order", "My Orders page", source_id="f01")
    r = cache.lookup("t1", "track order")
    assert r.entry is not None and r.entry.source_id == "f01"


def test_threshold_can_be_changed(cache: SemanticCache) -> None:
    cache.threshold = 0.9
    assert cache.threshold == 0.9
    with pytest.raises(ValueError):
        cache.threshold = 1.5
