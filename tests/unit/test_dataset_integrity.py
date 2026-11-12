"""The labelled datasets are part of the experiment, so they get tests like code does."""

from collections import Counter

from semcache.config import REPO_ROOT
from semcache.data import load_faqs, load_test_set
from semcache.guards import NegationGuard
from semcache.links import hosts_in
from semcache.pii import PIIDetector

FAQS = load_faqs(REPO_ROOT / "data" / "faqs.jsonl")
TESTS = load_test_set(REPO_ROOT / "data" / "test_set.jsonl")
FAQ_BY_ID = {f.id: f for f in FAQS}

# Paraphrases where real users add a word the guard reads as negation ("bata na").
# Kept on purpose: they measure what the guard costs, they are not labelling mistakes.
KNOWN_GUARD_COST = {"t059", "t060"}


def test_faq_set_shape() -> None:
    assert len(FAQS) == 30
    assert len(FAQ_BY_ID) == 30, "duplicate FAQ ids"
    assert {f.category for f in FAQS} == {"policy", "pricing", "general"}


def test_faq_answers_are_cache_safe() -> None:
    pii = PIIDetector()
    for f in FAQS:
        assert not pii.contains_pii(f.answer), f.id
        assert all(h.endswith("bazaarnow.in") for h in hosts_in(f.answer)), f.id


def test_test_set_shape() -> None:
    assert len(TESTS) == 100
    assert len({t.id for t in TESTS}) == 100
    assert len({t.text.lower() for t in TESTS}) == 100, "duplicate question text"
    should_hit = [t for t in TESTS if t.should_hit]
    assert len(should_hit) == 60
    assert all(t.kind == "paraphrase" for t in should_hit)
    assert all(t.kind != "paraphrase" for t in TESTS if not t.should_hit)


def test_every_label_points_to_a_real_faq_and_every_faq_is_covered() -> None:
    labels = Counter(t.expected_faq_id for t in TESTS if t.should_hit)
    assert set(labels) <= set(FAQ_BY_ID)
    assert set(labels) == set(FAQ_BY_ID), "every FAQ needs at least one paraphrase"


def test_hinglish_share_of_paraphrases_is_about_40_percent() -> None:
    paraphrases = [t for t in TESTS if t.should_hit]
    share = sum(t.lang == "hinglish" for t in paraphrases) / len(paraphrases)
    assert 0.35 <= share <= 0.45


def test_negative_set_has_negations_and_near_misses() -> None:
    kinds = Counter(t.kind for t in TESTS if not t.should_hit)
    assert kinds["negation"] >= 8
    assert kinds["near_miss"] >= 8
    guard = NegationGuard.default()
    for t in TESTS:
        if t.kind == "negation":
            assert guard.is_negated(t.text), t.id


def test_paraphrases_agree_with_guard_except_known_cost() -> None:
    guard = NegationGuard.default()
    conflicts = {
        t.id
        for t in TESTS
        if t.should_hit and not guard.allows(t.text, FAQ_BY_ID[t.expected_faq_id].question)
    }
    assert conflicts == KNOWN_GUARD_COST
