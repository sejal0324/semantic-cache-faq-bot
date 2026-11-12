import pytest
from hypothesis import given
from hypothesis import strategies as st

from semcache.guards import NegationGuard
from semcache.pii import PIIDetector
from semcache.text import normalize, tokenize

# ---------- text ----------


def test_normalize_nfkc_whitespace_and_zero_width() -> None:
    assert normalize("  refund\u200b   kab\tmilega\uff1f ") == "refund kab milega?"


def test_normalize_curly_apostrophe() -> None:
    assert normalize("don\u2019t") == "don't"


def test_tokenize_lowercases_and_keeps_apostrophes() -> None:
    assert tokenize("Why DON'T you ship to UK?") == ["why", "don't", "you", "ship", "to", "uk"]


@given(st.text(max_size=200))
def test_normalize_is_idempotent(s: str) -> None:
    assert normalize(normalize(s)) == normalize(s)


# ---------- negation guard ----------

guard = NegationGuard.default()


@pytest.mark.parametrize(
    "text",
    [
        "refund nahi milega?",
        "Why is COD not available?",
        "I never got my order",
        "Why can't I return this?",
        "Why don\u2019t you ship to the UK?",
        "order cancel nhi ho raha",
        "isko return mat karo",
        "without a receipt can I return",
    ],
)
def test_detects_negation(text: str) -> None:
    assert guard.is_negated(text)


@pytest.mark.parametrize(
    "text",
    ["refund kab milega?", "When will I get my refund?", "Is COD available?", "nothing special"],
)
def test_no_false_negation(text: str) -> None:
    assert not guard.is_negated(text)


def test_guard_blocks_only_when_exactly_one_side_negated() -> None:
    assert not guard.allows("refund milega", "refund nahi milega")
    assert guard.allows("refund milega", "refund kab milega")
    assert guard.allows("coupon not working", "coupon apply nahi ho raha")


def test_guard_custom_words() -> None:
    g = NegationGuard(frozenset({"nope"}))
    assert g.is_negated("nope that")
    assert not g.is_negated("not that")


# ---------- PII ----------

pii = PIIDetector()


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("where is OD1234567890123", "order_id"),
        ("my order od9876543210 is late", "order_id"),
        ("mail me at ravi.k+shop@gmail.com", "email"),
        ("call 9876543210", "phone"),
        ("call +91 98765 43210", "phone"),
        ("call +91-9876543210", "phone"),
        ("my name is Priya, refund please", "name"),
        ("mera naam Rahul hai", "name"),
    ],
)
def test_pii_detected(text: str, kind: str) -> None:
    assert kind in pii.find(text)
    assert pii.contains_pii(text)


@pytest.mark.parametrize(
    "text",
    [
        "When will I get my refund?",
        "Refunds take 5-7 business days.",
        "Orders above Rs 499 ship free",
        "Call 1800-123-4567 for support",  # toll-free support number, not a user's phone
        "Use code SAVE10",
    ],
)
def test_pii_not_detected(text: str) -> None:
    assert not pii.contains_pii(text)
