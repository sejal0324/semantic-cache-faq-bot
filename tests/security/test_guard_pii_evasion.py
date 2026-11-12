"""Attacker's view of the negation guard and the PII detector: try to slip past them."""

import time

import pytest

from semcache.guards import NegationGuard
from semcache.pii import PIIDetector

guard = NegationGuard.default()
pii = PIIDetector()


@pytest.mark.parametrize(
    "text",
    [
        "refund na\u200bhi milega",  # zero-width inside word
        "refund n\u043et milega",  # Cyrillic o in "not"
        "refund \u0578ahi",  # Armenian n look-alike
        "refund NAHI milega",  # case
        "refund \uff4e\uff41\uff48\uff49 milega",  # fullwidth letters
    ],
)
def test_negation_guard_resists_obfuscation(text: str) -> None:
    assert guard.is_negated(text), text


@pytest.mark.parametrize(
    "text",
    [
        "call 98765.43210",
        "call 9 8 7 6 5 4 3 2 1 0",
        "phone: 98765-43210",
        "order OD 1234567890",
        "order od-1234567890",
        "call \uff19\uff18\uff17\uff16\uff15\uff14\uff13\uff12\uff11\uff10",  # fullwidth digits
    ],
)
def test_pii_resists_formatting_tricks(text: str) -> None:
    assert pii.contains_pii(text), text


def test_pii_regex_is_not_catastrophic_on_long_input() -> None:
    evil = "a" * 50_000 + "@" + "b." * 20_000 + "!"
    start = time.perf_counter()
    pii.contains_pii(evil)
    assert time.perf_counter() - start < 1.0


def test_negation_guard_on_long_input_is_fast() -> None:
    start = time.perf_counter()
    guard.is_negated("refund " * 20_000)
    assert time.perf_counter() - start < 1.0
