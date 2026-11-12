from pathlib import Path

import pytest

from semcache.data import FAQ, EvalQuestion
from semcache.embeddings import FakeEmbedder
from semcache.experiments import (
    SweepRow,
    evaluate,
    headline,
    pick_threshold,
    prepare_model,
)
from semcache.readme import replace_block

FAQS = [
    FAQ(id="f01", question="How do I track my order?", answer="My Orders.", category="general"),
    FAQ(id="f02", question="refund milega kya?", answer="Yes.", category="policy"),
]
TESTS = [
    EvalQuestion(
        id="t001",
        text="How do I track my order?",
        expected_faq_id="f01",
        lang="en",
        kind="paraphrase",
    ),
    EvalQuestion(
        id="t002",
        text="refund nahi milega kya?",
        expected_faq_id="none",
        lang="hinglish",
        kind="negation",
    ),
    EvalQuestion(
        id="t003",
        text="Do you sell gift cards?",
        expected_faq_id="none",
        lang="en",
        kind="out_of_scope",
    ),
]
LLM_MS = {"t001": 1000.0, "t002": 1000.0, "t003": 1000.0}


def row(threshold: float, hit_rate: float, false_hits: int) -> SweepRow:
    return SweepRow(
        model="m",
        threshold=threshold,
        guard=True,
        hits=0,
        correct_hits=0,
        false_hits=false_hits,
        hit_rate=hit_rate,
        overall_hit_rate=0.0,
        precision=1.0,
        avg_lookup_ms=1.0,
        avg_e2e_ms=1.0,
        p95_e2e_ms=1.0,
        llm_calls=0,
    )


def test_pick_threshold_best_hit_rate_within_false_hit_budget() -> None:
    rows = [row(0.75, 0.9, 5), row(0.80, 0.8, 1), row(0.85, 0.8, 0), row(0.90, 0.6, 0)]
    chosen = pick_threshold(rows, max_false_hits=1)
    assert chosen.threshold == 0.85, "ties on hit rate go to the safer (higher) threshold"


def test_pick_threshold_falls_back_to_fewest_false_hits() -> None:
    rows = [row(0.75, 0.9, 5), row(0.80, 0.8, 3), row(0.85, 0.7, 3)]
    assert pick_threshold(rows, max_false_hits=1).threshold == 0.85


def test_pick_threshold_empty() -> None:
    with pytest.raises(ValueError):
        pick_threshold([], max_false_hits=1)


def test_evaluate_counts_hits_false_hits_and_latency() -> None:
    prepared = prepare_model(FakeEmbedder(), FAQS, TESTS, store_kind="memory")
    strict, records = evaluate(prepared, TESTS, threshold=0.95, guard=True, llm_ms=LLM_MS)
    assert strict.correct_hits == 1 and strict.false_hits == 0
    assert strict.hit_rate == 1.0  # 1 of 1 should-hit questions
    assert strict.llm_calls == 2
    assert records[0].matched_faq == "f01" and records[0].correct
    assert strict.avg_e2e_ms > strict.avg_lookup_ms

    loose_no_guard, _ = evaluate(prepared, TESTS, threshold=0.3, guard=False, llm_ms=LLM_MS)
    loose_guard, recs = evaluate(prepared, TESTS, threshold=0.3, guard=True, llm_ms=LLM_MS)
    assert loose_no_guard.false_hits > loose_guard.false_hits
    assert any(r.guard_blocked for r in recs)


def test_headline_format() -> None:
    text = headline(
        hit_rate=0.734,
        threshold=0.85,
        false_hits=1,
        llm_reduction=0.44,
        before_ms=1234.0,
        after_ms=650.4,
        n=100,
    )
    assert text == (
        "Cache hit rate 73.4% on 100 test questions at threshold 0.85 with 1 false hits; "
        "LLM calls down 44.0%; average latency down from 1234 ms to 650 ms."
    )


def test_replace_block(tmp_path: Path) -> None:
    text = "a\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nb"
    out = replace_block(text, "RESULTS", "new")
    assert out == "a\n<!-- RESULTS:START -->\nnew\n<!-- RESULTS:END -->\nb"
    with pytest.raises(ValueError, match="markers"):
        replace_block("no markers", "RESULTS", "x")
