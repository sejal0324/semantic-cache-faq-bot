import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from semcache.bot import FAQBot, seed_faqs
from semcache.cache import SemanticCache
from semcache.data import FAQ, EvalQuestion, load_faqs, load_jsonl, load_test_set
from semcache.embeddings import FakeEmbedder
from semcache.llm import FakeLLM, LLMError, Message
from semcache.metrics import Metrics


def make_bot(llm: FakeLLM | None = None, **kw: object) -> tuple[FAQBot, FakeLLM, SemanticCache]:
    llm = llm or FakeLLM(responder=lambda m: "Refunds take 5-7 business days.")
    cache = SemanticCache(FakeEmbedder(), threshold=0.8, store_kind="memory")
    return FAQBot(cache, llm, **kw), llm, cache  # type: ignore[arg-type]


# ---------- bot ----------


async def test_miss_calls_llm_then_repeat_hits_cache() -> None:
    bot, llm, _ = make_bot()
    r1 = await bot.ask("t", "When will I get my refund?")
    r2 = await bot.ask("t", "when will I get my refund")
    assert (r1.cached, r2.cached) == (False, True)
    assert r1.answer == r2.answer == "Refunds take 5-7 business days."
    assert llm.calls == 1
    assert r1.stored_reason == "stored"
    assert r1.tokens > 0 and r2.tokens == 0
    assert r2.similarity > 0.8


async def test_writeback_disabled_never_stores() -> None:
    bot, llm, cache = make_bot(writeback=False)
    await bot.ask("t", "When will I get my refund?")
    await bot.ask("t", "When will I get my refund?")
    assert llm.calls == 2
    assert cache.size("t") == 0


async def test_pii_question_answered_but_not_stored() -> None:
    bot, _, cache = make_bot()
    r = await bot.ask("t", "my order OD1234567890 is late, refund?")
    assert r.answer and not r.cached
    assert r.stored_reason == "pii"
    assert cache.size("t") == 0


async def test_llm_failure_propagates_and_is_counted() -> None:
    def boom(_: object) -> str:
        raise LLMError("down")

    bot, _, _ = make_bot(llm=FakeLLM(responder=boom))  # type: ignore[arg-type]
    with pytest.raises(LLMError):
        await bot.ask("t", "hello")
    assert bot.metrics.snapshot()["errors"] == 1


async def test_prompt_contains_system_and_question() -> None:
    seen: list[list[Message]] = []

    def capture(msgs: list[Message]) -> str:
        seen.append(list(msgs))
        return "ok"

    bot, _, _ = make_bot(llm=FakeLLM(responder=capture))  # type: ignore[arg-type]
    await bot.ask("t", "  Do you deliver on Sundays?  ")
    assert seen[0][0].role == "system"
    assert "BazaarNow" in seen[0][0].content
    assert seen[0][-1] == Message("user", "Do you deliver on Sundays?")


async def test_logs_hash_not_raw_question(caplog: pytest.LogCaptureFixture) -> None:
    bot, _, _ = make_bot()
    with caplog.at_level(logging.INFO, logger="semcache.bot"):
        await bot.ask("t", "my phone is 9876543210 refund please")
    rec = caplog.records[-1]
    fields = rec.fields  # type: ignore[attr-defined]
    assert "9876543210" not in json.dumps(fields)
    assert len(fields["q_hash"]) == 16
    assert fields["cached"] is False
    assert fields["pii"] is True


async def test_seed_faqs_makes_faq_questions_hit() -> None:
    bot, llm, cache = make_bot()
    faqs = [
        FAQ(id="f01", question="How do I track my order?", answer="My Orders.", category="general"),
        FAQ(id="f02", question="What is the Plus price?", answer="Rs 999.", category="pricing"),
    ]
    assert seed_faqs(cache, "t", faqs) == 2
    r = await bot.ask("t", "How do I track my order?")
    assert r.cached and r.answer == "My Orders."
    assert r.source_id == "f01"
    assert llm.calls == 0


async def test_metrics_reflect_traffic() -> None:
    bot, _, _ = make_bot()
    await bot.ask("t", "When will I get my refund?")
    await bot.ask("t", "When will I get my refund?")
    snap = bot.metrics.snapshot()
    assert snap["requests"] == 2
    assert snap["hits"] == 1 and snap["misses"] == 1
    assert snap["hit_rate"] == 0.5
    assert snap["llm_calls"] == 1
    assert snap["p50_ms"] >= 0 and snap["p95_ms"] >= snap["p50_ms"]


def test_metrics_empty_snapshot() -> None:
    snap = Metrics().snapshot()
    assert snap["requests"] == 0 and snap["hit_rate"] == 0.0 and snap["p95_ms"] == 0.0


# ---------- data ----------


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_faqs_and_test_set(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "faqs.jsonl",
        [{"id": "f01", "question": "q", "answer": "a", "category": "policy", "topic": "refund"}],
    )
    write_jsonl(
        tmp_path / "t.jsonl",
        [
            {
                "id": "t001",
                "text": "x",
                "expected_faq_id": "f01",
                "lang": "en",
                "kind": "paraphrase",
            },
            {
                "id": "t002",
                "text": "y",
                "expected_faq_id": "none",
                "lang": "hinglish",
                "kind": "negation",
            },
        ],
    )
    faqs = load_faqs(tmp_path / "faqs.jsonl")
    tests = load_test_set(tmp_path / "t.jsonl")
    assert faqs[0].id == "f01"
    assert tests[1].should_hit is False and tests[0].should_hit is True


def test_load_jsonl_rejects_bad_rows(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "f01", "question": "q"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_jsonl(p, FAQ)


def test_load_jsonl_skips_blank_lines_and_rejects_huge_files(tmp_path: Path) -> None:
    p = tmp_path / "ok.jsonl"
    p.write_text(
        '\n{"id":"f01","question":"q","answer":"a","category":"general"}\n\n', encoding="utf-8"
    )
    assert len(load_jsonl(p, FAQ)) == 1
    with pytest.raises(ValueError, match="too large"):
        load_jsonl(p, FAQ, max_bytes=10)


def test_test_question_validation() -> None:
    with pytest.raises(ValidationError):
        EvalQuestion(id="t1", text="x", expected_faq_id="f01", lang="fr", kind="paraphrase")  # type: ignore[arg-type]
