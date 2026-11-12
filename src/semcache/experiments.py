"""Experiment pipeline: threshold sweep, guard on/off, model comparison, before vs after.

Method (also printed in the README):
- The cache is seeded with the 30 curated FAQs and is read-only during evaluation, so the
  order of the 100 test questions cannot change the result.
- Each test question is embedded once per model; its embedding time is measured and reused
  across thresholds. Lookup latency = embedding time + vector search time.
- Each test question is also sent to the real LLM once (no cache) to measure its LLM latency.
  End-to-end latency for a question = lookup latency, plus its measured LLM latency on a miss.
  So every number is built from real measurements; nothing is assumed.
- A false hit is a cache hit that returns the wrong FAQ, or any hit on a "none" question.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from semcache.bot import SYSTEM_PROMPT, seed_faqs
from semcache.cache import SemanticCache
from semcache.config import Settings
from semcache.data import FAQ, EvalQuestion, load_faqs, load_test_set
from semcache.embeddings import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from semcache.guards import NegationGuard
from semcache.llm import FakeLLM, LLMClient, Message, build_llm
from semcache.readme import replace_block
from semcache.store import StoreKind, Vector

THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
DEFAULT_MODELS = ("all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2")
EVAL_TENANT = "eval"
SMOKE_LABEL = "fake-backend smoke run, not a real measurement"


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    qid: str
    text: str
    expected: str
    lang: str
    kind: str
    hit: bool
    matched_faq: str | None
    similarity: float
    correct: bool
    false_hit: bool
    guard_blocked: bool
    blocked_cached_question: str | None
    lookup_ms: float
    e2e_ms: float


@dataclass(frozen=True, slots=True)
class SweepRow:
    model: str
    threshold: float
    guard: bool
    hits: int
    correct_hits: int
    false_hits: int
    hit_rate: float  # correct hits / should-hit questions (recall on repeats)
    overall_hit_rate: float  # hits / all questions (share of traffic served from cache)
    precision: float
    avg_lookup_ms: float
    avg_e2e_ms: float
    p95_e2e_ms: float
    llm_calls: int


@dataclass
class PreparedModel:
    name: str
    cache: SemanticCache
    vectors: list[Vector]
    embed_ms: list[float]


@dataclass
class ExperimentResults:
    mode: str
    created_at: str
    machine: str
    llm_model: str
    models: list[str]
    rows: list[SweepRow]
    chosen: SweepRow
    baseline_llm_calls: int
    baseline_avg_ms: float
    baseline_p95_ms: float
    guard_off_false_hits: int
    negation_examples: list[tuple[str, str]]
    headline: str
    per_question: list[QuestionRecord] = field(default_factory=list)


# ------------------------------------------------------------------------------------------


def prepare_model(
    embedder: Embedder,
    faqs: Sequence[FAQ],
    tests: Sequence[EvalQuestion],
    *,
    store_kind: StoreKind = "faiss",
    name: str | None = None,
) -> PreparedModel:
    cache = SemanticCache(embedder, threshold=0.0, store_kind=store_kind, default_ttl_s=1e12)
    seed_faqs(cache, EVAL_TENANT, faqs)
    cache.embed("warm up")  # keep one-off model initialisation out of the timings
    vectors: list[Vector] = []
    embed_ms: list[float] = []
    for t in tests:
        start = time.perf_counter()
        vectors.append(cache.embed(t.text))
        embed_ms.append((time.perf_counter() - start) * 1000)
    return PreparedModel(name or embedder.name, cache, vectors, embed_ms)


def evaluate(
    prepared: PreparedModel,
    tests: Sequence[EvalQuestion],
    *,
    threshold: float,
    guard: bool,
    llm_ms: Mapping[str, float],
) -> tuple[SweepRow, list[QuestionRecord]]:
    cache = prepared.cache
    cache.threshold = threshold
    cache.negation_guard = NegationGuard.default() if guard else None
    records: list[QuestionRecord] = []
    for t, vector, e_ms in zip(tests, prepared.vectors, prepared.embed_ms, strict=True):
        start = time.perf_counter()
        r = cache.lookup(EVAL_TENANT, t.text, vector=vector)
        lookup_ms = e_ms + (time.perf_counter() - start) * 1000
        matched = r.entry.source_id if r.hit and r.entry else None
        correct = r.hit and t.should_hit and matched == t.expected_faq_id
        blocked = r.reason == "negation_mismatch"
        records.append(
            QuestionRecord(
                qid=t.id,
                text=t.text,
                expected=t.expected_faq_id,
                lang=t.lang,
                kind=t.kind,
                hit=r.hit,
                matched_faq=matched,
                similarity=round(r.similarity, 4),
                correct=correct,
                false_hit=r.hit and not correct,
                guard_blocked=blocked,
                blocked_cached_question=cache.negation_catches[-1][1] if blocked else None,
                lookup_ms=lookup_ms,
                e2e_ms=lookup_ms + (0.0 if r.hit else llm_ms[t.id]),
            )
        )
    should_hit = sum(t.should_hit for t in tests)
    hits = sum(r.hit for r in records)
    correct_hits = sum(r.correct for r in records)
    e2e = np.array([r.e2e_ms for r in records])
    row = SweepRow(
        model=prepared.name,
        threshold=threshold,
        guard=guard,
        hits=hits,
        correct_hits=correct_hits,
        false_hits=hits - correct_hits,
        hit_rate=correct_hits / should_hit if should_hit else 0.0,
        overall_hit_rate=hits / len(tests) if tests else 0.0,
        precision=correct_hits / hits if hits else 1.0,
        avg_lookup_ms=float(np.mean([r.lookup_ms for r in records])) if records else 0.0,
        avg_e2e_ms=float(e2e.mean()) if records else 0.0,
        p95_e2e_ms=float(np.percentile(e2e, 95)) if records else 0.0,
        llm_calls=len(tests) - hits,
    )
    return row, records


def pick_threshold(rows: Sequence[SweepRow], *, max_false_hits: int = 1) -> SweepRow:
    """Best hit rate with false hits within budget; ties go to the higher (safer) threshold.
    If nothing is within budget, take the fewest false hits."""
    if not rows:
        raise ValueError("no sweep rows")
    ok = [r for r in rows if r.false_hits <= max_false_hits]
    if ok:
        return max(ok, key=lambda r: (r.hit_rate, r.threshold))
    return min(rows, key=lambda r: (r.false_hits, -r.threshold))


async def measure_llm(
    llm: LLMClient, tests: Sequence[EvalQuestion], *, concurrency: int = 1
) -> dict[str, tuple[float, int]]:
    """Send every test question to the LLM once, without the cache. Returns ms and tokens."""
    sem = asyncio.Semaphore(concurrency)

    async def one(t: EvalQuestion) -> tuple[str, tuple[float, int]]:
        async with sem:
            msgs = [Message("system", SYSTEM_PROMPT), Message("user", t.text)]
            start = time.perf_counter()
            resp = await llm.complete(msgs, temperature=0.0, max_tokens=256)
            ms = (time.perf_counter() - start) * 1000
            return t.id, (ms, resp.input_tokens + resp.output_tokens)

    return dict(await asyncio.gather(*(one(t) for t in tests)))


def headline(
    *,
    hit_rate: float,
    threshold: float,
    false_hits: int,
    llm_reduction: float,
    before_ms: float,
    after_ms: float,
    n: int,
) -> str:
    return (
        f"Cache hit rate {hit_rate * 100:.1f}% on {n} test questions at threshold "
        f"{threshold:.2f} with {false_hits} false hits; LLM calls down {llm_reduction * 100:.1f}%;"
        f" average latency down from {before_ms:.0f} ms to {after_ms:.0f} ms."
    )


# ------------------------------------------------------------------------------------------


def _machine() -> str:
    return (
        f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, "
        f"Python {platform.python_version()}"
    )


def render_markdown(res: ExperimentResults) -> str:
    c = res.chosen
    lines: list[str] = []
    if res.mode == "smoke":
        lines += [f"> **{SMOKE_LABEL}.** Fake embedder and fake LLM; numbers prove only", ""]
        lines[-2] += " that the pipeline runs end to end."
    lines += [
        f"**{res.headline}**",
        "",
        f"_Run {res.created_at} · LLM `{res.llm_model}` · {res.machine}_",
        "",
        f"#### Threshold sweep (`{c.model}`, negation guard on)",
        "",
        "| Threshold | Hits (of 100) | Correct hits (of 60) | Recall on repeats | False hits "
        "| Precision | Avg lookup ms | Avg end-to-end ms | p95 end-to-end ms | LLM calls |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res.rows:
        if r.model == c.model and r.guard:
            mark = " **(chosen)**" if r.threshold == c.threshold else ""
            lines.append(
                f"| {r.threshold:.2f}{mark} | {r.hits} | {r.correct_hits} | {r.hit_rate:.1%} "
                f"| {r.false_hits} | {r.precision:.1%} | {r.avg_lookup_ms:.1f} "
                f"| {r.avg_e2e_ms:.0f} | {r.p95_e2e_ms:.0f} | {r.llm_calls} |"
            )
    lines += [
        "",
        f"Chosen threshold **{c.threshold:.2f}**: the highest recall with at most 1 false hit "
        "(ties go to the higher, safer threshold).",
        "",
        "#### Embedding models (best threshold each, guard on)",
        "",
        "| Model | Threshold | Recall on repeats | False hits | Avg lookup ms |",
        "|---|---|---|---|---|",
    ]
    for model in res.models:
        best = pick_threshold([r for r in res.rows if r.model == model and r.guard])
        lines.append(
            f"| `{model}` | {best.threshold:.2f} | {best.hit_rate:.1%} | {best.false_hits} "
            f"| {best.avg_lookup_ms:.1f} |"
        )
    lines += [
        "",
        f"#### Negation guard at threshold {c.threshold:.2f}",
        "",
        "| Guard | False hits |",
        "|---|---|",
        f"| off | {res.guard_off_false_hits} |",
        f"| on | {c.false_hits} |",
        "",
    ]
    if res.negation_examples:
        lines.append("Questions the guard stopped from getting a wrong cached answer:")
        lines.append("")
        for q, cached in res.negation_examples[:3]:
            lines.append(f'- "{q}" was not given the cached answer for "{cached}"')
        lines.append("")
    lines += [
        "#### Before vs after",
        "",
        "| | LLM calls | Avg latency | p95 latency |",
        "|---|---|---|---|",
        f"| No cache | {res.baseline_llm_calls} | {res.baseline_avg_ms:.0f} ms "
        f"| {res.baseline_p95_ms:.0f} ms |",
        f"| Semantic cache @ {c.threshold:.2f} | {c.llm_calls} | {c.avg_e2e_ms:.0f} ms "
        f"| {c.p95_e2e_ms:.0f} ms |",
        "",
        "Method: cache seeded with the 30 FAQs and read-only during the run; every question "
        "sent once to the LLM to measure its real latency; end-to-end = measured lookup + "
        "measured LLM time on a miss. Per-question results: `reports/per_question.jsonl`.",
    ]
    return "\n".join(lines)


def _write(res: ExperimentResults, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = asdict(res)
    per_question = summary.pop("per_question")
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for rec in per_question:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def run_experiments(
    settings: Settings,
    *,
    smoke: bool,
    out_dir: Path,
    readme: Path,
    models: Sequence[str] | None = None,
    thresholds: Sequence[float] = THRESHOLDS,
    max_false_hits: int = 1,
    embedder_factory: Callable[[str], Embedder] | None = None,
    llm: LLMClient | None = None,
) -> ExperimentResults:
    if not smoke and llm is None and settings.llm_provider == "fake":
        raise SystemExit(
            "Real experiments need a real LLM: set LLM_PROVIDER=ollama or gemini in .env "
            "(or use --smoke for a fake-backend run)."
        )
    faqs = load_faqs(settings.data_dir / "faqs.jsonl")
    tests = load_test_set(settings.data_dir / "test_set.jsonl")

    if smoke:
        model_names: Sequence[str] = ["fake-ngram-hash"]
        factory: Callable[[str], Embedder] = embedder_factory or (lambda _: FakeEmbedder())
        client = llm or FakeLLM(latency_s=(0.05, 0.25), seed=7)
        concurrency = 25
    else:
        model_names = list(models or DEFAULT_MODELS)
        factory = embedder_factory or SentenceTransformerEmbedder
        client = llm or build_llm(settings)
        concurrency = 1  # sequential, so latency numbers are not inflated by queueing

    llm_stats = await measure_llm(client, tests, concurrency=concurrency)
    llm_ms = {qid: ms for qid, (ms, _) in llm_stats.items()}

    rows: list[SweepRow] = []
    records: dict[tuple[str, bool, float], list[QuestionRecord]] = {}
    for name in model_names:
        prepared = prepare_model(
            factory(name), faqs, tests, store_kind=settings.vector_store, name=name
        )
        for guard in (True, False):
            for th in thresholds:
                row, recs = evaluate(prepared, tests, threshold=th, guard=guard, llm_ms=llm_ms)
                rows.append(row)
                records[(name, guard, th)] = recs

    best_per_model = [
        pick_threshold([r for r in rows if r.model == m and r.guard], max_false_hits=max_false_hits)
        for m in model_names
    ]
    chosen = pick_threshold(best_per_model, max_false_hits=max_false_hits)
    chosen_records = records[(chosen.model, True, chosen.threshold)]
    guard_off = next(
        r
        for r in rows
        if r.model == chosen.model and not r.guard and r.threshold == chosen.threshold
    )
    baseline = np.array(list(llm_ms.values()))
    res = ExperimentResults(
        mode="smoke" if smoke else "real",
        created_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        machine=_machine(),
        llm_model=client.model,
        models=list(model_names),
        rows=rows,
        chosen=chosen,
        baseline_llm_calls=len(tests),
        baseline_avg_ms=float(baseline.mean()),
        baseline_p95_ms=float(np.percentile(baseline, 95)),
        guard_off_false_hits=guard_off.false_hits,
        negation_examples=[
            (r.text, r.blocked_cached_question)
            for r in chosen_records
            if r.guard_blocked and r.blocked_cached_question
        ],
        headline=headline(
            hit_rate=chosen.overall_hit_rate,
            threshold=chosen.threshold,
            false_hits=chosen.false_hits,
            llm_reduction=(len(tests) - chosen.llm_calls) / len(tests),
            before_ms=float(baseline.mean()),
            after_ms=chosen.avg_e2e_ms,
            n=len(tests),
        ),
        per_question=chosen_records,
    )

    markdown = render_markdown(res)
    if smoke:
        target = out_dir / "smoke"
        _write(res, target)
        (target / "RESULTS.md").write_text(markdown + "\n", encoding="utf-8")
    else:
        _write(res, out_dir)
        text = readme.read_text(encoding="utf-8")
        text = replace_block(text, "HEADLINE", f"**{res.headline}**")
        text = replace_block(text, "RESULTS", markdown)
        readme.write_text(text, encoding="utf-8")
    return res
