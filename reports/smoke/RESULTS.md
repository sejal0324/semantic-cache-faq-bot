> **fake-backend smoke run, not a real measurement.** Fake embedder and fake LLM; numbers prove only that the pipeline runs end to end.

**Cache hit rate 0.0% on 100 test questions at threshold 0.95 with 0 false hits; LLM calls down 0.0%; average latency down from 145 ms to 145 ms.**

_Run 2026-11-12 16:13 UTC · LLM `fake-llm` · Linux x86_64, 2 CPUs, Python 3.13.18_

#### Threshold sweep (`fake-ngram-hash`, negation guard on)

| Threshold | Hits (of 100) | Correct hits (of 60) | Recall on repeats | False hits | Precision | Avg lookup ms | Avg end-to-end ms | p95 end-to-end ms | LLM calls |
|---|---|---|---|---|---|---|---|---|---|
| 0.75 | 4 | 0 | 0.0% | 4 | 0.0% | 0.1 | 138 | 240 | 96 |
| 0.80 | 1 | 0 | 0.0% | 1 | 0.0% | 0.1 | 144 | 240 | 99 |
| 0.85 | 1 | 0 | 0.0% | 1 | 0.0% | 0.1 | 143 | 240 | 99 |
| 0.90 | 0 | 0 | 0.0% | 0 | 100.0% | 0.1 | 145 | 240 | 100 |
| 0.95 **(chosen)** | 0 | 0 | 0.0% | 0 | 100.0% | 0.1 | 145 | 240 | 100 |

Chosen threshold **0.95**: the highest recall with at most 1 false hit (ties go to the higher, safer threshold).

#### Embedding models (best threshold each, guard on)

| Model | Threshold | Recall on repeats | False hits | Avg lookup ms |
|---|---|---|---|---|
| `fake-ngram-hash` | 0.95 | 0.0% | 0 | 0.1 |

#### Negation guard at threshold 0.95

| Guard | False hits |
|---|---|
| off | 0 |
| on | 0 |

#### Before vs after

| | LLM calls | Avg latency | p95 latency |
|---|---|---|---|
| No cache | 100 | 145 ms | 240 ms |
| Semantic cache @ 0.95 | 100 | 145 ms | 240 ms |

Method: cache seeded with the 30 FAQs and read-only during the run; every question sent once to the LLM to measure its real latency; end-to-end = measured lookup + measured LLM time on a miss. Per-question results: `reports/per_question.jsonl`.
