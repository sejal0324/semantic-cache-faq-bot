# Changelog

## [Unreleased]

- chore: project scaffold (pyproject, Makefile, CI, pre-commit)
- feat(llm): async LLM client interface with Ollama, Gemini and Fake backends; retries with jittered backoff
- feat(config): env-based settings validated at startup; JSON logging
- feat(embeddings): Embedder interface with sentence-transformers and a deterministic hashing fake
- feat(guards): negation guard (English + Hinglish); PII detector for order IDs, phones, emails, names
- security: fold look-alike letters and collapse digit separators before matching
- feat(cache): SemanticCache with FAISS/numpy stores, best-first candidates, per-category TTL, per-tenant LRU
- security: link allowlist before write-back (cache poisoning), tenant cap (memory DoS)
- feat(bot): FAQBot request flow (embed off-loop, lookup, LLM on miss, guarded write-back), metrics, typed JSONL loaders
- data: 30 hand-written e-commerce FAQs and a 100-question labelled test set (60 paraphrases, 45% Hinglish; 40 negations, near misses, out-of-scope), with integrity tests
- feat(api): FastAPI service (/ask, /healthz, /metrics) with strict validation and generic errors; factory wiring from settings
- feat(experiments): threshold sweep x 2 models x guard on/off, auto-pick threshold, before/after latency, README auto-fill; smoke mode isolated in reports/smoke
- feat(scripts): run_experiments.py, demo.py
- perf: exact-match fast path before embedding, single-flight for concurrent identical misses, load shedding (MAX_INFLIGHT_REQUESTS)
- test(load): load harness (50 to 5,000 users), baseline vs tuned results in LOAD_TEST.md
- docs: README (architecture, design decisions, interview notes), LOAD_TEST.md, SECURITY.md
