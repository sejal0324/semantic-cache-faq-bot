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
