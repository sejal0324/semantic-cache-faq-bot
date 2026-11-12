# Security review

After every feature I switched roles to "attacker" and tried to break what I had just built.
Each real finding has a regression test. Severity uses a simple scale: **High** (data leak or
cross-tenant impact), **Medium** (abuse or DoS that one user can trigger), **Low** (hardening).

This is a portfolio project, not a formally audited system.

## Feature 1: LLM clients, config, logging

| # | Finding | Severity | Status | Test |
|---|---|---|---|---|
| 1.1 | Gemini key passed as `?key=` query param would land in proxy and access logs | High | Fixed: key sent in the `x-goog-api-key` header | `test_gemini_builds_request_and_parses_response` |
| 1.2 | Provider error bodies can echo the prompt (user text) back; putting them in exceptions leaks it into logs | Medium | Fixed: errors carry only the status code | `test_ollama_400_is_not_retried_and_body_not_leaked` |
| 1.3 | Settings object printed in a traceback could show the API key | Medium | Fixed: `SecretStr` | `test_secret_not_in_repr` |
| 1.4 | Retry storms: many clients retrying in lockstep after a 429 make the outage worse | Medium | Fixed: capped exponential backoff with full jitter; 4xx never retried | `test_retry_delay_is_capped_and_non_negative`, `test_with_retries_does_not_retry_permanent_errors` |
| 1.5 | A misconfigured `LLM_PROVIDER=gemini` without a key only failed on the first request | Low | Fixed: fails at startup with a clear message | `test_gemini_without_key_fails_fast` |
| 1.6 | No cap on the response body size from the LLM server | Low | Accepted: `max_tokens` bounds normal output; the LLM host is trusted infrastructure | — |

**Scope for improvement**
- Add a circuit breaker so a dead LLM fails fast instead of every request waiting for its timeout.
- Read keys from a secrets manager instead of `.env` in any real deployment.
- Add `Retry-After` header support for 429s.

## Feature 2: text normalisation, negation guard, PII detector

| # | Finding | Severity | Status | Test |
|---|---|---|---|---|
| 2.1 | Zero-width characters inside a word (`na​hi`) hide the negation from the guard | Medium | Fixed: invisible chars stripped before NFKC | `test_negation_guard_resists_obfuscation` |
| 2.2 | Look-alike letters (Cyrillic `о` in "nоt", Armenian `ո` in "ոahi") dodge the guard, so a negated question can get a false cache hit | Medium | Fixed: a confusables fold (`skeleton()`) used only for matching; stored text is never altered | `test_negation_guard_resists_obfuscation` |
| 2.3 | Phone numbers written as `98765.43210` or `9 8 7 6 5 4 3 2 1 0` were not detected, so the answer could be cached with a user's number in it | High | Fixed: second pass with digit separators collapsed | `test_pii_resists_formatting_tricks` |
| 2.4 | Order IDs written as `OD 1234567890` / `od-1234567890` were missed | High | Fixed: optional separator after `OD` | `test_pii_resists_formatting_tricks` |
| 2.5 | ReDoS: could a crafted input make the PII regexes hang? | Medium | Checked: 90 kB input runs in milliseconds; patterns have no nested ambiguous quantifiers | `test_pii_regex_is_not_catastrophic_on_long_input` |
| 2.6 | Spelled-out PII ("ravi at gmail dot com", a bare name, an address) is not detected | Medium | Accepted, documented: pattern matching is not NER | — |
| 2.7 | Hinglish `na` is also a filler particle ("bata na"), so the guard can block a legitimate hit | Low | Accepted: it costs an LLM call, never a wrong answer; measured in the experiments | — |

**Scope for improvement**
- Swap the regex PII detector for a small NER model (e.g. Microsoft Presidio) behind the same interface.
- Use the full Unicode TR39 confusables table instead of a hand-picked list.
- Replace the negation word list with a tiny classifier trained on paraphrase/negation pairs.
