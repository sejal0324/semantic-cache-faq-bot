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
