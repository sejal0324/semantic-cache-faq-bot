### recheck-500

Server env: `{"LLM_PROVIDER": "fake", "EMBEDDER": "fake", "FAKE_LLM_LATENCY_S": "[0.3, 1.5]", "FAKE_EMBED_LATENCY_S": "0.01", "LOG_LEVEL": "WARNING", "MAX_INFLIGHT_REQUESTS": "256"}`

| Users | Requests | RPS | p50 ms | p95 ms | p99 ms | Error rate | Errors | Hit rate | LLM calls | Server CPU avg/max % | Server RSS MB | Load-gen CPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 4094 | 147.1 | 1151.6 | 4733.7 | 6573.6 | 0.0% | - | 0.954 | 112 | 64.2/111.7 | 84.7 | 78.5 |
