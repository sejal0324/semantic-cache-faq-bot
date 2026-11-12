### tuned-cpu-check

Server env: `{"LLM_PROVIDER": "fake", "EMBEDDER": "fake", "FAKE_LLM_LATENCY_S": "[0.3, 1.5]", "FAKE_EMBED_LATENCY_S": "0.01", "LOG_LEVEL": "WARNING", "MAX_INFLIGHT_REQUESTS": "256"}`

| Users | Requests | RPS | p50 ms | p95 ms | p99 ms | Error rate | Errors | Hit rate | LLM calls | Server CPU avg/max % | Server RSS MB | Load-gen CPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 4243 | 142.6 | 2863.1 | 11094.3 | 16357.5 | 0.0% | - | 0.942 | 126 | 58.8/95.5 | 107.7 | 80.4 |
| 2000 | 3663 | 93.5 | 12132.1 | 29122.1 | 33023.7 | 1.3% | {'503': 42, 'ReadError': 5} | 0.931 | 116 | 43.0/97.9 | 113.0 | 83.4 |
