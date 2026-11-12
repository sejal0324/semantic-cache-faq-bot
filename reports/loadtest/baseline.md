### baseline

Server env: `{"LLM_PROVIDER": "fake", "EMBEDDER": "fake", "FAKE_LLM_LATENCY_S": "[0.3, 1.5]", "FAKE_EMBED_LATENCY_S": "0.01", "LOG_LEVEL": "WARNING"}`

| Users | Requests | RPS | p50 ms | p95 ms | p99 ms | Error rate | Errors | Hit rate | LLM calls | Server CPU avg/max % | Server RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 845 | 31.4 | 22.5 | 1033.3 | 1416.6 | 0.0% | - | 0.896 | 88 | 50.4/77.8 | 101.6 |
| 500 | 1893 | 56.8 | 4769.5 | 12664.2 | 15183.4 | 0.2% | {'ReadError': 3} | 0.913 | 164 | 73.5/111.9 | 112.5 |
| 1000 | 2085 | 50.9 | 10408.2 | 28372.0 | 31438.4 | 0.1% | {'ReadError': 1} | 0.907 | 193 | 67.4/109.5 | 114.3 |
| 2000 | 3158 | 54.3 | 21022.0 | 46819.1 | 51360.9 | 0.1% | {'ReadError': 2} | 0.935 | 206 | 71.5/121.9 | 114.0 |
| 5000 | 6125 | 67.1 | 47386.4 | 65062.5 | 66195.0 | 18.3% | {'ConnectTimeout': 1061, 'ReadTimeout': 48, 'ReadError': 10} | 0.954 | 278 | 71.1/111.2 | 187.9 |
