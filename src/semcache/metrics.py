"""In-process request metrics for /metrics. Thread-safe; latency window is bounded."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class Metrics:
    def __init__(self, window: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self.requests = 0
        self.hits = 0
        self.errors = 0
        self.tokens = 0
        self.coalesced = 0
        self.shed = 0
        self.cost_usd = 0.0

    def record(
        self,
        *,
        cached: bool,
        latency_ms: float,
        tokens: int,
        coalesced: bool = False,
        cost_usd: float = 0.0,
    ) -> None:
        with self._lock:
            self.cost_usd += cost_usd
            self.requests += 1
            self.coalesced += int(coalesced)
            self.hits += int(cached)
            self.tokens += tokens
            self._latencies.append(latency_ms)

    def record_shed(self) -> None:
        with self._lock:
            self.shed += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            lat = np.array(self._latencies, dtype=np.float64)
            misses = self.requests - self.hits
            return {
                "requests": self.requests,
                "hits": self.hits,
                "misses": misses,
                "llm_calls": misses - self.coalesced,
                "coalesced": self.coalesced,
                "shed": self.shed,
                "cost_usd": round(self.cost_usd, 6),
                "errors": self.errors,
                "tokens": self.tokens,
                "hit_rate": round(self.hits / self.requests, 4) if self.requests else 0.0,
                "p50_ms": round(float(np.percentile(lat, 50)), 2) if lat.size else 0.0,
                "p95_ms": round(float(np.percentile(lat, 95)), 2) if lat.size else 0.0,
            }
