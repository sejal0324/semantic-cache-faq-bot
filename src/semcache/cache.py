"""The semantic cache: nearest-neighbour lookup over past questions, with guards and hygiene.

Lookup returns a hit only when ALL of these hold for a candidate:
  1. cosine similarity >= threshold
  2. the entry has not expired (per-category TTL)
  3. the negation guard allows it (both questions negated, or neither)
Candidates are checked best-first (top_k), so a blocked best match can fall through to the
next one instead of turning straight into a miss.

Writes are refused when the question or answer contains user-specific data (PII), so one
customer's order ID or phone number can never be served to another customer, and when the
answer contains a link outside the allowlist (cache poisoning through write-back).

Each tenant gets its own vector store: isolation by construction, not by filtering.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from semcache.embeddings import Embedder
from semcache.guards import NegationGuard
from semcache.links import has_disallowed_link
from semcache.pii import PIIDetector
from semcache.store import StoreKind, Vector, VectorStore, make_store
from semcache.text import normalize

Clock = Callable[[], float]
_DEFAULT_GUARD = NegationGuard.default()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: int
    tenant: str
    question: str
    answer: str
    category: str
    created_at: float
    expires_at: float
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class LookupResult:
    hit: bool
    entry: CacheEntry | None
    similarity: float
    reason: str  # hit | exact | empty | below_threshold | expired | negation_mismatch


@dataclass(frozen=True, slots=True)
class StoreDecision:
    stored: bool
    reason: str  # stored | pii | empty_answer | unsafe_link | tenant_limit


@dataclass
class _TenantSpace:
    store: VectorStore
    entries: OrderedDict[int, CacheEntry] = field(default_factory=OrderedDict)
    by_question: dict[str, int] = field(default_factory=dict)


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.85,
        store_kind: StoreKind = "faiss",
        ttl_s_by_category: Mapping[str, float] | None = None,
        default_ttl_s: float = 7 * 86_400.0,
        negation_guard: NegationGuard | None = _DEFAULT_GUARD,
        pii: PIIDetector | None = None,
        clock: Clock = time.time,
        max_entries_per_tenant: int = 10_000,
        max_tenants: int = 1_000,
        allowed_link_domains: frozenset[str] = frozenset(),
        top_k: int = 3,
    ) -> None:
        self._embedder = embedder
        self.threshold = threshold
        self._store_kind: StoreKind = store_kind
        self._ttl = dict(ttl_s_by_category or {})
        self._default_ttl = default_ttl_s
        self.negation_guard = negation_guard
        self._pii = pii or PIIDetector()
        self._clock = clock
        self._max_entries = max_entries_per_tenant
        self._max_tenants = max_tenants
        self._allowed_links = allowed_link_domains
        self._top_k = top_k
        self._spaces: dict[str, _TenantSpace] = {}
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self.negation_catches: deque[tuple[str, str]] = deque(maxlen=100)

    # -- properties ---------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._threshold = value

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def tenant_count(self) -> int:
        with self._lock:
            return len(self._spaces)

    def size(self, tenant: str) -> int:
        with self._lock:
            space = self._spaces.get(tenant)
            return len(space.entries) if space else 0

    # -- core ---------------------------------------------------------------------------

    def embed(self, question: str) -> Vector:
        vector: Vector = self._embedder.embed([normalize(question)])[0]
        return vector

    def lookup_exact(self, tenant: str, question: str) -> LookupResult | None:
        """O(1) exact-text match (case/space-insensitive). No embedding needed."""
        dedup_key = normalize(question).lower()
        with self._lock:
            space = self._spaces.get(tenant)
            key = space.by_question.get(dedup_key) if space else None
            if space is None or key is None:
                return None
            entry = space.entries[key]
            if entry.expires_at <= self._clock():
                self._evict(space, key)
                return None
            space.entries.move_to_end(key)
            return LookupResult(True, entry, 1.0, "exact")

    def lookup(self, tenant: str, question: str, *, vector: Vector | None = None) -> LookupResult:
        q = normalize(question)
        with self._lock:
            space = self._spaces.get(tenant)
            if space is None or not space.entries:
                return LookupResult(False, None, 0.0, "empty")
        v = vector if vector is not None else self.embed(q)
        with self._lock:
            results = space.store.search(v, self._top_k)
            best = results[0][1] if results else 0.0
            reason = "below_threshold"
            now = self._clock()
            for key, score in results:
                if score < self._threshold:
                    break
                entry = space.entries.get(key)
                if entry is None:  # pragma: no cover - store and entries are updated together
                    continue
                if entry.expires_at <= now:
                    self._evict(space, key)
                    reason = "expired"
                    continue
                if self.negation_guard and not self.negation_guard.allows(q, entry.question):
                    self.negation_catches.append((q, entry.question))
                    reason = "negation_mismatch"
                    continue
                space.entries.move_to_end(key)
                return LookupResult(True, entry, score, "hit")
            return LookupResult(False, None, best, reason)

    def put(
        self,
        tenant: str,
        question: str,
        answer: str,
        *,
        category: str = "general",
        source_id: str | None = None,
        vector: Vector | None = None,
    ) -> StoreDecision:
        q = normalize(question)
        if not answer.strip():
            return StoreDecision(False, "empty_answer")
        if self._pii.contains_pii(q) or self._pii.contains_pii(answer):
            return StoreDecision(False, "pii")
        if has_disallowed_link(answer, self._allowed_links):
            return StoreDecision(False, "unsafe_link")
        v = vector if vector is not None else self.embed(q)
        now = self._clock()
        ttl = self._ttl.get(category, self._default_ttl)
        dedup_key = q.lower()
        with self._lock:
            space = self._spaces.get(tenant)
            if space is None:
                if len(self._spaces) >= self._max_tenants:
                    return StoreDecision(False, "tenant_limit")
                space = _TenantSpace(store=make_store(self._store_kind, int(v.shape[0])))
                self._spaces[tenant] = space
            old = space.by_question.get(dedup_key)
            if old is not None:
                self._evict(space, old)
            key = next(self._ids)
            entry = CacheEntry(key, tenant, q, answer, category, now, now + ttl, source_id)
            space.store.add(key, v)
            space.entries[key] = entry
            space.by_question[dedup_key] = key
            while len(space.entries) > self._max_entries:
                oldest = next(iter(space.entries))
                self._evict(space, oldest)
        return StoreDecision(True, "stored")

    def purge(
        self, tenant: str, *, source_id: str | None = None, generated_only: bool = False
    ) -> int:
        """Invalidate entries: one FAQ's answers (`source_id`), only LLM-generated answers, or
        the whole tenant. Use it when a policy changes instead of waiting for the TTL."""
        with self._lock:
            space = self._spaces.get(tenant)
            if space is None:
                return 0
            doomed = [
                key
                for key, e in space.entries.items()
                if (source_id is None or e.source_id == source_id)
                and (not generated_only or e.source_id is None)
            ]
            for key in doomed:
                self._evict(space, key)
            return len(doomed)

    @staticmethod
    def _evict(space: _TenantSpace, key: int) -> None:
        entry = space.entries.pop(key, None)
        space.store.remove(key)
        if entry is not None:
            space.by_question.pop(entry.question.lower(), None)
