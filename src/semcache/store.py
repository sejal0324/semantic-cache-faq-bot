"""Vector stores. Both use inner product on L2-normalised vectors (== cosine similarity).

- InMemoryNumpyStore: brute force with numpy. Simple, used in tests.
- FaissStore: FAISS IndexFlatIP wrapped in IndexIDMap2 so entries can be removed (TTL, LRU).
  Flat (exact) search is the right call below ~100k vectors; switch to IVF/HNSW above that.
"""

from __future__ import annotations

import threading
from typing import Literal, Protocol

import faiss
import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]
StoreKind = Literal["faiss", "memory"]


class VectorStore(Protocol):
    def add(self, key: int, vector: Vector) -> None: ...

    def remove(self, key: int) -> None: ...

    def search(self, vector: Vector, k: int) -> list[tuple[int, float]]: ...

    def __len__(self) -> int: ...


def _as_row(vector: Vector, dim: int) -> Vector:
    v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    if v.shape[1] != dim:
        raise ValueError(f"vector dimension {v.shape[1]} != store dimension {dim}")
    return v


class InMemoryNumpyStore:
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._vectors: dict[int, Vector] = {}
        self._lock = threading.Lock()

    def add(self, key: int, vector: Vector) -> None:
        row = _as_row(vector, self._dim)[0]
        with self._lock:
            self._vectors[key] = row

    def remove(self, key: int) -> None:
        with self._lock:
            self._vectors.pop(key, None)

    def search(self, vector: Vector, k: int) -> list[tuple[int, float]]:
        q = _as_row(vector, self._dim)[0]
        with self._lock:
            if not self._vectors:
                return []
            keys = list(self._vectors)
            matrix = np.stack([self._vectors[key] for key in keys])
        scores = matrix @ q
        order = np.argsort(-scores)[:k]
        return [(keys[i], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self._vectors)


class FaissStore:
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._keys: set[int] = set()
        self._lock = threading.Lock()

    def add(self, key: int, vector: Vector) -> None:
        row = _as_row(vector, self._dim)
        ids = np.array([key], dtype=np.int64)
        with self._lock:
            if key in self._keys:
                self._index.remove_ids(ids)  # type: ignore[arg-type]  # numpy ids accepted at runtime
            self._index.add_with_ids(row, ids)
            self._keys.add(key)

    def remove(self, key: int) -> None:
        with self._lock:
            if key in self._keys:
                self._index.remove_ids(np.array([key], dtype=np.int64))  # type: ignore[arg-type]
                self._keys.discard(key)

    def search(self, vector: Vector, k: int) -> list[tuple[int, float]]:
        q = _as_row(vector, self._dim)
        with self._lock:
            n = int(self._index.ntotal)
            if n == 0:
                return []
            scores, ids = self._index.search(q, min(k, n))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def __len__(self) -> int:
        return len(self._keys)


def make_store(kind: StoreKind, dim: int) -> VectorStore:
    if kind == "faiss":
        return FaissStore(dim)
    if kind == "memory":
        return InMemoryNumpyStore(dim)
    raise ValueError(f"unknown vector store: {kind}")
