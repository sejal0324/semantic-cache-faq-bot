import threading

import numpy as np
import pytest

from semcache.embeddings import l2_normalize
from semcache.store import FaissStore, InMemoryNumpyStore, VectorStore, make_store

DIM = 8


def vec(*xs: float) -> np.ndarray:
    v = np.zeros((1, DIM), dtype=np.float32)
    v[0, : len(xs)] = xs
    return l2_normalize(v)[0]


@pytest.fixture(params=["memory", "faiss"])
def store(request: pytest.FixtureRequest) -> VectorStore:
    return make_store(request.param, DIM)


def test_empty_search(store: VectorStore) -> None:
    assert store.search(vec(1), k=3) == []
    assert len(store) == 0


def test_add_and_search_sorted_by_score(store: VectorStore) -> None:
    store.add(1, vec(1, 0))
    store.add(2, vec(1, 1))
    store.add(3, vec(0, 1))
    results = store.search(vec(1, 0.1), k=3)
    assert [k for k, _ in results] == [1, 2, 3]
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(float(vec(1, 0.1) @ vec(1, 0)), abs=1e-5)


def test_k_larger_than_size(store: VectorStore) -> None:
    store.add(7, vec(1))
    assert len(store.search(vec(1), k=10)) == 1


def test_remove(store: VectorStore) -> None:
    store.add(1, vec(1))
    store.add(2, vec(0, 1))
    store.remove(1)
    assert [k for k, _ in store.search(vec(1), k=5)] == [2]
    assert len(store) == 1
    store.remove(999)  # unknown key is a no-op


def test_add_existing_key_replaces(store: VectorStore) -> None:
    store.add(1, vec(1))
    store.add(1, vec(0, 1))
    assert len(store) == 1
    assert store.search(vec(0, 1), k=1)[0][1] == pytest.approx(1.0, abs=1e-5)


def test_dimension_mismatch_rejected(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="dimension"):
        store.add(1, np.ones(DIM + 1, dtype=np.float32))


def test_concurrent_adds_are_safe(store: VectorStore) -> None:
    def worker(offset: int) -> None:
        for i in range(200):
            store.add(offset + i, vec(1, i % 5))

    threads = [threading.Thread(target=worker, args=(t * 1000,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == 800


def test_make_store_types() -> None:
    assert isinstance(make_store("memory", 4), InMemoryNumpyStore)
    assert isinstance(make_store("faiss", 4), FaissStore)
    with pytest.raises(ValueError):
        make_store("bogus", 4)  # type: ignore[arg-type]
