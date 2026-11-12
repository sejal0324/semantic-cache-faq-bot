import sys
import types

import numpy as np
import pytest

from semcache.config import Settings
from semcache.embeddings import (
    FakeEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
    l2_normalize,
)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def test_l2_normalize_handles_zero_rows() -> None:
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(m)
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])
    assert out.dtype == np.float32


def test_fake_embedder_shape_norm_and_determinism() -> None:
    e = FakeEmbedder(dim=128)
    a = e.embed(["When will I get my refund?", "hello"])
    b = FakeEmbedder(dim=128).embed(["When will I get my refund?", "hello"])
    assert a.shape == (2, 128)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, b), "must not depend on Python's salted hash()"
    assert e.dim == 128
    assert "fake" in e.name


def test_fake_embedder_paraphrase_closer_than_unrelated() -> None:
    e = FakeEmbedder()
    q, para, other = e.embed(
        ["When will I get my refund?", "when will my refund come", "Do you sell gift cards?"]
    )
    assert cos(q, para) > cos(q, other)


def test_fake_embedder_empty_input() -> None:
    assert FakeEmbedder(dim=8).embed([]).shape == (0, 8)


def test_sentence_transformer_embedder_uses_library(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class StubModel:
        def __init__(self, name: str, device: str | None = None) -> None:
            calls["name"] = name

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts, normalize_embeddings, convert_to_numpy):  # type: ignore[no-untyped-def]
            calls["normalize"] = normalize_embeddings
            return np.ones((len(texts), 3), dtype=np.float64)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    e = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    out = e.embed(["a", "b"])
    assert calls == {"name": "all-MiniLM-L6-v2", "normalize": True}
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert e.dim == 3
    assert e.name == "all-MiniLM-L6-v2"
    assert e.embed([]).shape == (0, 3)


def test_sentence_transformer_missing_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[models\]\""):
        SentenceTransformerEmbedder("x")


def test_build_embedder_fake() -> None:
    assert isinstance(build_embedder(Settings(_env_file=None, embedder="fake")), FakeEmbedder)


def test_fake_embedder_cpu_bound_latency_spins() -> None:
    import time

    e = FakeEmbedder(latency_s=0.02, cpu_bound=True)
    start = time.process_time()
    e.embed(["x"])
    assert time.process_time() - start >= 0.015, "CPU-bound mode must burn CPU, not sleep"


def test_build_embedder_fake_uses_latency_settings() -> None:
    s = Settings(_env_file=None, embedder="fake", fake_embed_latency_s=0.01)
    e = build_embedder(s)
    assert isinstance(e, FakeEmbedder) and e.latency_s == 0.01
