"""Embedding abstraction. Vectors are always L2-normalised, so dot product == cosine similarity."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from semcache.config import Settings

FloatMatrix = npt.NDArray[np.float32]
_WORD = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> FloatMatrix: ...


def l2_normalize(matrix: npt.NDArray[Any]) -> FloatMatrix:
    m = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: FloatMatrix = (m / norms).astype(np.float32)
    return out


class FakeEmbedder:
    """Deterministic hashing embedder (word unigrams + character trigrams).

    Not a language model: it only knows surface overlap, so paraphrases that share words
    score higher than unrelated questions. Good enough to make unit tests meaningful and to
    run the pipeline offline. Uses blake2b, not `hash()`, so results are stable across runs.
    """

    def __init__(self, dim: int = 256, *, latency_s: float = 0.0) -> None:
        self._dim = dim
        self._latency_s = latency_s

    @property
    def name(self) -> str:
        return f"fake-ngram-hash-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        words = _WORD.findall(text.lower())
        feats: list[tuple[str, float]] = [(f"w:{w}", 2.0) for w in words]
        for w in words:
            padded = f"<{w}>"
            feats.extend((f"c:{padded[i : i + 3]}", 1.0) for i in range(len(padded) - 2))
        return feats

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        if self._latency_s:
            time.sleep(self._latency_s)
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self._dim] += sign * weight
        return l2_normalize(out)


class SentenceTransformerEmbedder:
    """Real embeddings via sentence-transformers. The model downloads on first use."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no branch
            raise RuntimeError(
                'sentence-transformers is not installed: run pip install -e ".[models]"'
            ) from exc
        self._name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return l2_normalize(vecs)


def build_embedder(settings: Settings, model_name: str | None = None) -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder()
    return SentenceTransformerEmbedder(model_name or settings.embedding_model)
