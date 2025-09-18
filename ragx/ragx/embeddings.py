"""Embedding backends using deterministic pseudo-random vectors."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List


class EmbeddingBackend:
    """Abstract embedding backend."""

    dimension: int = 128

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:  # pragma: no cover - interface
        raise NotImplementedError


class RandomEmbeddingBackend(EmbeddingBackend):
    """Deterministic embedding backend relying on hashing."""

    def __init__(self, dimension: int = 128, global_seed: int = 13) -> None:
        self.dimension = dimension
        self.global_seed = global_seed

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        values: List[float] = []
        for index in range(self.dimension):
            digest = hashlib.sha256(f"{self.global_seed}:{index}:{text}".encode("utf-8")).digest()
            raw = int.from_bytes(digest[:4], "big", signed=False)
            normalized = (raw % 2000) / 1000.0 - 1.0
            values.append(normalized)
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]


def cosine_similarity(query: List[float], matrix: List[List[float]]) -> List[float]:
    scores: List[float] = []
    for vector in matrix:
        score = sum(q * v for q, v in zip(query, vector))
        scores.append(score)
    return scores


__all__ = ["EmbeddingBackend", "RandomEmbeddingBackend", "cosine_similarity"]
