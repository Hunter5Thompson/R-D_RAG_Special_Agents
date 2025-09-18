"""Tensor retriever placeholder to hook ColBERT/ColPali implementations."""

from __future__ import annotations

from typing import Iterable, List

from ..models import Chunk, RetrievedChunk


class TensorRetrieverInterface:
    def __init__(self) -> None:
        self.indexed_chunks: List[Chunk] = []

    def index(self, chunks: Iterable[Chunk]) -> None:
        self.indexed_chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        # Placeholder: return empty list. Backend can override.
        return []


__all__ = ["TensorRetrieverInterface"]
