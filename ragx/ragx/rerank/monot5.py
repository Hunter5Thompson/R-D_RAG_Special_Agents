"""Stub monoT5 reranker with deterministic scoring."""

from __future__ import annotations

from typing import Iterable, List

from ..index.dynamic import _tokenize
from ..models import RetrievedChunk


class MonoT5Reranker:
    def rerank(self, chunks: Iterable[RetrievedChunk], query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_tokens = set(_tokenize(query))
        scored = []
        for chunk in chunks:
            chunk_tokens = set(_tokenize(chunk.text))
            overlap = len(query_tokens & chunk_tokens)
            score = overlap / max(1, len(query_tokens))
            scored.append(chunk.model_copy(update={"rerank_score": float(score)}))
        scored.sort(key=lambda c: (c.rerank_score or 0.0, c.retrieval_score), reverse=True)
        return scored[:top_k]


__all__ = ["MonoT5Reranker"]
