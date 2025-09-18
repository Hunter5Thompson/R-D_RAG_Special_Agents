"""Stub TILDEv2 reranker focusing on term coverage."""

from __future__ import annotations

from typing import Iterable, List

from ..index.dynamic import _tokenize
from ..models import RetrievedChunk


class TILDEV2Reranker:
    def rerank(self, chunks: Iterable[RetrievedChunk], query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_tokens = _tokenize(query)
        scored = []
        for chunk in chunks:
            chunk_tokens = _tokenize(chunk.text)
            coverage = sum(chunk_tokens.count(token) for token in query_tokens)
            score = coverage / max(1, len(chunk_tokens))
            scored.append(chunk.model_copy(update={"rerank_score": float(score)}))
        scored.sort(key=lambda c: (c.rerank_score or 0.0, c.retrieval_score), reverse=True)
        return scored[:top_k]


__all__ = ["TILDEV2Reranker"]
