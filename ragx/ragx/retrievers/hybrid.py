"""Hybrid retriever combining lexical and dense scores."""

from __future__ import annotations

from typing import List

from ..embeddings import cosine_similarity
from ..models import RetrievedChunk
from ..index.dynamic import DynamicIndexer, _tokenize


class HybridRetriever:
    def __init__(self, indexer: DynamicIndexer, alpha: float = 0.3) -> None:
        self.indexer = indexer
        self.alpha = alpha

    def set_alpha(self, value: float) -> None:
        self.alpha = float(min(max(value, 0.0), 1.0))

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        top_k = top_k or 10
        query_tokens = _tokenize(query)
        bm25_scores = self.indexer.bm25_scores(query_tokens)
        matrix = self.indexer.index.embedding_matrix()
        if matrix:
            query_vec = self.indexer.index.embedding_backend.embed_query(query)
            dense_scores = cosine_similarity(query_vec, matrix)
        else:
            dense_scores = [0.0 for _ in self.indexer.index.chunk_ids]

        dense_scores = _normalize_dense(dense_scores)
        bm25_array = _scores_to_array(bm25_scores, self.indexer.index.chunk_ids)
        bm25_array = _normalize_array(bm25_array)

        combined = [
            self.alpha * dense + (1 - self.alpha) * bm25
            for dense, bm25 in zip(dense_scores, bm25_array)
        ]
        ranked_indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]
        results: List[RetrievedChunk] = []
        for idx in ranked_indices:
            chunk_id = self.indexer.index.chunk_ids[idx]
            chunk = self.indexer.index.chunks[chunk_id]
            results.append(
                RetrievedChunk(
                    **chunk.model_dump(),
                    retrieval_score=float(combined[idx]),
                    rerank_score=None,
                    via="hybrid",
                )
            )
        return results


def _scores_to_array(scores: dict[str, float], ordered_ids: List[str]) -> List[float]:
    return [scores.get(chunk_id, 0.0) for chunk_id in ordered_ids]


def _normalize_array(values: List[float]) -> List[float]:
    if not values:
        return values
    max_value = max(values)
    if max_value <= 0:
        return [0.0 for _ in values]
    return [value / max_value for value in values]


def _normalize_dense(values: List[float]) -> List[float]:
    if not values:
        return values
    return [(value + 1.0) / 2.0 for value in values]


__all__ = ["HybridRetriever"]
