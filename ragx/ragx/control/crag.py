"""CRAG evaluator heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..models import RetrievedChunk


@dataclass
class CRAGEvaluation:
    score: float
    coverage: float
    diversity: float
    contradiction: float
    action: str


class CRAGEvaluator:
    def __init__(self, min_score: float = 0.55) -> None:
        self.min_score = min_score

    def evaluate(self, chunks: Iterable[RetrievedChunk]) -> CRAGEvaluation:
        chunk_list = list(chunks)
        if not chunk_list:
            return CRAGEvaluation(0.0, 0.0, 0.0, 0.0, "retry")

        coverage = sum(chunk.retrieval_score for chunk in chunk_list) / len(chunk_list)
        unique_docs = {chunk.doc_id for chunk in chunk_list}
        diversity = len(unique_docs) / len(chunk_list)
        contradiction = 0.0  # Placeholder for NLI hook
        score = (coverage + diversity - contradiction) / 2
        action = "retry" if score < self.min_score else "ok"
        return CRAGEvaluation(score, coverage, diversity, contradiction, action)


__all__ = ["CRAGEvaluator", "CRAGEvaluation"]
