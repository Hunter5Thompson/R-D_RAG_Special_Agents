"""Minimal GraphRAG adapter using simple adjacency maps."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Set

from ..models import Chunk, RetrievedChunk


def _extract_terms(text: str) -> List[str]:
    words = [word.strip(".,;:()[]{}<>!?\"") for word in text.split()]
    return [word.lower() for word in words if len(word) > 4]


class GraphRAGAdapter:
    """Constructs a lightweight knowledge graph of chunk terms."""

    def __init__(self) -> None:
        self.chunk_terms: Dict[str, List[str]] = {}
        self.term_to_chunks: Dict[str, Set[str]] = defaultdict(set)

    def index_chunk(self, chunk: Chunk) -> None:
        terms = _extract_terms(chunk.text)
        self.chunk_terms[chunk.chunk_id] = terms
        for term in terms:
            self.term_to_chunks[term].add(chunk.chunk_id)

    def remove_chunk(self, chunk_id: str) -> None:
        terms = self.chunk_terms.pop(chunk_id, [])
        for term in terms:
            bucket = self.term_to_chunks.get(term)
            if not bucket:
                continue
            bucket.discard(chunk_id)
            if not bucket:
                self.term_to_chunks.pop(term, None)

    def retrieve_via_graph(
        self,
        query: str,
        chunk_lookup: Dict[str, Chunk],
        top_k: int = 3,
    ) -> List[RetrievedChunk]:
        terms = _extract_terms(query)
        if not terms:
            return []
        scores = Counter()
        for term in terms:
            for chunk_id in self.term_to_chunks.get(term, []):
                scores[chunk_id] += 1.0
        ranked = scores.most_common(top_k)
        results = []
        for chunk_id, score in ranked:
            chunk = chunk_lookup[chunk_id]
            results.append(
                RetrievedChunk(
                    **chunk.model_dump(),
                    retrieval_score=float(score),
                    via="graphrag",
                )
            )
        return results


__all__ = ["GraphRAGAdapter"]
