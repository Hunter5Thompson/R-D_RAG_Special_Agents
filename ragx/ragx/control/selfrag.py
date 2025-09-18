"""Self-RAG style query rewriting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..index.dynamic import _tokenize
from ..models import RetrievedChunk


@dataclass
class RewriteResult:
    final_query: str
    rewrites: List[str]


class SelfRAGRewriter:
    def __init__(self, max_loops: int = 2) -> None:
        self.max_loops = max_loops

    def propose(self, query: str, chunks: Iterable[RetrievedChunk]) -> str:
        query_tokens = set(_tokenize(query))
        additions: List[str] = []
        for chunk in chunks:
            for token in _tokenize(chunk.text):
                if len(token) <= 4:
                    continue
                if token in query_tokens or token in additions:
                    continue
                additions.append(token)
                if len(additions) >= 3:
                    break
            if len(additions) >= 3:
                break
        if not additions:
            return query
        return f"{query} {' '.join(additions[:2])}".strip()

    def rewrite_loop(self, query: str, chunks: Iterable[RetrievedChunk]) -> RewriteResult:
        rewrites: List[str] = []
        current_query = query
        for _ in range(self.max_loops):
            new_query = self.propose(current_query, chunks)
            if new_query == current_query:
                break
            rewrites.append(new_query)
            current_query = new_query
        return RewriteResult(final_query=current_query, rewrites=rewrites)


__all__ = ["SelfRAGRewriter", "RewriteResult"]
