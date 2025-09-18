"""Dynamic document indexing with PII redaction and graph hooks."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from ..chunking import DocumentChunker
from ..embeddings import EmbeddingBackend, RandomEmbeddingBackend
from ..models import Chunk
from .pii import redact_pii


class InMemoryIndex:
    """Stores chunks, embeddings and statistics for BM25 scoring."""

    def __init__(self, embedding_backend: EmbeddingBackend | None = None) -> None:
        self.embedding_backend = embedding_backend or RandomEmbeddingBackend()
        self.chunks: Dict[str, Chunk] = {}
        self.doc_chunks: Dict[str, List[str]] = defaultdict(list)
        self.chunk_embeddings: List[List[float]] = []
        self.chunk_ids: List[str] = []
        self.term_freqs: Dict[str, Counter[str]] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.total_chunks: int = 0

    def add_chunk(self, chunk: Chunk) -> None:
        if chunk.chunk_id in self.chunks:
            return
        self.chunks[chunk.chunk_id] = chunk
        self.doc_chunks[chunk.doc_id].append(chunk.chunk_id)
        vector = self.embedding_backend.embed_query(chunk.text)
        self.chunk_embeddings.append(vector)
        self.chunk_ids.append(chunk.chunk_id)
        tokens = _tokenize(chunk.text)
        counter = Counter(tokens)
        self.term_freqs[chunk.chunk_id] = counter
        for token in counter:
            self.doc_freqs[token] += 1
        self.total_chunks += 1

    def embedding_matrix(self) -> List[List[float]]:
        return list(self.chunk_embeddings)

    def clear(self) -> None:
        self.chunks.clear()
        self.doc_chunks.clear()
        self.chunk_embeddings.clear()
        self.chunk_ids.clear()
        self.term_freqs.clear()
        self.doc_freqs.clear()
        self.total_chunks = 0


class DynamicIndexer:
    """High-level orchestrator handling chunking, PII redaction and indexing."""

    def __init__(
        self,
        chunker: DocumentChunker | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        graph_adapter=None,
    ) -> None:
        self.chunker = chunker or DocumentChunker()
        self.index = InMemoryIndex(embedding_backend=embedding_backend)
        self.graph_adapter = graph_adapter
        self._doc_counter = itertools.count()

    def index_documents(self, documents: Iterable[str]) -> int:
        count = 0
        for text in documents:
            doc_id = f"doc-{next(self._doc_counter)}"
            count += self._ingest_document(doc_id, text)
        return count

    def upsert_documents(self, documents: Iterable[str]) -> int:
        return self.index_documents(documents)

    def _ingest_document(self, doc_id: str, text: str) -> int:
        sanitized = redact_pii(text)
        chunks = self.chunker.chunk(doc_id, sanitized)
        for chunk in chunks:
            self.index.add_chunk(chunk)
            if self.graph_adapter:
                self.graph_adapter.index_chunk(chunk)
        return len(chunks)

    def bm25_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if not query_tokens:
            return scores
        total_chunks = max(1, self.index.total_chunks)
        avg_doc_len = sum(sum(counter.values()) for counter in self.index.term_freqs.values())
        avg_doc_len = avg_doc_len / total_chunks if total_chunks else 0
        k1 = 1.5
        b = 0.75
        for chunk_id, counter in self.index.term_freqs.items():
            doc_len = sum(counter.values())
            score = 0.0
            for token in query_tokens:
                if token not in counter:
                    continue
                df = self.index.doc_freqs.get(token, 1)
                numerator = max((total_chunks - df + 0.5) / (df + 0.5), 1e-6)
                idf = math.log(numerator)
                tf = counter[token]
                denom = tf + k1 * (1 - b + b * doc_len / (avg_doc_len or 1))
                score += idf * ((tf * (k1 + 1)) / denom)
            if score > 0:
                scores[chunk_id] = float(score)
        return scores


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in text.split() if token]


__all__ = ["DynamicIndexer", "InMemoryIndex", "_tokenize"]
