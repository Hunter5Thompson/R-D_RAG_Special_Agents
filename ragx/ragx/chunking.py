"""Utilities to split documents into chunks."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable, List

from .models import Chunk


def _split_sentences(text: str) -> List[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


@dataclass
class ChunkerConfig:
    window_size: int = 3
    stride: int = 2
    small_to_big: bool = True


class DocumentChunker:
    """Sentence-based chunker supporting sliding windows and small-to-big merging."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._counter = itertools.count()

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        if self.config.small_to_big and len(sentences) <= self.config.window_size:
            chunk_id = f"{doc_id}::chunk::{next(self._counter)}"
            return [Chunk(chunk_id=chunk_id, doc_id=doc_id, text=" ".join(sentences))]

        window = max(1, self.config.window_size)
        stride = max(1, self.config.stride)
        chunks: List[Chunk] = []
        for start in range(0, len(sentences), stride):
            window_sentences = sentences[start : start + window]
            if not window_sentences:
                break
            chunk_id = f"{doc_id}::chunk::{next(self._counter)}"
            chunk_text = " ".join(window_sentences)
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text))
            if start + window >= len(sentences):
                break
        return chunks

    def chunk_many(self, documents: Iterable[tuple[str, str]]) -> List[Chunk]:
        return [chunk for doc_id, text in documents for chunk in self.chunk(doc_id, text)]


__all__ = ["DocumentChunker", "ChunkerConfig"]
