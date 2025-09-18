"""Fallback summarizer approximating LongLLMLingua."""

from __future__ import annotations

from typing import Iterable

from ..models import RetrievedChunk


class LongLLMLinguaSummarizer:
    def summarize(self, chunks: Iterable[RetrievedChunk], target_tokens: int = 300) -> str:
        text = " ".join(chunk.text for chunk in chunks)
        words = text.split()
        return " ".join(words[:target_tokens])


__all__ = ["LongLLMLinguaSummarizer"]
