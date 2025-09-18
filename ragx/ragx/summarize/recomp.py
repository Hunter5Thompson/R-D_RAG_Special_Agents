"""Abstractive summarizer stub representing RECOMP."""

from __future__ import annotations

from typing import Iterable

from ..models import RetrievedChunk


class RecompSummarizer:
    def summarize(self, chunks: Iterable[RetrievedChunk], target_tokens: int = 400) -> str:
        sentences = []
        for chunk in chunks:
            for sentence in chunk.text.split(". "):
                sentence = sentence.strip()
                if sentence and sentence not in sentences:
                    sentences.append(sentence)
        summary = []
        word_budget = target_tokens
        for sentence in sentences:
            words = sentence.split()
            if len(words) > word_budget:
                break
            summary.append(sentence)
            word_budget -= len(words)
            if word_budget <= 0:
                break
        return ". ".join(summary)


__all__ = ["RecompSummarizer"]
