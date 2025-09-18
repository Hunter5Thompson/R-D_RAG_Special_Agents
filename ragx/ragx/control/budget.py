"""Token budget gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..models import RetrievedChunk

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


@dataclass
class BudgetResult:
    context: str
    strategy: str


class ContextBudgeter:
    def __init__(self, max_tokens: int, primary, fallback) -> None:
        self.max_tokens = max_tokens
        self.primary = primary
        self.fallback = fallback
        self._tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken else None

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return len(text.split())

    def enforce(self, context: str, chunks: Iterable[RetrievedChunk]) -> BudgetResult:
        tokens = self._count_tokens(context)
        if tokens <= self.max_tokens:
            return BudgetResult(context=context, strategy="none")
        primary_summary = self.primary.summarize(chunks, target_tokens=self.max_tokens)
        if self._count_tokens(primary_summary) <= self.max_tokens:
            return BudgetResult(context=primary_summary, strategy="primary")
        fallback_summary = self.fallback.summarize(chunks, target_tokens=self.max_tokens)
        truncated = self._truncate(fallback_summary)
        return BudgetResult(context=truncated, strategy="fallback")

    def _truncate(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        budget_words = words[: self.max_tokens]
        return " ".join(budget_words)


__all__ = ["ContextBudgeter", "BudgetResult"]
