"""Faithfulness critic heuristics."""

from __future__ import annotations

import json
from typing import Iterable

from ..models import RetrievedChunk


class Critic:
    def evaluate(self, response: str, context: str, chunks: Iterable[RetrievedChunk]) -> str:
        if not response:
            return "LOW_CONFIDENCE"
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return "LOW_CONFIDENCE"

        quotes = payload.get("quotes", [])
        answer = payload.get("answer", "")
        if not quotes:
            return "LOW_CONFIDENCE"
        context_lower = context.lower()
        for quote in quotes:
            if quote and quote.lower() in context_lower:
                return "OK"
        if answer and answer.lower() in context_lower:
            return "OK"
        return "LOW_CONFIDENCE"


__all__ = ["Critic"]
