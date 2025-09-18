"""Generator producing schema-aligned responses."""

from __future__ import annotations

import json
from typing import Iterable

from ..models import RetrievedChunk


class Generator:
    def generate(self, query: str, context: str, chunks: Iterable[RetrievedChunk]) -> str:
        chunk_list = list(chunks)
        if not context.strip() or not chunk_list:
            payload = {
                "answer": "Keine passenden Informationen gefunden.",
                "quotes": [],
                "disclaimer": "LOW_CONFIDENCE",
            }
            return json.dumps(payload, ensure_ascii=False)

        best_chunk = max(chunk_list, key=lambda c: c.rerank_score or c.retrieval_score)
        first_sentence = best_chunk.text.split(". ")[0].strip()
        payload = {
            "answer": first_sentence,
            "quotes": [first_sentence[:120]],
        }
        if len(chunk_list) > 1:
            payload["quotes"].append(chunk_list[1].text.split(". ")[0].strip())
        return json.dumps(payload, ensure_ascii=False)


__all__ = ["Generator"]
