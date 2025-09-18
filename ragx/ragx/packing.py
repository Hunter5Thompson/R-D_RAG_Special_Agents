"""Context repacking strategies."""

from __future__ import annotations

from typing import Iterable, List

from .models import RetrievedChunk


class Repacker:
    def pack(self, chunks: Iterable[RetrievedChunk], mode: str = "reverse") -> str:
        chunk_list = list(chunks)
        if not chunk_list:
            return ""
        if mode == "reverse":
            ordered = sorted(chunk_list, key=lambda c: c.retrieval_score, reverse=True)
        elif mode == "forward":
            ordered = sorted(chunk_list, key=lambda c: c.retrieval_score)
        elif mode == "sides":
            ordered = self._sides_order(chunk_list)
        else:
            ordered = chunk_list
        return "\n\n".join(chunk.text for chunk in ordered)

    @staticmethod
    def _sides_order(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        sorted_chunks = sorted(chunks, key=lambda c: c.retrieval_score, reverse=True)
        ordered: List[RetrievedChunk] = []
        left = 0
        right = len(sorted_chunks) - 1
        toggle = True
        while left <= right:
            if toggle:
                ordered.append(sorted_chunks[left])
                left += 1
            else:
                ordered.append(sorted_chunks[right])
                right -= 1
            toggle = not toggle
        return ordered


__all__ = ["Repacker"]
