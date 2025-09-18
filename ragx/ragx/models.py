"""Data models for RAGX implemented with Pydantic BaseModel."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pragma: no cover - exercised indirectly in tests
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - fallback when dependency unavailable
    from ._compat import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(Chunk):
    retrieval_score: float = 0.0
    rerank_score: Optional[float] = None
    via: Optional[str] = None


class ContextBundle(BaseModel):
    bundle_id: str
    task: str
    constraints: List[str] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    glossary: Dict[str, str] = Field(default_factory=dict)
    provenance: List[Dict[str, Any]] = Field(default_factory=list)
    ttl: Optional[str] = "7d"
    version: int = 1


class QueryResult(BaseModel):
    query: str
    needs_retrieval: bool
    query_type: str
    retrieved: List[RetrievedChunk] = Field(default_factory=list)
    context: str = ""
    response: str = ""
    meta: Dict[str, str] = Field(default_factory=dict)


__all__ = [
    "Chunk",
    "RetrievedChunk",
    "ContextBundle",
    "QueryResult",
]
