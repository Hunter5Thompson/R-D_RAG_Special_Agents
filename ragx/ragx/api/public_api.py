"""Public API for RAGX."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List

from ..config import ConfigManager
from ..control.budget import ContextBudgeter
from ..control.crag import CRAGEvaluator
from ..control.planner import Planner
from ..control.selfrag import SelfRAGRewriter
from ..generation.critic import Critic
from ..generation.generator import Generator
from ..index.dynamic import DynamicIndexer
from ..models import QueryResult, RetrievedChunk
from ..packing import Repacker
from ..retrievers.graphrag_adapter import GraphRAGAdapter
from ..retrievers.hybrid import HybridRetriever
from ..rerank.monot5 import MonoT5Reranker
from ..rerank.tildev2 import TILDEV2Reranker
from ..summarize.longllmlingua import LongLLMLinguaSummarizer
from ..summarize.recomp import RecompSummarizer
from ..embeddings import RandomEmbeddingBackend
from ..chunking import DocumentChunker
from ..retrievers.tensor_iface import TensorRetrieverInterface


@dataclass
class _AppState:
    config: ConfigManager
    indexer: DynamicIndexer
    retriever: HybridRetriever
    tensor_iface: TensorRetrieverInterface
    graph: GraphRAGAdapter
    planner: Planner
    repacker: Repacker
    generator: Generator
    critic: Critic
    crag: CRAGEvaluator
    selfrag: SelfRAGRewriter


_graph_adapter = GraphRAGAdapter()
_indexer = DynamicIndexer(
    chunker=DocumentChunker(), embedding_backend=RandomEmbeddingBackend(), graph_adapter=_graph_adapter
)

_state = _AppState(
    config=ConfigManager(),
    graph=_graph_adapter,
    indexer=_indexer,
    retriever=HybridRetriever(_indexer),
    tensor_iface=TensorRetrieverInterface(),
    planner=Planner(),
    repacker=Repacker(),
    generator=Generator(),
    critic=Critic(),
    crag=CRAGEvaluator(),
    selfrag=SelfRAGRewriter(),
)

_SUMMARIZERS = {
    "recomp": RecompSummarizer(),
    "longllmlingua": LongLLMLinguaSummarizer(),
}
_RERANKERS = {
    "monoT5": MonoT5Reranker(),
    "tildev2": TILDEV2Reranker(),
}


def index_documents(documents: List[str]) -> int:
    """Chunk -> Embed -> Index (Hybrid + Graph + TensorIF)."""

    count = _state.indexer.index_documents(documents)
    _refresh_tensor_index()
    return count


def upsert_documents(docs: List[str]) -> int:
    """Inkrementell upserten (Redaction+Index+KG+TensorIF)."""

    count = _state.indexer.upsert_documents(docs)
    _refresh_tensor_index()
    return count


def answer(query: str, profile: str = "balanced_efficiency") -> QueryResult:
    plan = ["retrieve", "rerank", "pack", "summarize", "budget", "generate", "critic"]
    return _run_pipeline(query, profile, plan)


def answer_agentic(query: str, profile: str = "balanced_efficiency") -> QueryResult:
    plan_steps = [step.name for step in _state.planner.plan(query)]
    return _run_pipeline(query, profile, plan_steps)


def set_alpha(value: float) -> None:
    _state.retriever.set_alpha(value)


def _run_pipeline(query: str, profile: str, plan: List[str]) -> QueryResult:
    config = _state.config.load(profile=profile)
    retrieval_top_k = config["retrieval"].get("top_k", 10)
    rerank_top_k = config["rerank"].get("top_k", 5)
    alpha = config["retrieval"].get("alpha", 0.3)
    _state.retriever.set_alpha(alpha)

    retrieved: List[RetrievedChunk] = []
    reranked: List[RetrievedChunk] = []
    context = ""
    response = ""
    rewrites: List[str] = []
    current_query = query
    primary = _SUMMARIZERS[config["summarization"].get("primary", "recomp")]
    fallback = _SUMMARIZERS[config["summarization"].get("fallback", "longllmlingua")]
    budgeter = ContextBudgeter(config["budget"].get("max_context_tokens", 2500), primary, fallback)
    critic_status = "PENDING"
    crag_score = 0.0
    budget_strategy = "none"
    critic_retries = 0

    for step in plan:
        if step == "retrieve":
            retrieved = _state.retriever.retrieve(current_query, top_k=retrieval_top_k)
            evaluation = _state.crag.evaluate(retrieved)
            crag_score = evaluation.score
            if evaluation.action == "retry":
                rewrite_result = _state.selfrag.rewrite_loop(current_query, retrieved)
                rewrites = rewrite_result.rewrites
                current_query = rewrite_result.final_query
                if rewrites:
                    retrieved = _state.retriever.retrieve(current_query, top_k=retrieval_top_k)
                    crag_score = _state.crag.evaluate(retrieved).score
        elif step == "graph_retrieve":
            additional = _state.graph.retrieve_via_graph(current_query, _state.indexer.index.chunks)
            known_ids = {chunk.chunk_id for chunk in retrieved}
            for chunk in additional:
                if chunk.chunk_id not in known_ids:
                    retrieved.append(chunk)
                    known_ids.add(chunk.chunk_id)
        elif step == "rerank":
            reranker_name = config["rerank"].get("method", "monoT5")
            reranker = _RERANKERS.get(reranker_name, MonoT5Reranker())
            reranked = reranker.rerank(retrieved, current_query, top_k=rerank_top_k)
        elif step == "pack":
            context = _state.repacker.pack(reranked, mode=config.get("packing", "reverse"))
        elif step == "summarize":
            # Summaries are produced during budget enforcement
            pass
        elif step == "budget":
            budget_result = budgeter.enforce(context, reranked)
            context = budget_result.context
            budget_strategy = budget_result.strategy
        elif step == "generate":
            response = _state.generator.generate(query, context, reranked)
        elif step == "critic":
            critic_status = _state.critic.evaluate(response, context, reranked)
            if critic_status == "LOW_CONFIDENCE":
                fallback_context = fallback.summarize(reranked, target_tokens=config["budget"].get("max_context_tokens", 2500))
                context = budgeter._truncate(fallback_context)
                response = _state.generator.generate(query, context, reranked)
                critic_status = _state.critic.evaluate(response, context, reranked)
                critic_retries += 1

    query_type = _state.planner.classify(query)
    return QueryResult(
        query=query,
        needs_retrieval=True,
        query_type=query_type,
        retrieved=reranked,
        context=context,
        response=response,
        meta={
            "profile": profile,
            "rewrites": str(len(rewrites)),
            "critic_status": critic_status,
            "alpha": f"{alpha:.2f}",
            "crag_score": f"{crag_score:.3f}",
            "budget_strategy": budget_strategy,
            "critic_retries": str(critic_retries),
        },
    )


def _refresh_tensor_index() -> None:
    _state.tensor_iface.index(_state.indexer.index.chunks.values())


def _reset_state_for_tests() -> None:  # pragma: no cover - used in unit tests
    _state.graph = GraphRAGAdapter()
    _state.indexer.graph_adapter = _state.graph
    _state.indexer.index.clear()
    _state.indexer._doc_counter = itertools.count()
    _state.tensor_iface = TensorRetrieverInterface()


__all__ = ["index_documents", "answer", "answer_agentic", "upsert_documents", "set_alpha"]
