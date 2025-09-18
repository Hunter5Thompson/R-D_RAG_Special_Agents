from ragx.control.crag import CRAGEvaluator
from ragx.control.selfrag import SelfRAGRewriter
from ragx.models import RetrievedChunk


def test_crag_triggers_retry():
    chunks = [
        RetrievedChunk(chunk_id="c1", doc_id="d1", text="Sparse info", retrieval_score=0.05),
        RetrievedChunk(chunk_id="c2", doc_id="d2", text="More sparse info", retrieval_score=0.07),
    ]
    evaluator = CRAGEvaluator(min_score=0.55)
    result = evaluator.evaluate(chunks)
    assert result.action == "retry"
    assert result.score < 0.55


def test_selfrag_rewrite_loop():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Photosynthesis efficiency depends on chlorophyll concentration and photon flux.",
            retrieval_score=0.9,
        )
    ]
    rewriter = SelfRAGRewriter(max_loops=2)
    original_query = "plant growth"
    result = rewriter.rewrite_loop(original_query, chunks)
    assert result.final_query != original_query
    assert len(result.rewrites) <= 2
    assert "chlorophyll" in result.final_query
