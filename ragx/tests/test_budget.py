from ragx.control.budget import ContextBudgeter
from ragx.models import RetrievedChunk
from ragx.summarize.longllmlingua import LongLLMLinguaSummarizer
from ragx.summarize.recomp import RecompSummarizer


def test_budget_enforces_limit():
    chunk_text = "Energy efficiency improvements reduce overall demand and lower emissions significantly. " * 10
    chunk = RetrievedChunk(chunk_id="c1", doc_id="d1", text=chunk_text, retrieval_score=0.9)
    context = " ".join(chunk_text.split() * 2)
    budgeter = ContextBudgeter(50, RecompSummarizer(), LongLLMLinguaSummarizer())
    result = budgeter.enforce(context, [chunk])
    assert len(result.context.split()) <= 50
    assert result.strategy in {"primary", "fallback"}
