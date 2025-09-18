import pytest

from ragx.api import public_api


DOCS = [
    "Climate change increases global temperatures and drives extreme weather patterns.",
    "Renewable energy adoption grows as solar and wind become cheaper than coal.",
    "Economic recovery depends on sustainable policies supporting green infrastructure.",
]


@pytest.fixture(autouse=True)
def reset_state():
    public_api._reset_state_for_tests()
    yield
    public_api._reset_state_for_tests()


@pytest.fixture
def indexed_corpus():
    public_api.index_documents(DOCS)
    return DOCS


def test_hybrid_alpha_stability(indexed_corpus):
    query = "climate change energy"
    first_chunks = []
    for alpha in [0.1, 0.3, 0.5]:
        public_api.set_alpha(alpha)
        retrieved = public_api._state.retriever.retrieve(query, top_k=3)
        assert retrieved, "Expected non-empty retrieval results"
        first_chunks.append(retrieved[0].chunk_id)
    assert len(set(first_chunks)) == 1


def test_dense_bm25_deterministic(indexed_corpus):
    query = "renewable energy adoption"
    public_api.set_alpha(0.3)
    first_call = public_api._state.retriever.retrieve(query, top_k=2)
    second_call = public_api._state.retriever.retrieve(query, top_k=2)
    assert [chunk.chunk_id for chunk in first_call] == [chunk.chunk_id for chunk in second_call]


def test_combined_scores_range(indexed_corpus):
    query = "green infrastructure"
    public_api.set_alpha(0.5)
    retrieved = public_api._state.retriever.retrieve(query, top_k=3)
    for chunk in retrieved:
        assert 0.0 <= chunk.retrieval_score <= 1.0
