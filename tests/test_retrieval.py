from opsrag.retrieval import rrf
from opsrag.store import Hit


def hit(chunk_id: int) -> Hit:
    return Hit(chunk_id, "k8s", f"{chunk_id}.md", "url", "title", "heading", "content", None, 0.0)


def test_rrf_prefers_chunks_found_by_both_retrievers():
    vector = [hit(1), hit(2), hit(3)]
    fts = [hit(3), hit(1), hit(4)]
    assert [h.chunk_id for h in rrf([vector, fts])] == [1, 3, 2, 4]


def test_rrf_scores():
    [only] = rrf([[hit(7)]], k=60)
    assert only.score == 1 / 61


def test_rrf_single_ranking_keeps_order():
    assert [h.chunk_id for h in rrf([[hit(5), hit(9), hit(2)]])] == [5, 9, 2]
