from dataclasses import replace
from typing import Literal

import psycopg

from opsrag import rerank, store
from opsrag.config import Settings
from opsrag.ollama import Ollama
from opsrag.store import Hit

Mode = Literal["vector", "fts", "hybrid", "rerank"]

# Standard constant from the original RRF paper, dampens the advantage of the very top ranks
RRF_K = 60


def rrf(rankings: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: merge ranked lists using only positions, not raw scores.

    Cosine similarity and ts_rank live on different scales, so they can't be added directly.
    A chunk ranked high by both retrievers ends up first.
    """
    scores: dict[int, float] = {}
    hits: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (k + rank)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [replace(hits[chunk_id], score=scores[chunk_id]) for chunk_id in ordered]


def search(
    conn: psycopg.Connection,
    ollama: Ollama,
    settings: Settings,
    query: str,
    mode: Mode = "rerank",
    limit: int | None = None,
    source: str | None = None,
) -> list[Hit]:
    limit = limit or settings.top_k
    if mode == "rerank":
        pool = search(conn, ollama, settings, query, "hybrid", settings.rerank_pool, source)
        return reorder(settings, query, pool)[:limit]
    rankings = []
    if mode in ("vector", "hybrid"):
        [embedding] = ollama.embed([settings.query_prefix + query])
        rankings.append(store.vector_search(conn, embedding, settings.candidates, source))
    if mode in ("fts", "hybrid"):
        rankings.append(store.fts_search(conn, query, settings.candidates, source))
    if mode == "hybrid":
        return rrf(rankings)[:limit]
    return rankings[0][:limit]


def reorder(settings: Settings, query: str, hits: list[Hit]) -> list[Hit]:
    """Reranked hits, or the same order when the reranker is off or not downloaded."""
    reranker = rerank.load(settings)
    return reranker.rerank(query, hits) if reranker else hits
