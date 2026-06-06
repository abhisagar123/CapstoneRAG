"""Reranking — shared contract (interface).

The retriever (brick 4) is fast but rough: it compares query and chunk vectors
that were embedded separately. A reranker is slower but more accurate — it looks
at the (query, chunk) pair TOGETHER and scores how well they match. The usual
pattern: retrieve a wide net (e.g. top-20), then rerank down to the true best
top_n (e.g. 5). Analogy: retriever = resume screener, reranker = interviewer.

This file holds ONLY the contract so every reranking strategy honours one
interface — same base.py pattern as the other components.
"""

from typing import Protocol, runtime_checkable

from ..indexing import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """The contract every reranker honours."""

    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_n: int = 5) -> list[RetrievedChunk]:
        """Re-score the (query, chunk) pairs and return the best top_n, re-ranked
        (rank re-assigned 0..top_n-1). Returns fewer than top_n only if fewer
        chunks were given."""
        ...
