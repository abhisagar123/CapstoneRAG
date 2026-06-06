"""Retriever — the front door to search: text question → nearest chunks.

The Index works in vectors; the Retriever works in TEXT. It embeds the query
with the same embedder used to build the index, asks the index for the nearest
chunks, and returns them. This is the online half of "search by meaning".

Baseline = dense retrieval only (vector similarity). Sparse (BM25), hybrid (RRF),
and reranking are later registered strategies / a later brick — the interface
below is deliberately small so those slot in without changing callers.

Critical invariant: the query MUST be embedded with the SAME embedder that built
the index (same model → same vector space). The DenseRetriever holds both, so
that can't drift.
"""

from typing import Protocol, runtime_checkable

from .registry import register
from .indexer import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """The contract every retriever honours."""
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        ...


@register("retriever", "dense")
class DenseRetriever:
    """Embed the query, return the index's k nearest chunks. Pure vector search."""

    def __init__(self, embedder, index):
        # embedder: anything with .embed([...]) -> (n, dim)  (an Embedder)
        # index:    a built Index with .search(qvec, k)      (e.g. FaissIndex)
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed([query])[0]   # (dim,) — same model as the index
        return self.index.search(query_vec, k=k)
