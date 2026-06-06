"""Retrieval — shared contract (interface).

The Index works in vectors; the Retriever works in TEXT. A retriever takes a
text query and returns the most relevant chunks — the online half of "search by
meaning". This file holds ONLY the contract so every retrieval strategy (dense
now; sparse/BM25, hybrid/RRF later) honours one interface. See LLD §3.
"""

from typing import Protocol, runtime_checkable

from ..indexing import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """The contract every retriever honours."""

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the k most relevant chunks for a text query."""
        ...
