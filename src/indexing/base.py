"""Indexing — shared contract (interface + result type).

An index stores chunk vectors and finds the nearest ones to a query vector.
"Nearest" = highest inner product, which equals cosine similarity because the
embedder returns L2-normalized vectors (brick 3 `normalize=True`).

This file holds ONLY the shared pieces so every index backend (FAISS now, maybe
Chroma/Milvus later) honours one contract — the same base.py pattern as
chunking/ and embeddings/. See LLD §3.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by search, with WHY it was returned: its similarity
    score and its rank among the results (0 = best match)."""
    chunk: Chunk
    score: float
    rank: int


@runtime_checkable
class Index(Protocol):
    """The contract every vector-index backend honours."""

    def add(self, chunks: list[Chunk], vectors) -> None:
        """Store chunks and their (n, dim) vectors (callable repeatedly to grow)."""
        ...

    def search(self, query_vector, k: int = 5) -> list[RetrievedChunk]:
        """Return the k stored chunks whose vectors are nearest the query vector."""
        ...

    def __len__(self) -> int:
        """How many chunks are currently stored."""
        ...
