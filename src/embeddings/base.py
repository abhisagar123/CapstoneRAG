"""Embedding — shared contract (interface).

An embedder turns text into vectors (lists of numbers) that capture meaning:
texts with similar meaning get vectors that are close together. This is what
makes "search by meaning" possible downstream — we embed chunks once (offline),
embed the query at search time, and find the nearest chunk vectors.

This file holds ONLY the shared contract so every embedding strategy (different
models) honours one interface. See LLD §3.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """The contract every embedder honours."""

    def embed(self, texts: list[str]):
        """Encode texts → a 2-D float array of shape (len(texts), dim)."""
        ...

    @property
    def dim(self) -> int:
        """The length of each vector (e.g. 384 for all-MiniLM-L6-v2)."""
        ...
