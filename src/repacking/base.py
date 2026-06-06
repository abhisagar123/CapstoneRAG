"""Repacking — shared contract (interface).

A repacker decides the ORDER in which the (already chosen) chunks are placed in
the prompt. It does NOT filter or re-score — pure reordering. Order matters
because LLMs attend most to the start and end of their context and can overlook
the middle ("lost in the middle"); so where we put the most-relevant chunk
affects the answer.

This file holds ONLY the contract so every ordering strategy (forward / reverse
/ sides) honours one interface — same base.py pattern as the other components.
"""

from typing import Protocol, runtime_checkable

from ..indexing import RetrievedChunk


@runtime_checkable
class Repacker(Protocol):
    """The contract every repacker honours."""

    def pack(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Reorder chunks for the prompt. Input is assumed ranked best→worst
        (rank 0 first). Returns the same chunks in a new order — no add/remove."""
        ...
