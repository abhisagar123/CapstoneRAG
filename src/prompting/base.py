"""Prompting — shared contract (interface + chunk-formatting helper).

A PromptBuilder assembles the single text block we hand the LLM: it WRAPS the
retrieved chunks with the question (and, for grounded variants, instructions).
It does NOT shrink chunk text (that would be the optional Summarizer) and does
NOT apply model-specific chat tokens (that's the Generator, brick 7) — it
returns a plain, model-agnostic string. See LLD §3.

The exact wording is our biggest lever on the TRACe metrics:
  - "use ONLY the context"      → Adherence (anti-hallucination)
  - how chunks are presented    → Utilization / Completeness
  - "say so if not in context"  → negative rejection (matters for RGB)
So prompts are swappable strategies we A/B in ablations.
"""

from typing import Protocol, runtime_checkable

from ..indexing import RetrievedChunk


@runtime_checkable
class PromptBuilder(Protocol):
    """The contract every prompt builder honours."""

    def build(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Assemble the full prompt string from the question + retrieved chunks."""
        ...


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as a numbered context block, e.g.:
        [1] <chunk text>
        [2] <chunk text>
    Shared by the prompt variants. Order is preserved (the repacker already
    decided it); numbering is 1-based for human/LLM readability."""
    return "\n".join(f"[{i}] {rc.chunk.text}" for i, rc in enumerate(chunks, start=1))
