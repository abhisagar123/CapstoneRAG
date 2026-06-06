"""MinimalPromptBuilder — the bare-bones contrast arm (no grounding instruction).

Just context + question, with no "use only the context / refuse if absent"
instruction. It exists to be compared against the grounded variant: the
Adherence/refusal gap between them measures how much the grounding instruction
actually buys us (rather than assuming it helps).

Registered as prompt type "minimal".
"""

from ..registry import register
from .base import format_chunks


@register("prompt", "minimal")
class MinimalPromptBuilder:
    def build(self, query: str, chunks) -> str:
        context = format_chunks(chunks)
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
