"""MinimalGroundedPromptBuilder — minimal framing WITH grounding constraint.
Combines the brevity of minimal (no role-play, no refusal clause) with a
single grounding + completeness instruction. Hypothesis: small models respond
better to short, direct instructions than long elaborate ones.
Registered as prompt type "minimal_grounded".
"""
from ..registry import register
from .base import format_chunks

@register("prompt", "minimal_grounded")
class MinimalGroundedPromptBuilder:
    def build(self, query: str, chunks) -> str:
        context = format_chunks(chunks)
        return (
            f"Answer the question using only the context below. "
            f"Be thorough and include every relevant detail from the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
