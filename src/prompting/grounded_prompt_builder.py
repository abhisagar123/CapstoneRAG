"""GroundedPromptBuilder — the baseline grounding prompt (anti-hallucination).

Explicitly instructs the model to answer using ONLY the provided context and to
refuse when the answer isn't present. This grounding instruction is the main
driver of Adherence (no hallucination) and of refusal behaviour. It is the
default prompt; the minimal variant is the contrast arm that omits the
instruction, so ablations can measure how much the instruction is worth.

Registered as prompt type "grounded".
"""

from ..registry import register
from .base import format_chunks

DEFAULT_REFUSAL = "I cannot answer this question based on the provided context."

INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "below. If the answer is not contained in the context, reply exactly: "
    '"{refusal}"'
)


@register("prompt", "grounded")
class GroundedPromptBuilder:
    def __init__(self, refusal: str = DEFAULT_REFUSAL):
        # refusal string is configurable — RGB (Task 2) needs a specific one.
        self.refusal = refusal

    def build(self, query: str, chunks) -> str:
        instruction = INSTRUCTION.format(refusal=self.refusal)
        context = format_chunks(chunks)
        return (
            f"{instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
