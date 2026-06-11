"""GroundedCompletePromptBuilder — grounding PLUS a completeness push.

Motivation (Experiment 3 / reference comparison): our pipeline's two gaps vs the
RAGBench reference are ADHERENCE and COMPLETENESS. The baseline "grounded" prompt
targets adherence ("use ONLY the context") but says nothing about COVERAGE — and
completeness is exactly "of the relevant material, how much did the answer cover?".

This variant keeps the strict grounding + refusal (to hold adherence) and ADDS an
instruction to be thorough: include every relevant detail *that is in the context*.

⚠️ Built-in tension worth measuring: "be thorough" (recall) can tempt a small model
to pad the answer with unsupported claims, which would HURT adherence (precision).
The phrasing deliberately scopes thoroughness to "from the context" to keep the
grounding intact. Whether completeness rises WITHOUT adherence falling is the
experimental question — a clean one-variable (prompt-only) ablation vs "grounded".

Registered as prompt type "grounded_complete".
"""

from ..registry import register
from .base import format_chunks

DEFAULT_REFUSAL = "I cannot answer this question based on the provided context."

# Strict grounding (adherence) + a scoped thoroughness push (completeness). The
# refusal clause is preserved verbatim from the grounded baseline so the ONLY
# behavioural change vs "grounded" is the added completeness sentence.
INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "below. Be thorough: include every relevant detail that the context provides "
    "for answering the question, but do not add any information that is not in the "
    "context. If the answer is not contained in the context, reply exactly: "
    '"{refusal}"'
)


@register("prompt", "grounded_complete")
class GroundedCompletePromptBuilder:
    def __init__(self, refusal: str = DEFAULT_REFUSAL):
        # refusal string is configurable — kept identical to GroundedPromptBuilder
        # so the two differ ONLY by the thoroughness instruction.
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
