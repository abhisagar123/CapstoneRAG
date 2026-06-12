"""ExtractivePromptBuilder — grounding that pushes the model to STICK TO SOURCE WORDING.

Motivation (Experiment 3/4): adherence lags because a small (3B) generator, when asked
to answer "in its own words", drifts from the source and adds unsupported phrasing that
the judge marks as not-grounded. `grounded` and `grounded_complete` both still invite free
rephrasing. This variant attacks adherence from a DIFFERENT angle than grounded_complete:
instead of "be thorough" (which traded adherence for completeness in Exp 4), it tells the
model to answer by QUOTING or closely paraphrasing the context's own sentences — extractive
rather than abstractive. The hypothesis: staying close to source wording raises adherence
(fewer invented phrasings) at the possible cost of fluency.

Keeps the strict grounding + refusal clause verbatim from GroundedPromptBuilder, so the ONLY
change vs "grounded" is the extractive instruction — a clean one-variable comparison.

Registered as prompt type "extractive".
"""

from ..registry import register
from .base import format_chunks

DEFAULT_REFUSAL = "I cannot answer this question based on the provided context."

# Strict grounding (adherence) + an extractive steer: prefer the context's OWN words.
INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the context below. "
    "Quote or closely paraphrase the context's own sentences — do NOT rephrase in your "
    "own words or add anything not stated in the context. If the answer is not contained "
    'in the context, reply exactly: "{refusal}"'
)


@register("prompt", "extractive")
class ExtractivePromptBuilder:
    def __init__(self, refusal: str = DEFAULT_REFUSAL):
        # refusal kept identical to GroundedPromptBuilder so the variants differ ONLY by
        # the extractive instruction.
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
