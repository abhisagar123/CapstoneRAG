"""GroundedFewshotPromptBuilder — teach thorough+grounded answering by DEMONSTRATION.

Motivation (Pooled Exp G + Exp E): on CustomerSupport the stuck metric is completeness —
the 8B answers but writes short, over-summarised answers that touch fewer of the relevant
context sentences. Every prompt we have so far TELLS the model what to do (an instruction):
"be thorough" (grounded_complete, the winner), "quote the source" (extractive), the
procedural "consider each passage in turn" (grounded_coverage) — which actually BACKFIRED
(it made the 8B refuse more and ground worse). The one prompt mode we had NOT tried is
SHOWING instead of telling: small models often imitate a worked example far better than
they obey an adjective.

This variant keeps the winner's instruction VERBATIM and PREPENDS two short, SYNTHETIC
demonstrations (not from the test set — no leakage) before the real question:
  - Example 1 answers a question by pulling a fact from EVERY passage -> demonstrates the
    extent of "thorough" coverage by example (targets completeness/utilization).
  - Example 2 is an UNANSWERABLE question -> the demo answer is the exact refusal string,
    which DEMONSTRATES the adherence guardrail (refuse when the answer is absent) rather
    than just stating it. This is the deliberate counter to the Exp G failure mode (pushing
    coverage must not erode "say nothing you can't support").

⚠️ This is a measurement-CLEAN intervention: the demonstrations live in the INPUT prompt;
the model still emits ONE answer after the final "Answer:", and the judge scores only that
emitted answer. The demos are never part of the scored output (unlike a two-step
"list facts then answer" prompt, whose copied-context preamble would fake-inflate
utilization/completeness — deliberately avoided). The final Context/Question/Answer scaffold
is byte-identical to grounded_complete, so the ONLY change vs the winner is the prepended
demonstrations — a clean prompt-only ablation.

Registered as prompt type "grounded_fewshot".
"""

from ..registry import register
from .base import format_chunks

DEFAULT_REFUSAL = "I cannot answer this question based on the provided context."

# The winner's instruction, kept VERBATIM (grounded_complete) so the only change is the demos.
INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "below. Be thorough: include every relevant detail that the context provides "
    "for answering the question, but do not add any information that is not in the "
    "context. If the answer is not contained in the context, reply exactly: "
    '"{refusal}"'
)

# Two SYNTHETIC worked examples (CustomerSupport-flavoured, NOT from the test set).
# Example 1 deliberately pulls one fact from EACH of its three passages into a single
# thorough answer (shows the extent of coverage). Example 2 is unanswerable from its
# context, so its answer is the refusal string (shows the adherence guardrail by example).
# Only brace is {refusal}; "$50"/"5-7" are literal and safe for str.format.
FEWSHOT = (
    "\n"
    "Here are two examples of how to answer. Study them, then answer the final "
    "question in the same way.\n"
    "\n"
    "Example 1\n"
    "Context:\n"
    "[1] Standard shipping takes 5-7 business days.\n"
    "[2] Express shipping arrives in 2 business days for an extra fee.\n"
    "[3] Orders over $50 qualify for free standard shipping.\n"
    "Question: What are the shipping options?\n"
    "Answer: There are two shipping options. Standard shipping takes 5-7 business "
    "days and is free on orders over $50; express shipping arrives in 2 business "
    "days for an extra fee.\n"
    "\n"
    "Example 2\n"
    "Context:\n"
    "[1] The app supports light and dark themes.\n"
    "Question: Does the app integrate with calendar software?\n"
    "Answer: {refusal}\n"
)


@register("prompt", "grounded_fewshot")
class GroundedFewshotPromptBuilder:
    def __init__(self, refusal: str = DEFAULT_REFUSAL):
        # refusal kept identical to grounded_complete so the variants differ ONLY by the
        # prepended demonstrations (and the refusal demo uses this exact string).
        self.refusal = refusal

    def build(self, query: str, chunks) -> str:
        instruction = INSTRUCTION.format(refusal=self.refusal)
        fewshot = FEWSHOT.format(refusal=self.refusal)
        context = format_chunks(chunks)
        return (
            f"{instruction}\n"
            f"{fewshot}\n"
            f"Now answer this question.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
