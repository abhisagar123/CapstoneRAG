"""GroundedCoveragePromptBuilder — a PROCEDURAL coverage push (anti-terseness).

Motivation (Pooled Exp E): on CustomerSupport, completeness/utilization are the stuck
metrics. Telemetry showed the failure is NOT hallucination and NOT the token cap — the
8B (and gemma2:9b) ANSWER almost everything but write SHORT, over-summarised answers
that touch FEWER of the relevant context sentences (abstain ~2%, answer length 245-362
chars, adherence held). The current winner prompt ("grounded_complete") already says
"be thorough" — and the model demonstrably ignores that abstract adjective.

This variant attacks that exact mechanism by replacing the abstract "be thorough" with a
PROCEDURAL instruction: the context is numbered passages, so walk them ONE AT A TIME and
include each passage's relevant detail. Small models follow a concrete recipe far better
than an adjective. "Consider each numbered passage in turn" operationalises "use more of
the relevant sentences"; it does not merely re-exhort thoroughness.

⚠️ Adherence guardrail (Pooled Exp 16b lesson): pushing a small model to say MORE can flip
it from abstain->answer, and those extra sentences can be ungrounded -> adherence craters.
So the push is framed strictly as a COVERAGE rule ("do not OMIT a relevant detail to keep
your answer short"), NOT a LENGTH reward ("be longer"). The strict grounding opener, the
"do not add information not in the context" clause, and the refusal line are preserved
VERBATIM from grounded_complete, so the ONLY behavioural change vs that winner is the
procedural coverage steer — a clean one-variable (prompt-only) ablation.

Registered as prompt type "grounded_coverage".
"""

from ..registry import register
from .base import format_chunks

DEFAULT_REFUSAL = "I cannot answer this question based on the provided context."

# Strict grounding (adherence) + a PROCEDURAL coverage steer (completeness). vs
# grounded_complete, the abstract "Be thorough: include every relevant detail..."
# sentence is replaced by the "numbered passages; consider each in turn ... do not omit
# a relevant detail to keep your answer short" recipe. The grounding opener, the no-outside
# -info clause, and the refusal clause are kept identical so this differs from
# grounded_complete ONLY by the coverage steer.
INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "below. The context is given as numbered passages; consider each passage in "
    "turn and include every detail in it that is relevant to the question. Do not "
    "add any information that is not in the context, and do not omit a relevant "
    "detail to keep your answer short. If the answer is not contained in the "
    'context, reply exactly: "{refusal}"'
)


@register("prompt", "grounded_coverage")
class GroundedCoveragePromptBuilder:
    def __init__(self, refusal: str = DEFAULT_REFUSAL):
        # refusal kept identical to GroundedPromptBuilder / grounded_complete so the
        # variants differ ONLY by the coverage instruction.
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
