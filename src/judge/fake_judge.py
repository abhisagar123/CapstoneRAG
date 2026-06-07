"""FakeJudge — deterministic, no-model judge for offline tests + pipeline wiring.

Produces PLAUSIBLE-SHAPED labels (valid RAGBench JSON) without any model: marks a
deterministic fraction of document sentences as relevant/utilized and reports the
answer as supported. The scores it yields are NOT meaningful — it exists only to
exercise the segment->judge->score path locally and in tests. The real judge
(HuggingFaceJudge / OpenAIJudge) replaces it. Registered as judge type "fake".
"""

from ..registry import register


@register("judge", "fake")
class FakeJudge:
    def __init__(self, relevant_frac: float = 0.5, utilized_frac: float = 0.34,
                 supported: bool = True):
        self.relevant_frac = relevant_frac
        self.utilized_frac = utilized_frac
        self.supported = supported

    def label(self, question: str, keyed: dict) -> dict:
        doc_keys = [s[0] for doc in keyed["documents_sentences"] for s in doc]
        n = len(doc_keys)
        relevant = doc_keys[: max(1, int(n * self.relevant_frac))] if n else []
        utilized = doc_keys[: max(1, int(n * self.utilized_frac))] if n else []
        return {
            "relevance_explanation": "(fake) deterministic labels for testing",
            "all_relevant_sentence_keys": relevant,
            "overall_supported_explanation": "(fake)",
            "overall_supported": self.supported,
            "sentence_support_information": [
                {"response_sentence_key": k, "explanation": "(fake)",
                 "supporting_sentence_keys": (relevant[:1] if self.supported else []),
                 "fully_supported": self.supported}
                for k, _ in keyed["response_sentences"]
            ],
            "all_utilized_sentence_keys": utilized,
        }
