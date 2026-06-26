"""HeuristicClassifier — cheap rules, no model load.

Decides needs_retrieval by simple signals:
  - very short queries (≤ `min_length` tokens) — likely greetings / chitchat → no retrieval
  - queries containing any of `skip_keywords` (e.g. "hello", "thanks") → no retrieval
  - everything else → retrieve

This is intentionally simple. The point of having a classifier stage at all
(per Project PDF §3 / Paper-2 §4.1) is more about pipeline shape than head-of-
queue accuracy — a stronger BERT or LLM classifier can be dropped in later
without touching the pipeline.

Registered as query_classifier type "heuristic".
"""

import re

from ..registry import register
from .base import ClassificationResult


_DEFAULT_SKIP_KEYWORDS = (
    "hello", "hi ", "hey", "thanks", "thank you", "bye",
    "good morning", "good evening", "good night",
)


@register("query_classifier", "heuristic")
class HeuristicClassifier:
    def __init__(
        self,
        min_length: int = 3,
        skip_keywords: tuple[str, ...] | list[str] | None = None,
    ):
        self.min_length = int(min_length)
        self.skip_keywords = tuple(skip_keywords) if skip_keywords is not None else _DEFAULT_SKIP_KEYWORDS

    def classify(self, query: str) -> ClassificationResult:
        q = (query or "").strip().lower()
        if not q:
            return ClassificationResult(needs_retrieval=False, confidence=1.0)

        # Token-count short-circuit: 1-2 word queries usually aren't real questions.
        n_tokens = len(re.findall(r"\b\w+\b", q))
        if n_tokens < self.min_length:
            return ClassificationResult(needs_retrieval=False, confidence=0.8)

        # Keyword short-circuit: obvious chitchat → don't retrieve.
        for kw in self.skip_keywords:
            if kw in q:
                return ClassificationResult(needs_retrieval=False, confidence=0.7)

        # Default: real informational query → retrieve.
        return ClassificationResult(needs_retrieval=True, confidence=0.9)
