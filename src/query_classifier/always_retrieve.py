"""AlwaysRetrieveClassifier — the identity / baseline.

Always returns needs_retrieval=True. This is the default for our pipeline if
no classifier is configured (every existing config still works without
declaring a `query_classifier:` stage). It also serves as the baseline that
the heuristic / model-based classifiers are compared against.

Registered as query_classifier type "always_retrieve".
"""

from ..registry import register
from .base import ClassificationResult


@register("query_classifier", "always_retrieve")
class AlwaysRetrieveClassifier:
    def classify(self, query: str) -> ClassificationResult:  # noqa: ARG002 — query unused by design
        return ClassificationResult(needs_retrieval=True, confidence=1.0)
