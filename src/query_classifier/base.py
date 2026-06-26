"""Query classification — shared contract (interface).

A query classifier decides whether the incoming query needs retrieval at all.
Wang et al. (EMNLP 2024 §4.1) show that this single binary decision lifts the
end-to-end avg score by +0.015 and cuts latency ~30% by skipping retrieval on
questions the LLM can answer from its own parametric knowledge.

Two outputs:
  needs_retrieval (bool)  — True → run the rest of the pipeline
                            False → bypass retrieve/rerank/repack/summarize
                            and let the generator answer directly
  confidence (float)      — 0..1, optional signal for downstream logging

This file is the contract only; concrete strategies live alongside (heuristic,
always_retrieve, …). Same base.py pattern as the other component packages.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ClassificationResult:
    """Output of a query classifier."""

    needs_retrieval: bool
    confidence: float = 1.0


@runtime_checkable
class QueryClassifier(Protocol):
    """The contract every query classifier honours."""

    def classify(self, query: str) -> ClassificationResult:
        """Decide whether `query` requires retrieval. Pure function."""
        ...
