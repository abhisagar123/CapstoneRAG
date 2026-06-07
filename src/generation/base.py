"""Generation — shared contract (interface).

A generator takes the assembled prompt string (from the PromptBuilder) and
produces the answer text. This is the LLM step — the "G" in RAG.

Per company policy, real model generation runs on COLAB, never locally (see
AI_CONTEXT §18 brick 7 / env-decisions). So this package has two strategies:
  - HuggingFaceGenerator (type "hf")  — the real LLM, run on Colab (heavy)
  - EchoGenerator        (type "echo")— pure-Python test target/fallback (no model)

This file holds ONLY the contract so both honour one interface. See LLD §3.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    """The contract every generator honours."""

    def generate(self, prompt: str) -> str:
        """Produce answer text for a fully-assembled prompt string."""
        ...
