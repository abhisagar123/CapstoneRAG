"""Generation — shared contract (interface).

A generator takes the assembled prompt string (from the PromptBuilder) and
produces the answer text. This is the LLM step — the "G" in RAG.

Policy (10 Jun 2026): open-source models may run LOCALLY as long as they are NOT
Chinese models. So this package has three strategies:
  - OllamaGenerator      (type "ollama")— real LLM via local Ollama server (Mac); light
  - HuggingFaceGenerator (type "hf")    — real LLM in-process via transformers (Colab GPU)
  - EchoGenerator        (type "echo")  — pure-Python test target/fallback (no model)

This file holds ONLY the contract so both honour one interface. See LLD §3.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    """The contract every generator honours."""

    def generate(self, prompt: str) -> str:
        """Produce answer text for a fully-assembled prompt string."""
        ...
