"""EchoGenerator — pure-Python test target / fallback (NO model).

Does not actually answer the question — it returns a deterministic templated
string derived from the prompt. Purpose:
  - let the FULL pipeline run end-to-end locally (wiring check) with no model,
  - give the offline test suite a Generator to exercise the interface,
  - serve as a safe fallback when no model backend is available.

It extracts the question (the line after "Question:") so its output is
recognizable and testable. Registered as generator type "echo". Light (no heavy
deps) → registered on `import src`.
"""

from ..registry import register

ECHO_PREFIX = "[ECHO]"


@register("generator", "echo")
class EchoGenerator:
    def generate(self, prompt: str) -> str:
        question = ""
        for line in prompt.splitlines():
            if line.strip().lower().startswith("question:"):
                question = line.split(":", 1)[1].strip()
                break
        # Deterministic, recognizable, and references the prompt so tests can assert.
        return f"{ECHO_PREFIX} received a prompt of {len(prompt)} chars" + (
            f"; question was: {question}" if question else ""
        )
