"""OllamaGenerator — the RAG answer generator served LOCALLY by Ollama (type "ollama").

Policy: open-source models may run locally as long as they are NOT Chinese models.
On Apple Silicon, Ollama serves Llama / Mistral / Gemma efficiently. This generator
POSTs the assembled prompt to the local Ollama server and returns the answer text.

LIGHT (just httpx — no torch/transformers), but registered via load_generators()
alongside the hf generator so all real backends are discovered in one place.

Prereq: `ollama serve` running and the model pulled (e.g. `ollama pull llama3.1:8b`).
"""

from ..registry import register

DEFAULT_MODEL = "llama3.1:8b"          # strong, non-Chinese, locally-runnable
DEFAULT_HOST = "http://localhost:11434"


@register("generator", "ollama")
class OllamaGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 temperature: float = 0.0, max_new_tokens: int = 256, timeout: float = 600.0,
                 num_ctx: int = 16384):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        # num_ctx = runtime context window. Ollama defaults to ~2-4k tokens and SILENTLY
        # truncates longer prompts. The generator prompt holds the retrieved chunks (smaller
        # than the judge's full-doc prompt), but a long-context domain could still overflow
        # the default — so we set it explicitly. Match the judge's default.
        self.num_ctx = num_ctx

    def generate(self, prompt: str) -> str:
        """Send the prompt to the local Ollama server and return the answer text.

        temperature 0 by default (deterministic — what we want for evaluation runs).
        num_predict caps the answer length (Ollama's name for max new tokens);
        num_ctx sizes the input window (else Ollama silently truncates long prompts).
        """
        import httpx

        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": self.temperature,
                               "num_predict": self.max_new_tokens,
                               "num_ctx": self.num_ctx}}
        resp = httpx.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["response"]
