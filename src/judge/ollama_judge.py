"""OllamaJudge — the real OSS judge served LOCALLY by Ollama (type "ollama").

Why this exists: company policy allows running open-source models LOCALLY as long
as they are NOT Chinese models (so Qwen is out; Llama / Mistral / Gemma are fine).
On an Apple-Silicon Mac (e.g. M3 Max, 96 GB) Ollama runs these efficiently — no
Colab, no usage limits, no T4 OOM, and big models (up to ~70B in 4-bit) fit.

Same job + same contract as HuggingFaceJudge: build the Appendix-7.4 prompt
(optionally with the conservative steer), call the model, parse the JSON labels,
retry once on malformed JSON. The ONLY difference is the engine — it POSTs to the
local Ollama HTTP server instead of loading the model into this process. So it is
LIGHT (no torch/transformers): registered via load_judges() like the hf judge,
purely to keep the heavy hf import out of the light path.

Prereq: `ollama serve` running and the model pulled (e.g. `ollama pull llama3.1:8b`).
"""

from ..registry import register
from .base import build_prompt, parse_label_json

# Default = a strong, NON-Chinese, locally-runnable instruct model. Override via config.
# (Ollama model tags: llama3.1:8b, llama3.1:70b, mistral, gemma2:9b, gemma2:27b, ...)
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HOST = "http://localhost:11434"


@register("judge", "ollama")
class OllamaJudge:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 max_retries: int = 1, conservative: bool = False, timeout: float = 600.0):
        self.model = model
        self.host = host.rstrip("/")
        self.max_retries = max_retries
        self.conservative = conservative          # append the conservative steer to the prompt?
        self.timeout = timeout

    def _generate(self, prompt: str, sample: bool = False) -> str:
        """POST to Ollama's /api/generate and return the raw completion text.

        First pass is greedy (temperature 0 — deterministic, best for ground truth).
        On a retry we raise the temperature so the re-generation actually DIFFERS;
        a greedy retry would reproduce the same unparseable text and be pointless.
        `format: "json"` nudges Ollama to emit a single JSON object.
        """
        import httpx

        options = {"temperature": 0.3 if sample else 0.0}
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "format": "json", "options": options}
        resp = httpx.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["response"]

    def label(self, question: str, keyed: dict) -> dict:
        prompt = build_prompt(question, keyed, conservative=self.conservative)
        last_err = None
        for attempt in range(self.max_retries + 1):
            raw = self._generate(prompt, sample=(attempt > 0))   # greedy first, then sample
            try:
                return parse_label_json(raw)
            except ValueError as e:
                last_err = e                          # malformed JSON -> retry with sampling
        raise ValueError(f"judge produced unparseable JSON after retries: {last_err}")
