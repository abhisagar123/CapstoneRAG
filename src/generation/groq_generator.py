"""GroqGenerator — RAG answer generator served by the GROQ API (type "groq").

Why this exists: the mentor (20 Jun 2026) directed us to use Groq so we can run a
BIGGER generator (Llama-3.3-70B) at high speed — local 70B was impractical
(~2.5h/domain, which is why the local-70B judge was abandoned). Groq serves
OPEN-WEIGHT models (Llama-3.3-70B, Llama-3.1-8B, GPT-OSS), so this honours the
open-source rule: the constraint is about model LICENSING, not where it runs
(hosted ≠ closed). NO Chinese models here either (no Qwen) — same rule everywhere.

Groq exposes an OpenAI-COMPATIBLE chat endpoint, so this is a close TWIN of
OllamaGenerator but speaks the chat schema, NOT Ollama's /api/generate:
    request   messages=[{role:"user", content: prompt}]   (not a raw "prompt" string)
    response  choices[0].message.content                  (not "response")
    cap name  max_tokens                                  (not num_predict)
Reuses httpx — NO new dependency (keeps the public repo clean). LIGHT, so it is
registered via load_generators() alongside the ollama/hf backends.

Boundaries (AI_CONTEXT §4/§5 — hold firm):
  - Generator ONLY. The SCORING judge stays local llama3.1:8b (LOCKED). A separate
    GroqJudge exists purely for the bigger-judge VALIDATION, never as the default scorer.
  - Hosted inference is NOT bit-deterministic even at temperature 0 (unlike local
    Ollama) — acceptable for the generator (within the N=50 ±0.02–0.04 noise band),
    which is exactly why the judge stays local.
  - GROQ_API_KEY comes from the environment / .env and is NEVER committed (the repo is
    PUBLIC). The key is read LAZILY (at generate() time), so building a config offline
    — e.g. config validation, which constructs the object — works with no key; only a
    real API call needs it.

Prereq: export GROQ_API_KEY=...   (get a key at console.groq.com; the free tier is
rate-limited per token/min + daily, hence the 429 back-off below).
"""

import os

from ..registry import register

# Default = the 8B BRIDGE model. We run this FIRST (vs local llama3.1:8b) to isolate the
# hosting/quantization effect, THEN step to the 70B via config so a 70B delta is pure
# capacity, not the hosting change. Override with model: llama-3.3-70b-versatile.
DEFAULT_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


@register("generator", "groq")
class GroqGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, max_new_tokens: int = 256,
                 temperature: float = 0.0, timeout: float = 120.0,
                 api_key: str | None = None, max_retries: int = 3):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        # Read the key NOW if present, but do NOT require it at construction — that keeps
        # offline config validation (which builds the object to validate it) working with
        # no key. The clear "set GROQ_API_KEY" error fires only when generate() actually
        # hits the API. An explicit api_key= wins (handy for tests).
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.max_retries = max_retries          # retries on HTTP 429 (free-tier rate limit)

    def generate(self, prompt: str) -> str:
        """POST the prompt to Groq's OpenAI-compatible chat endpoint; return answer text.

        The PromptBuilder produced a plain, model-agnostic string; we wrap it as a single
        chat 'user' turn (same as HuggingFaceGenerator) so the prompt stayed backend-agnostic.
        temperature 0 by default — the most reproducible answers a hosted (non-deterministic)
        backend will give. On HTTP 429 (rate limit) we honour Retry-After, else exponential
        back-off, and retry — the judge-validation sweep pushes full CUAD contracts through
        the free tier, so 429s are expected on big runs and must not crash the sweep.
        """
        import time
        import httpx

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it (export GROQ_API_KEY=...) or pass "
                "api_key=...; get a key at console.groq.com. The repo is PUBLIC — never commit it."
            )

        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": self.temperature,
                   "max_tokens": self.max_new_tokens,
                   "stream": False}

        for attempt in range(self.max_retries + 1):
            resp = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 429 and attempt < self.max_retries:
                # Respect Retry-After (seconds) if Groq sent it; else exponential back-off.
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait)
                continue
            resp.raise_for_status()             # 4xx/5xx (incl. a final 429) -> raise
            data = resp.json()
            # content can be null on a content-filter / empty finish — coerce to "" so the
            # caller always gets a str (the Generator contract).
            return data["choices"][0]["message"]["content"] or ""
