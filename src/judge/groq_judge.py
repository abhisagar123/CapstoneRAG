"""GroqJudge — TRACe-label judge served by the GROQ API (type "groq").

PURPOSE — this is a VALIDATION instrument, NOT the default scorer. The mentor
(20 Jun 2026) noted the judge should ideally be ≥ the generator; once the generator
becomes Groq Llama-3.3-70B, our LOCKED local judge (llama3.1:8b) is smaller. We
address that by RE-VALIDATING — not by reflexively swapping the locked judge:
  (i)  run THIS bigger judge through the existing judge_validate harness against
       RAGBench's GPT-4 gold scores, ALL domains (Groq removes the Colab-GPU cost
       wall that limited the original Exp-1 bigger-judge test to a slice). Asks "is
       a bigger judge actually MORE ACCURATE vs gold?".
  (ii) score a sample of the 70B-generated answers with this judge ALONGSIDE the
       locked 8B → agreement on the harder text.
Only if BOTH point that way do we swap the locked scoring judge (and re-score key
configs). See AI_CONTEXT §3.12 / §5 / NEXT-STEPS-#3b.

Same job + same contract as OllamaJudge — build the verbatim Appendix-7.4 prompt
(optional conservative steer), call the model, parse the JSON labels, retry once on
malformed JSON. The ONLY difference is the engine: Groq's OpenAI-COMPATIBLE *chat*
endpoint, so it's a TWIN of OllamaJudge but speaks the chat schema:
    request   messages=[{role:"user", content: prompt}]   (not Ollama's "prompt" string)
    json mode response_format={"type":"json_object"}       (OpenAI's name for Ollama format:json)
    response  choices[0].message.content                   (not "response")
Reuses build_prompt / parse_label_json from base.py UNCHANGED + httpx (NO new dep).
LIGHT → registered via load_judges() alongside the hf/ollama judges.

Default model = llama-3.3-70b-versatile (same FAMILY as the locked llama3.1:8b, just
bigger → the cleanest "does size help?" test; a different family like GPT-OSS-120B
would confound size with family). Override via config: openai/gpt-oss-120b, etc.
NO Chinese models here either (no Qwen) — same rule everywhere.

Two differences from OllamaJudge worth knowing:
  - NO num_ctx param. Ollama defaults to a tiny window and SILENTLY truncates long
    prompts (the CUAD gotcha) — so OllamaJudge must set num_ctx high. Groq sizes the
    context server-side and ERRORS (4xx) if a prompt exceeds the model's window rather
    than truncating silently — strictly safer; that error is caught + counted by the
    sweep as a normal skip, never a meaningless truncated score.
  - Hosted inference is NOT bit-deterministic even at temperature 0 — fine for this
    VALIDATION role, and another reason the *scoring* judge stays local + locked.

Prereq: export GROQ_API_KEY=...   (console.groq.com; free tier is rate-limited per
token/min + daily, hence the 429 back-off below). The key is read LAZILY so building
a config offline (e.g. validation) needs no key; only a real API call does.
"""

import os
import threading
import time

from ..registry import register
from .base import build_prompt, parse_label_json

# Same-family, bigger-than-locked-8B judge → cleanest size test. Override via config.
DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── shared request-rate throttle ──────────────────────────────────────────────────
# Groq's free tier limits TOKENS-PER-MINUTE (e.g. 12k on llama-3.3-70b), not request
# count. Our judge prompt embeds full documents (~1.8k tok short-domain), so only a few
# calls fit per minute — and the validation sweep judges examples CONCURRENTLY, which
# bursts straight past the budget. This module-level limiter spaces ALL requests (across
# threads) to at most `requests_per_minute`. Holding the lock across the sleep is
# deliberate: it serializes sends so concurrency can't defeat a per-minute budget.
_RATE_LOCK = threading.Lock()
_LAST_CALL = [0.0]          # monotonic timestamp of the most recent request (shared)


def _throttle(min_interval: float) -> None:
    """Block until `min_interval` s have elapsed since the last request (any thread)."""
    if min_interval <= 0:
        return
    with _RATE_LOCK:
        wait = min_interval - (time.monotonic() - _LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.monotonic()


@register("judge", "groq")
class GroqJudge:
    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 1,
                 conservative: bool = False, timeout: float = 120.0,
                 api_key: str | None = None, max_rate_limit_retries: int = 5,
                 requests_per_minute: float = 0.0, max_backoff: float = 90.0):
        self.model = model
        self.max_retries = max_retries                  # malformed-JSON retries (temp up on retry)
        self.conservative = conservative                # append the conservative steer to the prompt?
        self.timeout = timeout
        # Read the key lazily — do NOT require it at construction, so offline config
        # validation (which builds the object) works keyless. An explicit api_key= wins (tests).
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.max_rate_limit_retries = max_rate_limit_retries   # HTTP 429 back-off (free-tier limit)
        # Proactive pacing: Groq's free tier limits TOKENS/min, and our prompts are big
        # (full docs). requests_per_minute > 0 spaces every send so we stay UNDER the budget
        # instead of relying only on reactive 429 back-off. 0 = off (offline tests / paid tier).
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        # Ceiling on a single 429 wait. Groq sends Retry-After: a per-MINUTE limit resets in
        # seconds (worth waiting for), but the per-DAY token cap (TPD) resets in ~20+ MINUTES —
        # honouring that verbatim parks the whole run in one time.sleep(). If Retry-After
        # exceeds this ceiling we STOP retrying and raise, so the sweep skips+counts the example
        # (a daily cap means the run is done for the day anyway — fail fast, don't hang).
        self.max_backoff = max_backoff

    def _generate(self, prompt: str, sample: bool = False) -> str:
        """POST to Groq's OpenAI-compatible chat endpoint; return the raw completion text.

        First pass is greedy (temperature 0 — deterministic, best for ground truth). On a
        JSON-retry we raise the temperature so the re-generation actually DIFFERS (a greedy
        retry would reproduce the same unparseable text). response_format=json_object asks
        for a single JSON object — note OpenAI requires the word "JSON" to appear in the
        prompt for that mode; the Appendix-7.4 template already says "respond with a JSON
        object", so this holds. On HTTP 429 we honour Retry-After / back off and retry, so a
        big CUAD validation sweep on the free tier doesn't crash on rate limits.
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
                   "temperature": 0.3 if sample else 0.0,
                   "response_format": {"type": "json_object"},
                   "stream": False}

        for attempt in range(self.max_rate_limit_retries + 1):
            _throttle(self._min_interval)               # proactive pacing (stay under TPM)
            resp = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 429 and attempt < self.max_rate_limit_retries:
                # Honour Retry-After if sent; else exponential back-off.
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                if wait > self.max_backoff:
                    # A wait this long means a per-DAY cap (resets in ~20+ min), not the
                    # per-minute bucket. Don't park the run — raise so the sweep skips+counts.
                    raise RuntimeError(
                        f"Groq rate limit needs {wait:.0f}s (> max_backoff {self.max_backoff:.0f}s) "
                        f"— likely the daily token cap (TPD). Stopping retries. Body: {resp.text[:200]}"
                    )
                time.sleep(wait)
                continue
            resp.raise_for_status()                     # 4xx/5xx (incl. context-too-long) -> raise
            data = resp.json()
            # Record token usage + $ against the running budget (Legal sweep cap).
            # BudgetExceeded MUST propagate; any other tracker glitch is swallowed.
            try:
                from .. import cost_tracker
                cost_tracker.record("judge", data.get("usage"), self.model)
            except cost_tracker.BudgetExceeded:
                raise
            except Exception:
                pass
            return data["choices"][0]["message"]["content"] or ""

    def label(self, question: str, keyed: dict) -> dict:
        """Build the Appendix-7.4 prompt, call Groq, parse JSON; retry once with sampling
        on malformed JSON (identical control flow to OllamaJudge — only the engine differs)."""
        prompt = build_prompt(question, keyed, conservative=self.conservative)
        last_err = None
        for attempt in range(self.max_retries + 1):
            raw = self._generate(prompt, sample=(attempt > 0))   # greedy first, then sample
            try:
                return parse_label_json(raw)
            except ValueError as e:
                last_err = e                            # malformed JSON -> retry with sampling
        raise ValueError(f"judge produced unparseable JSON after retries: {last_err}")
