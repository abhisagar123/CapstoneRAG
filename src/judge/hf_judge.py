"""HuggingFaceJudge — the real OSS judge (strong instruct model, runs on COLAB).

Builds the Appendix-7.4 prompt, runs a HuggingFace instruct model, and parses the
JSON labels. The judge defines ground truth, so this should be a STRONG model
(e.g. Llama-3-70B / Qwen2.5-32B in 4-bit, VRAM permitting) — model name is a
config param. Validate against reference scores before trusting (§9.4).

HEAVY: imports transformers/torch lazily in __init__; registered only via
load_judges() so `import src` stays light. Retries once on malformed JSON
(OSS models are less reliable than GPT-4 at clean JSON).
"""

from ..registry import register
from .base import build_prompt, parse_label_json

# Strong default; override per Colab VRAM. (User favours Llama-3-70B if it fits.)
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


@register("judge", "hf")
class HuggingFaceJudge:
    def __init__(self, model: str = DEFAULT_MODEL, max_new_tokens: int = 1024,
                 load_in_4bit: bool = False, max_retries: int = 1,
                 conservative: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.conservative = conservative          # append the conservative steer to the prompt?
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self._torch = torch

    def _generate(self, prompt: str, sample: bool = False) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # First pass is greedy (deterministic — best for ground truth). On a retry we
        # SAMPLE (sample=True) so the re-generation actually DIFFERS; a greedy retry
        # would reproduce the same unparseable text byte-for-byte and be pointless.
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": sample}
        if sample:
            gen_kwargs["temperature"] = 0.3            # mild — vary phrasing, not content
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)

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
