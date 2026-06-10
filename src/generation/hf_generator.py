"""HuggingFaceGenerator — the real LLM generator (in-process transformers; Colab GPU).

Wraps a HuggingFace causal LM. The model name is a config param (same pattern as
the embedder), so smoke-testing a small model and scaling to a large one is a config
change, not new code:
    {type: hf, model: meta-llama/Llama-3.2-3B-Instruct}                 # smoke test
    {type: hf, model: meta-llama/Llama-3.1-8B-Instruct, load_in_4bit: true}  # real (Colab GPU)

NOTE (policy, 10 Jun 2026): open-source models may also run LOCALLY as long as they
are NOT Chinese models. On a Mac, the lighter path is the OllamaGenerator (type
"ollama") — no torch/bitsandbytes. This hf backend is for the in-process/Colab path.
Use NON-Chinese models (Llama / Mistral / Gemma); do not default to Qwen.

Separation of concerns: the PromptBuilder produced a plain, model-agnostic
string; THIS class wraps it as a chat 'user' turn and applies the model's own
chat template (the model-specific tokens). That's why PromptBuilder stayed plain.

HEAVY: imports transformers/torch lazily inside __init__, and is registered only
via load_generators() — so `import src` never pulls in the ML stack.
"""

from ..registry import register

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"   # small non-Chinese smoke default; scale via config


@register("generator", "hf")
class HuggingFaceGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, max_new_tokens: int = 256,
                 temperature: float = 0.0, load_in_4bit: bool = False,
                 device: str | None = None):
        # Lazy, heavy imports — only when a generator is actually constructed.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        kwargs = {}
        if load_in_4bit:                       # Colab GPU: 4-bit quantization
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        elif device:
            kwargs["device_map"] = device
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self._torch = torch

    def generate(self, prompt: str) -> str:
        # Wrap the plain prompt as a chat user-turn and apply the model's template.
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)   # temperature 0 → deterministic (for eval)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        # Decode only the NEWLY generated tokens (strip the prompt back off).
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
