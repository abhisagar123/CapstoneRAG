"""Checks for generation (src/generation/).

  * OFFLINE — EchoGenerator behavior, registry, lazy-load discipline. No model.
  * MODEL  — real HuggingFaceGenerator. Gated behind MODEL=1 and intended for
    COLAB (per policy, real models don't run locally):
        MODEL=1 python tests/check_generation.py

Run directly:  python tests/check_generation.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401
from src.registry import build, available
from src.generation import Generator, EchoGenerator, load_generators
from src.generation.echo_generator import ECHO_PREFIX

WANT_MODEL = os.environ.get("MODEL") == "1"


# ---------------- OFFLINE (no model, no torch) ----------------

def test_only_echo_registered_before_load():
    assert "echo" in available("generator")
    assert "hf" not in available("generator")          # heavy one not yet loaded
    assert "torch" not in sys.modules                  # import src stayed light
    assert "transformers" not in sys.modules


def test_echo_generator_interface_and_output():
    g = EchoGenerator()
    assert isinstance(g, Generator)
    out = g.generate("Context:\n[1] foo\n\nQuestion: What is X?\nAnswer:")
    assert out.startswith(ECHO_PREFIX)
    assert "What is X?" in out                         # pulled the question out


def test_echo_handles_prompt_without_question():
    out = EchoGenerator().generate("just some text, no question line")
    assert out.startswith(ECHO_PREFIX)                 # doesn't crash


def test_build_echo_via_registry():
    g = build("generator", "echo")
    assert isinstance(g, EchoGenerator)


def test_load_generators_registers_hf():
    load_generators()
    assert "hf" in available("generator")              # registered, not constructed


def test_load_generators_registers_ollama():
    load_generators()
    assert "ollama" in available("generator")          # local-server backend


def test_load_generators_registers_groq():
    load_generators()
    assert "groq" in available("generator")            # hosted-API backend


def test_ollama_generator_posts_and_parses(monkeypatch=None):
    # OFFLINE: mock httpx.post so we verify the request + response handling with NO server.
    import httpx
    from src.generation.ollama_generator import OllamaGenerator

    captured = {}

    class _FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"response": "Paris is the capital."}

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResp()

    orig = httpx.post
    httpx.post = _fake_post
    try:
        g = OllamaGenerator(model="llama3.1:8b", max_new_tokens=42, temperature=0.0)
        out = g.generate("Question: capital of France?\nAnswer:")
    finally:
        httpx.post = orig                              # always restore

    assert out == "Paris is the capital."              # parsed the 'response' field
    assert captured["url"].endswith("/api/generate")
    assert captured["payload"]["model"] == "llama3.1:8b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["num_predict"] == 42
    assert captured["payload"]["options"]["temperature"] == 0.0
    assert captured["payload"]["options"]["num_ctx"] >= 8192    # avoid silent truncation


def test_groq_generator_posts_chat_schema_and_parses():
    # OFFLINE: mock httpx.post. Groq speaks the OpenAI CHAT schema, not Ollama's — verify
    # we send messages[] (not a raw prompt) and parse choices[0].message.content.
    import httpx
    from src.generation.groq_generator import GroqGenerator, GROQ_URL

    captured = {}

    class _FakeResp:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "Paris is the capital."},
                                 "finish_reason": "stop"}]}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return _FakeResp()

    orig = httpx.post
    httpx.post = _fake_post
    try:
        # api_key passed explicitly so the test never depends on the environment.
        g = GroqGenerator(model="llama-3.3-70b-versatile", max_new_tokens=42,
                          temperature=0.0, api_key="test-key")
        out = g.generate("Question: capital of France?\nAnswer:")
    finally:
        httpx.post = orig                              # always restore

    assert out == "Paris is the capital."              # parsed choices[0].message.content
    assert captured["url"] == GROQ_URL                 # OpenAI-compatible chat endpoint
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "llama-3.3-70b-versatile"
    assert captured["payload"]["messages"][0]["role"] == "user"   # prompt wrapped as a chat turn
    assert captured["payload"]["messages"][0]["content"].startswith("Question:")
    assert captured["payload"]["max_tokens"] == 42                # chat schema, NOT num_predict
    assert captured["payload"]["temperature"] == 0.0
    assert "prompt" not in captured["payload"]                    # NOT Ollama's prompt-string schema


def test_groq_generator_requires_key_lazily():
    # No key at construction is OK (offline config validation builds the object); the clear
    # error fires only when generate() actually needs to hit the API.
    from src.generation.groq_generator import GroqGenerator
    g = GroqGenerator(api_key=None)                    # constructs fine, no env key needed
    try:
        g.generate("Question: anything?\nAnswer:")
        assert False, "expected RuntimeError when GROQ_API_KEY is missing"
    except RuntimeError as e:
        assert "GROQ_API_KEY" in str(e)                # actionable message, not a KeyError


def test_groq_generator_retries_on_429():
    # OFFLINE: first call returns 429 (rate limited), second returns 200 — verify the
    # back-off loop retries and ultimately parses the success. time.sleep is stubbed to 0.
    import time
    import httpx
    from src.generation.groq_generator import GroqGenerator

    calls = {"n": 0}

    class _Resp429:
        status_code = 429
        headers = {"Retry-After": "0"}                 # honoured; 0 so the test is instant
        def raise_for_status(self): raise AssertionError("429 should be retried, not raised")
        def json(self): return {}

    class _Resp200:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return _Resp429() if calls["n"] == 1 else _Resp200()

    orig_post, orig_sleep = httpx.post, time.sleep
    httpx.post = _fake_post
    time.sleep = lambda s: None                        # don't actually wait
    try:
        g = GroqGenerator(api_key="test-key", max_retries=3)
        out = g.generate("Question: q?\nAnswer:")
    finally:
        httpx.post, time.sleep = orig_post, orig_sleep

    assert out == "ok"
    assert calls["n"] == 2                              # retried exactly once (429 -> 200)


# ---------------- MODEL (real LLM — run on Colab) ----------------

def test_hf_generator_produces_text():
    load_generators()
    g = build("generator", "hf", {"model": "Qwen/Qwen2.5-3B-Instruct", "max_new_tokens": 32})
    out = g.generate("Context:\n[1] Paris is the capital of France.\n\n"
                     "Question: What is the capital of France?\nAnswer:")
    assert isinstance(out, str) and len(out) > 0
    assert "Paris" in out                               # should answer from context


def _run():
    offline = [test_only_echo_registered_before_load,
               test_echo_generator_interface_and_output,
               test_echo_handles_prompt_without_question,
               test_build_echo_via_registry,
               test_load_generators_registers_hf,
               test_load_generators_registers_ollama,
               test_load_generators_registers_groq,
               test_ollama_generator_posts_and_parses,
               test_groq_generator_posts_chat_schema_and_parses,
               test_groq_generator_requires_key_lazily,
               test_groq_generator_retries_on_429]
    model = [test_hf_generator_produces_text]

    for fn in offline:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"{len(offline)} offline checks passed.")

    if WANT_MODEL:
        print("\nrunning model checks (MODEL=1; COLAB — loads a real LLM)...")
        for fn in model:
            fn(); print(f"  ✅ {fn.__name__}")
        print(f"{len(model)} model checks passed.")
    else:
        print("\n(skipped model checks — set MODEL=1 on Colab to run them)")


if __name__ == "__main__":
    _run()
