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
               test_ollama_generator_posts_and_parses]
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
