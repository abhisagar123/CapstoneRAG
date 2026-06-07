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
               test_load_generators_registers_hf]
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
