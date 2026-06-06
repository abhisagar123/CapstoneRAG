"""OFFLINE checks for prompting (src/prompting/). Pure string assembly, no model.

Run directly:  python tests/check_prompting.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401
from src.registry import build, available
from src.prompting import (
    PromptBuilder, GroundedPromptBuilder, MinimalPromptBuilder, format_chunks,
)
from src.indexing import RetrievedChunk
from src.chunking import Chunk


def _chunks(texts):
    return [RetrievedChunk(chunk=Chunk(text=t, doc_id="0", chunk_id=f"0-{i}"),
                           score=1.0 - i * 0.1, rank=i) for i, t in enumerate(texts)]


def test_registered():
    assert set(available("prompt")) == {"grounded", "minimal"}


def test_format_chunks_numbered_in_order():
    out = format_chunks(_chunks(["alpha", "beta", "gamma"]))
    assert out == "[1] alpha\n[2] beta\n[3] gamma"           # 1-based, order preserved
    assert format_chunks([]) == ""                           # empty → empty block


def test_grounded_has_instruction_question_and_context():
    p = GroundedPromptBuilder().build("What is X?", _chunks(["ctx one", "ctx two"]))
    assert "ONLY the context" in p                           # grounding instruction
    assert "What is X?" in p                                 # the question
    assert "[1] ctx one" in p and "[2] ctx two" in p         # numbered context
    assert p.rstrip().endswith("Answer:")                    # cue for the model


def test_minimal_omits_instruction_but_keeps_context_and_question():
    p = MinimalPromptBuilder().build("What is X?", _chunks(["ctx one"]))
    assert "ONLY the context" not in p                       # no grounding instruction
    assert "What is X?" in p and "[1] ctx one" in p


def test_grounded_refusal_is_configurable():
    # default refusal present...
    assert "cannot answer" in GroundedPromptBuilder().build("q", _chunks(["c"]))
    # ...and overridable (RGB will need a specific refusal string)
    custom = GroundedPromptBuilder(refusal="NO ANSWER").build("q", _chunks(["c"]))
    assert "NO ANSWER" in custom


def test_both_satisfy_interface_and_return_str():
    for t in ("grounded", "minimal"):
        b = build("prompt", t)
        assert isinstance(b, PromptBuilder)
        out = b.build("q?", _chunks(["c"]))
        assert isinstance(out, str) and len(out) > 0


def test_empty_chunks_still_builds():
    # No retrieved context shouldn't crash — the prompt is just context-less.
    p = GroundedPromptBuilder().build("q?", [])
    assert "Question: q?" in p


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} prompting checks passed.")


if __name__ == "__main__":
    _run()
