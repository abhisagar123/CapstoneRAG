"""Checks for reranking (src/reranking/).

  * OFFLINE — registry, NoOpReranker behavior, lazy-load discipline. No model.
  * MODEL  — real CrossEncoderReranker improves ordering. Gated behind MODEL=1:
        MODEL=1 python tests/check_reranking.py

Run directly:  python tests/check_reranking.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401
from src.registry import build, available
from src.reranking import Reranker, NoOpReranker, load_rerankers
from src.indexing import RetrievedChunk
from src.chunking import Chunk

WANT_MODEL = os.environ.get("MODEL") == "1"


def _cands(n):
    """n candidates in retriever order, ids 0-0 .. 0-(n-1)."""
    return [RetrievedChunk(chunk=Chunk(text=f"c{i}", doc_id="0", chunk_id=f"0-{i}"),
                           score=1.0 - i * 0.1, rank=i) for i in range(n)]


# ---------------- OFFLINE ----------------

def test_only_noop_registered_before_load():
    # The heavy cross-encoder must NOT be registered just from `import src`.
    assert "none" in available("reranker")
    assert "torch" not in sys.modules, "import src pulled in torch via reranker!"


def test_noop_truncates_preserving_order():
    r = NoOpReranker()
    assert isinstance(r, Reranker)
    out = r.rerank("q", _cands(10), top_n=3)
    assert [c.chunk.chunk_id for c in out] == ["0-0", "0-1", "0-2"]   # retriever order kept
    assert r.rerank("q", [], top_n=5) == []                          # empty safe


def test_load_rerankers_registers_cross_encoder():
    load_rerankers()
    assert "cross_encoder" in available("reranker")


# ---------------- MODEL ----------------

def test_cross_encoder_reranks_and_reassigns_rank():
    load_rerankers()
    r = build("reranker", "cross_encoder")
    cands = [
        RetrievedChunk(Chunk("The capital of France is Paris.", "0", "0-0"), 0.5, 0),
        RetrievedChunk(Chunk("Bananas are a good source of potassium.", "0", "0-1"), 0.49, 1),
        RetrievedChunk(Chunk("Paris is the largest city in France.", "0", "0-2"), 0.48, 2),
    ]
    out = r.rerank("What is the capital of France?", cands, top_n=2)
    assert len(out) == 2
    assert [c.rank for c in out] == [0, 1]               # ranks re-assigned
    # the banana chunk is clearly irrelevant → must not be the #1 result
    assert "Banana" not in out[0].chunk.text


def _run():
    offline = [test_only_noop_registered_before_load,
               test_noop_truncates_preserving_order,
               test_load_rerankers_registers_cross_encoder]
    model = [test_cross_encoder_reranks_and_reassigns_rank]

    for fn in offline:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"{len(offline)} offline checks passed.")

    if WANT_MODEL:
        print("\nrunning model checks (MODEL=1; may download ~80MB)...")
        for fn in model:
            fn(); print(f"  ✅ {fn.__name__}")
        print(f"{len(model)} model checks passed.")
    else:
        print("\n(skipped model checks — set MODEL=1 to run them)")


if __name__ == "__main__":
    _run()
