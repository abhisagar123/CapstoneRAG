"""OFFLINE checks for the Pipeline (src/pipeline.py).

Uses a FAKE embedder (deterministic random vectors, no model) + echo generator
so the full assembly + chain run locally with no model (policy-safe). Real
embed/generate is exercised on Colab.

Run directly:  python tests/check_pipeline.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import src  # noqa: F401
from src.registry import register, available
from src.config import from_dict
from src.pipeline import build_pipeline, Pipeline


# A model-free embedder so pipeline tests need no downloads. Registered once.
if "fake" not in available("embedder"):
    @register("embedder", "fake")
    class _FakeEmbedder:
        dim = 16
        def embed(self, texts):
            out = []
            for t in texts:
                rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
                v = rng.standard_normal(self.dim).astype("float32")
                out.append(v / (np.linalg.norm(v) + 1e-9))
            return np.array(out)


BASE = {
    "domain": "Legal",
    "chunker":   {"type": "fixed", "size": 40, "overlap": 5},
    "embedder":  {"type": "fake"},
    "index":     {"type": "faiss", "corpus_mode": "per_example"},
    "retriever": {"type": "dense", "k": 8},
    "reranker":  {"type": "none", "top_n": 3},
    "repacker":  {"type": "reverse"},
    "prompt":    {"type": "grounded"},
    "generator": {"type": "echo"},
    "splitter":  {"type": "regex"},
}

DOCS = [
    "Either party may terminate with thirty days notice. Notice must be written. "
    "Termination does not affect accrued obligations.",
    "This agreement is governed by Delaware law. Disputes go to arbitration.",
]


def _pipe(overrides=None):
    cfg = from_dict({**BASE, **(overrides or {})})
    return build_pipeline(cfg)


def test_build_assembles_all_components():
    p = _pipe()
    assert type(p.chunker).__name__ == "FixedChunker"
    assert type(p.index).__name__ == "FaissIndex"
    assert type(p.retriever).__name__ == "DenseRetriever"
    assert type(p.reranker).__name__ == "NoOpReranker"
    assert type(p.repacker).__name__ == "ReverseRepacker"
    assert type(p.generator).__name__ == "EchoGenerator"


def test_runtime_knobs_extracted_not_passed_to_ctor():
    # The bug we hit: k/top_n must be pulled out, not passed to component __init__.
    p = _pipe()
    assert p.k == 8           # from retriever.params
    assert p.top_n == 3       # from reranker.params


def test_index_documents_returns_chunk_count():
    p = _pipe()
    n = p.index_documents(DOCS)
    assert n > 0 and len(p.index) == n


def test_answer_runs_full_chain():
    p = _pipe()
    p.index_documents(DOCS)
    out = p.answer("What is the termination notice period?")
    assert {"answer", "sources", "context"}.issubset(out)
    assert out["answer"].startswith("[ECHO]")
    assert 0 < len(out["sources"]) <= p.top_n     # rerank/truncate respected top_n
    assert isinstance(out["context"], str) and len(out["context"]) > 0


def test_optional_stages_skipped_when_none():
    p = _pipe({"reranker": None, "repacker": None})
    assert p.reranker is None and p.repacker is None
    p.index_documents(DOCS)
    out = p.answer("termination?")               # must still run end-to-end
    assert out["answer"].startswith("[ECHO]")
    assert len(out["sources"]) <= p.top_n         # falls back to default top_n


def test_swap_changes_components_same_code():
    p1 = _pipe({"repacker": {"type": "forward"}})
    p2 = _pipe({"repacker": {"type": "reverse"}})
    assert type(p1.repacker).__name__ == "ForwardRepacker"
    assert type(p2.repacker).__name__ == "ReverseRepacker"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} pipeline checks passed.")


if __name__ == "__main__":
    _run()
