"""Checks for the index + retriever (src/indexer.py, src/retriever.py).

Split into:
  * OFFLINE checks — registry wiring, the FAISS index's add/search math using
    hand-made vectors (needs faiss + numpy, but NO model download). Run always.
  * MODEL checks — full text→chunk→embed→index→retrieve on a real model.
    Gated behind MODEL=1:  MODEL=1 python tests/check_retrieval.py

Run directly:  python tests/check_retrieval.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import src  # noqa: F401
from src.registry import build, available
from src.indexing import FaissIndex, RetrievedChunk
from src.retrieval import DenseRetriever, Retriever
from src.chunking import Chunk

WANT_MODEL = os.environ.get("MODEL") == "1"


# ---------------- OFFLINE (faiss + numpy, no model) ----------------

def test_index_and_retriever_registered():
    assert "faiss" in available("index")
    assert "dense" in available("retriever")


def _chunk(i):
    return Chunk(text=f"chunk {i}", doc_id="0", chunk_id=f"0-{i}")


def test_index_search_returns_nearest_in_order():
    # 3 orthonormal vectors; a query equal to vector 1 must rank chunk 1 first.
    idx = FaissIndex(dim=3)
    vecs = np.eye(3, dtype="float32")            # [[1,0,0],[0,1,0],[0,0,1]]
    idx.add([_chunk(0), _chunk(1), _chunk(2)], vecs)
    hits = idx.search(np.array([0, 1, 0], dtype="float32"), k=3)
    assert [h.chunk.chunk_id for h in hits][0] == "0-1"     # exact match first
    assert [h.rank for h in hits] == [0, 1, 2]              # ranks assigned in order
    assert hits[0].score >= hits[1].score >= hits[2].score # scores descending


def test_index_len_and_empty_search():
    idx = FaissIndex(dim=4)
    assert len(idx) == 0
    assert idx.search(np.ones(4, dtype="float32"), k=3) == []   # empty index → no hits


def test_k_larger_than_corpus_is_safe():
    idx = FaissIndex(dim=2)
    idx.add([_chunk(0)], np.array([[1, 0]], dtype="float32"))
    hits = idx.search(np.array([1, 0], dtype="float32"), k=10)   # ask for 10, hold 1
    assert len(hits) == 1


def test_dim_and_count_mismatches_raise():
    idx = FaissIndex(dim=4)
    try:
        idx.add([_chunk(0)], np.ones((1, 3), dtype="float32"))   # wrong dim
        assert False
    except ValueError:
        pass
    try:
        idx.add([_chunk(0), _chunk(1)], np.ones((1, 4), dtype="float32"))  # count mismatch
        assert False
    except ValueError:
        pass


def test_corpus_mode_validated():
    FaissIndex(dim=4, corpus_mode="pooled")          # ok
    try:
        FaissIndex(dim=4, corpus_mode="everything")  # invalid
        assert False
    except ValueError:
        pass


def test_dense_retriever_uses_its_embedder_and_index():
    # Fake embedder + tiny index: retriever should embed the query and search.
    class FakeEmbedder:
        dim = 2
        def embed(self, texts):
            # map "match" → [1,0], anything else → [0,1]
            return np.array([[1.0, 0.0] if "match" in t else [0.0, 1.0] for t in texts],
                            dtype="float32")
    idx = FaissIndex(dim=2)
    idx.add([_chunk(0), _chunk(1)],
            np.array([[1, 0], [0, 1]], dtype="float32"))
    r = DenseRetriever(embedder=FakeEmbedder(), index=idx)
    assert isinstance(r, Retriever)
    hit = r.retrieve("please match this", k=1)[0]
    assert hit.chunk.chunk_id == "0-0"               # query [1,0] → chunk 0


# ---------------- MODEL (real embedder + real doc) ----------------

def test_end_to_end_retrieval_finds_relevant():
    from src.embeddings import load_embedders
    from src.data_loader import load_domain
    load_embedders()

    doc = load_domain("Legal", split="test", n=1)[0]["documents"]
    chunks = build("chunker", "fixed", {"size": 120, "overlap": 20}).chunk(doc)
    emb = build("embedder", "minilm")
    idx = build("index", "faiss", {"dim": emb.dim})
    idx.add(chunks, emb.embed([c.text for c in chunks]))
    r = DenseRetriever(embedder=emb, index=idx)

    on = r.retrieve("termination of the agreement", k=1)[0].score
    off = r.retrieve("banana smoothie recipe", k=1)[0].score
    assert on > off                                  # relevant beats irrelevant
    scores = [h.score for h in r.retrieve("payment of fees", k=5)]
    assert scores == sorted(scores, reverse=True)    # ranked


def _run():
    offline = [test_index_and_retriever_registered,
               test_index_search_returns_nearest_in_order,
               test_index_len_and_empty_search,
               test_k_larger_than_corpus_is_safe,
               test_dim_and_count_mismatches_raise,
               test_corpus_mode_validated,
               test_dense_retriever_uses_its_embedder_and_index]
    model = [test_end_to_end_retrieval_finds_relevant]

    for fn in offline:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"{len(offline)} offline checks passed.")

    if WANT_MODEL:
        print("\nrunning model checks (MODEL=1)...")
        for fn in model:
            fn(); print(f"  ✅ {fn.__name__}")
        print(f"{len(model)} model checks passed.")
    else:
        print("\n(skipped model checks — set MODEL=1 to run them)")


if __name__ == "__main__":
    _run()
