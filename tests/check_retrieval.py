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
from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever, Retriever
from src.chunking import Chunk

WANT_MODEL = os.environ.get("MODEL") == "1"


# ---------------- OFFLINE (faiss + numpy, no model) ----------------

def test_index_and_retriever_registered():
    assert "faiss" in available("index")
    assert {"dense", "bm25", "hybrid"} <= set(available("retriever"))


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


# ---------------- BM25 + HYBRID (no model — lexical / fused) ----------------

def _text_chunk(i, text):
    return Chunk(text=text, doc_id="0", chunk_id=f"0-{i}")


_DOCS = [_text_chunk(0, "the termination clause requires thirty days written notice"),
         _text_chunk(1, "payment is due net thirty days after invoice"),
         _text_chunk(2, "a banana smoothie needs bananas milk and ice")]


def test_bm25_retrieves_by_lexical_match():
    r = BM25Retriever()
    assert isinstance(r, Retriever)
    assert r.retrieve("anything", k=3) == []          # before index_corpus → empty, no crash
    r.index_corpus(_DOCS)
    hits = r.retrieve("termination notice", k=3)
    assert hits[0].chunk.chunk_id == "0-0"            # lexical hit on the termination chunk
    assert [h.rank for h in hits] == list(range(len(hits)))   # ranks 0,1,2...
    # a rare exact term beats a topically-vague query
    assert r.retrieve("banana", k=1)[0].chunk.chunk_id == "0-2"


class _KeywordEmbedder:
    """Fake embedder: vector points at whichever keyword the text contains, so dense
    similarity is predictable without a model."""
    dim = 3
    _axes = {"termination": [1, 0, 0], "payment": [0, 1, 0], "banana": [0, 0, 1]}
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0, 0.0, 0.0]
            for kw, axis in self._axes.items():
                if kw in t:
                    v = [float(a) for a in axis]
            out.append(v if any(v) else [0.0, 0.0, 0.0])
        return np.array(out, dtype="float32")


def _hybrid(w_dense, w_sparse):
    emb = _KeywordEmbedder()
    idx = FaissIndex(dim=3)
    idx.add(_DOCS, emb.embed([c.text for c in _DOCS]))
    h = HybridRetriever(embedder=emb, index=idx, w_dense=w_dense, w_sparse=w_sparse)
    h.index_corpus(_DOCS)
    return h


def test_hybrid_fuses_dense_and_sparse():
    h = _hybrid(1.0, 1.0)
    assert isinstance(h, Retriever)
    hits = h.retrieve("termination", k=3)
    assert hits[0].chunk.chunk_id == "0-0"            # both dense + bm25 agree → fused #1
    assert [x.rank for x in hits] == list(range(len(hits)))


def test_hybrid_weights_degenerate_to_single_retriever():
    # w_sparse=0 → dense only; w_dense=0 → bm25 only. The configurable extremes.
    dense_only = _hybrid(1.0, 0.0).retrieve("payment", k=3)
    bm25_only = _hybrid(0.0, 1.0).retrieve("payment", k=3)
    assert dense_only[0].chunk.chunk_id == "0-1"      # dense maps "payment" → chunk 1
    assert bm25_only[0].chunk.chunk_id == "0-1"       # bm25 lexical "payment" → chunk 1
    # with w_sparse=0, a purely-lexical-only term shouldn't pull a dense-blind chunk:
    # "invoice" appears only in chunk 1's text (lexical), dense embedder has no axis for it
    bm25_hit = _hybrid(0.0, 1.0).retrieve("invoice", k=1)
    assert bm25_hit[0].chunk.chunk_id == "0-1"


def test_hybrid_rrf_score_magnitude():
    # A chunk ranked #0 by BOTH lists, equal weights, k_rrf=60 → 2 * 1/(60+0) ≈ 0.0333.
    h = _hybrid(1.0, 1.0)
    top = h.retrieve("termination", k=1)[0]
    assert abs(top.score - (1/60 + 1/60)) < 1e-6


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
               test_dense_retriever_uses_its_embedder_and_index,
               test_bm25_retrieves_by_lexical_match,
               test_hybrid_fuses_dense_and_sparse,
               test_hybrid_weights_degenerate_to_single_retriever,
               test_hybrid_rrf_score_magnitude]
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
