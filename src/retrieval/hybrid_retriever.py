"""HybridRetriever — fuse dense + BM25 rankings with weighted Reciprocal Rank Fusion.

Dense retrieval matches meaning; BM25 matches words. Hybrid runs BOTH and merges
their ranked lists, so a chunk that either method ranks highly surfaces. This is the
"Best Practices in RAG" paper's top retrieval recommendation.

Fusion = weighted RRF (Reciprocal Rank Fusion):
    score(chunk) = w_dense · 1/(k_rrf + rank_dense) + w_sparse · 1/(k_rrf + rank_sparse)

Why RRF (rank-based) instead of blending the raw scores: dense cosine (~0..1) and
BM25 (unbounded) live on different scales, so adding their scores needs fragile
normalization. RRF uses only the RANK each method assigns, so no normalization — and
a chunk missing from one list simply contributes 0 from that side.

The weights make it configurable AND degenerate cleanly at the extremes (your call):
  - w_dense=1, w_sparse=1  -> standard RRF (the paper's plain version) — DEFAULT
  - w_sparse=0             -> dense only;   w_dense=0 -> BM25 only
  - tune the pair to shift the dense/sparse balance (the paper found the balance matters)

k_rrf (default 60, the canonical value) dampens how much top ranks dominate.

LIGHT at import (the heavy embedder lives in the dense sub-retriever, built elsewhere).
Registered as retriever type "hybrid". Built specially in the pipeline (it wraps the
embedder+index like dense, plus its own BM25), not via a bare registry call.
"""

from ..registry import register
from ..indexing import RetrievedChunk
from .dense_retriever import DenseRetriever
from .bm25_retriever import BM25Retriever


@register("retriever", "hybrid")
class HybridRetriever:
    def __init__(self, embedder, index, *, w_dense: float = 1.0, w_sparse: float = 1.0,
                 k_rrf: int = 60):
        self.dense = DenseRetriever(embedder=embedder, index=index)
        self.sparse = BM25Retriever()
        self.w_dense = w_dense
        self.w_sparse = w_sparse
        self.k_rrf = k_rrf

    def index_corpus(self, chunks: list) -> None:
        """Forward the corpus to the BM25 sub-retriever (dense uses the vector index)."""
        self.sparse.index_corpus(chunks)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Fuse dense + BM25 top-k via weighted RRF, return the fused top-k.

        We pull k from EACH side (a chunk strong in only one list still deserves a shot),
        fuse by RRF over their ranks, then return the k highest-fused chunks. Chunk
        identity is its `chunk_id` (the same chunk can appear in both lists)."""
        dense_hits = self.dense.retrieve(query, k=k)
        sparse_hits = self.sparse.retrieve(query, k=k)

        fused: dict = {}          # chunk_id -> [rrf_score, RetrievedChunk]
        for hits, weight in ((dense_hits, self.w_dense), (sparse_hits, self.w_sparse)):
            if weight == 0:
                continue
            for rc in hits:
                contribution = weight / (self.k_rrf + rc.rank)
                cid = rc.chunk.chunk_id
                if cid in fused:
                    fused[cid][0] += contribution
                else:
                    fused[cid] = [contribution, rc]

        ranked = sorted(fused.values(), key=lambda pair: pair[0], reverse=True)[:k]
        # Re-stamp score (= fused RRF score) and rank (0-based) on the returned chunks,
        # so downstream (reranker/repacker) sees a clean, consistent ordering.
        return [RetrievedChunk(chunk=rc.chunk, score=float(rrf), rank=rank)
                for rank, (rrf, rc) in enumerate(ranked)]
