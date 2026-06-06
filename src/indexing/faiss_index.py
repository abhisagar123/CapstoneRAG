"""FaissIndex — exact inner-product vector index backed by FAISS.

Backend: FAISS `IndexFlatIP` = a FLAT (exact, brute-force) inner-product index.
At our scale (≤ tens of thousands of chunks per domain) exact search is fast and
needs no approximation. FAISS stores only vectors + integer ids, so we keep the
Chunk objects in a parallel list and map ids → chunks ourselves.

Registered as index type "faiss". faiss is imported lazily (only when an index
is actually constructed), so it is not pulled in by light package imports.
"""

from ..registry import register
from ..chunking import Chunk
from .base import RetrievedChunk


@register("index", "faiss")
class FaissIndex:
    """Exact inner-product vector index over chunks.

    `corpus_mode` is recorded metadata (the caller decides which chunks to add);
    it lets a run record which setting produced it:
        "per_example" → only one question's documents were added
        "pooled"      → a whole domain's documents were added
    """

    def __init__(self, dim: int, corpus_mode: str = "per_example"):
        import faiss  # lazy: FAISS only needed when an index is actually built
        if corpus_mode not in ("per_example", "pooled"):
            raise ValueError(f"corpus_mode must be 'per_example' or 'pooled', got {corpus_mode!r}")
        self.dim = dim
        self.corpus_mode = corpus_mode
        self._index = faiss.IndexFlatIP(dim)   # flat = exact; IP = inner product
        self._chunks: list[Chunk] = []         # parallel list: faiss id i ↔ _chunks[i]

    def add(self, chunks: list[Chunk], vectors) -> None:
        """Add chunks and their (n, dim) vectors. Call once (or repeatedly to grow)."""
        import numpy as np
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks vs {len(vectors)} vectors")
        if len(chunks) == 0:
            return
        vecs = np.asarray(vectors, dtype="float32")
        if vecs.shape[1] != self.dim:
            raise ValueError(f"vector dim {vecs.shape[1]} != index dim {self.dim}")
        self._index.add(vecs)
        self._chunks.extend(chunks)

    def search(self, query_vector, k: int = 5) -> list[RetrievedChunk]:
        """Return the k chunks whose vectors are nearest the query vector."""
        import numpy as np
        if len(self._chunks) == 0:
            return []
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        k = min(k, len(self._chunks))          # can't return more than we hold
        scores, ids = self._index.search(q, k)
        out = []
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):
            if idx == -1:                      # faiss pads with -1 if fewer than k
                continue
            out.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score), rank=rank))
        return out

    def __len__(self) -> int:
        return len(self._chunks)
