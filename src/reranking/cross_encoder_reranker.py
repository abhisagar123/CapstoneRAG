"""CrossEncoderReranker — the accurate "interviewer" reranker.

A cross-encoder reads the query and a chunk TOGETHER and outputs one relevance
score for that pair (unlike the bi-encoder embedder, which encoded them
separately). We score every candidate, sort by the fresh score, keep the best
top_n, and re-assign ranks 0..n-1.

The returned RetrievedChunks carry the cross-encoder score (a different scale
from the retriever's cosine score) and the NEW rank — so downstream sees the
reranked order with reranker scores, not stale retriever scores.

Heavy dep: imports sentence_transformers.CrossEncoder lazily inside __init__,
so it is NOT loaded by `import src` (registration is gated behind
load_rerankers(); see this package's __init__). Registered type "cross_encoder".
"""

from ..registry import register
from ..indexing import RetrievedChunk

# Small, fast, CPU-friendly reranker (~80MB). Strong models can be passed via model=.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@register("reranker", "cross_encoder")
class CrossEncoderReranker:
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32):
        from sentence_transformers import CrossEncoder  # lazy: only on construction
        self.model_name = model
        self.batch_size = batch_size
        self._model = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_n: int = 5) -> list[RetrievedChunk]:
        if not chunks:
            return []
        # Score every (query, chunk) pair together.
        pairs = [(query, rc.chunk.text) for rc in chunks]
        scores = self._model.predict(pairs, batch_size=self.batch_size,
                                     show_progress_bar=False)
        # Sort candidates by the fresh score (desc), keep top_n, re-rank.
        order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
        out = []
        for new_rank, i in enumerate(order[:top_n]):
            out.append(RetrievedChunk(chunk=chunks[i].chunk,
                                      score=float(scores[i]),
                                      rank=new_rank))
        return out
