"""BM25Retriever — sparse (lexical) retrieval over chunk texts.

Where DenseRetriever matches by *meaning* (embedding similarity), BM25 matches by
*words*: it scores a chunk by how well its terms overlap the query, weighted by
term rarity (Okapi BM25). This catches exact-term hits a dense embedder can blur —
product names, codes, acronyms — which is why hybrid (dense + BM25) often beats
either alone, and why it's the most plausible fix for domains where dense keeps
pulling topically-similar-but-wrong docs.

We use the `rank-bm25` library (BM25Okapi) rather than hand-rolling: BM25 has
subtle correctness details (IDF smoothing, k1/b saturation, negative-IDF epsilon
clamp) that a tested library gets right — a subtly-wrong hand-rolled BM25 wouldn't
crash, it would silently retrieve slightly-off chunks and corrupt the experiment.
The "implement it ourselves" rule applies to the graded TRACe *metrics*, not to
retrieval components (we already use FAISS / sentence-transformers / cross-encoder).

Unlike dense, BM25 has no vector index to lean on, so it builds its own term index
from the chunk texts via `index_corpus(chunks)` — the pipeline calls this after
chunking. LIGHT (rank-bm25 is pure-Python, no torch).

Registered as retriever type "bm25".
"""

from ..registry import register
from ..indexing import RetrievedChunk


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization — matches our word-based chunker's simplicity.
    BM25 is bag-of-words, so this is sufficient; a fancier tokenizer is a later swap."""
    return text.lower().split()


@register("retriever", "bm25")
class BM25Retriever:
    def __init__(self):
        self._bm25 = None
        self._chunks: list = []

    def index_corpus(self, chunks: list) -> None:
        """Build the BM25 term index over the chunk texts. Called by the pipeline after
        chunking (and again on reset_index, so it's rebuilt per-example when needed)."""
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi([_tokenize(ch.text) for ch in self._chunks])

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the k chunks with the highest BM25 score for the query."""
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        k = min(k, len(self._chunks))
        # indices of the top-k scores, highest first
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]), rank=rank)
                for rank, i in enumerate(top)]
