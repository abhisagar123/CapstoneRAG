"""DenseRetriever — pure vector search: embed the query, return nearest chunks.

The baseline retrieval strategy. Sparse (BM25) and hybrid (RRF) retrievers will
be added as sibling files behind the same Retriever interface.

Critical invariant: the query MUST be embedded with the SAME embedder that built
the index (same model → same vector space), or similarities are meaningless.
DenseRetriever holds both the embedder and the index, so they can't drift apart.
"""

from ..registry import register
from ..indexing import RetrievedChunk


@register("retriever", "dense")
class DenseRetriever:
    def __init__(self, embedder, index):
        # embedder: anything with .embed([...]) -> (n, dim)  (an Embedder)
        # index:    a built Index with .search(qvec, k)      (e.g. FaissIndex)
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed([query])[0]   # (dim,) — same model as the index
        return self.index.search(query_vec, k=k)
