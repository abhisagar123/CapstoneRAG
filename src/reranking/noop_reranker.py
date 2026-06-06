"""NoOpReranker — the "no reranking" control arm (no model).

Trusts the retriever's existing order and simply truncates to top_n. This is the
ablation baseline: comparing it against a real reranker answers "is reranking
worth the extra cost?". Kept in its own light file so it registers without
importing the cross-encoder / torch stack.

Registered as reranker type "none".
"""

from ..registry import register
from ..indexing import RetrievedChunk


@register("reranker", "none")
class NoOpReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_n: int = 5) -> list[RetrievedChunk]:
        # Keep the retriever's order; just take the first top_n.
        return list(chunks[:top_n])
