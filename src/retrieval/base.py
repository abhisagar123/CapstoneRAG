"""Retrieval — shared contract (interface).

The Index works in vectors; the Retriever works in TEXT. A retriever takes a
text query and returns the most relevant chunks — the online half of "search by
meaning". This file holds ONLY the contract so every retrieval strategy (dense
now; sparse/BM25, hybrid/RRF later) honours one interface. See LLD §3.
"""

from typing import Protocol, runtime_checkable

from ..indexing import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """The contract every retriever honours.

    `retrieve()` is the one required method (kept in the runtime_checkable Protocol so
    isinstance() works). There is also an OPTIONAL hook some retrievers implement:

        index_corpus(self, chunks: list) -> None

    A SPARSE retriever (BM25) has no vector Index to lean on — it must build its own
    term index over the chunk TEXTS, which only exist after chunking. The pipeline calls
    `index_corpus(chunks)` inside index_documents() so sparse / hybrid retrievers get the
    corpus; dense retrieval doesn't need it. It is deliberately NOT a Protocol method —
    requiring it would break isinstance() for dense (which omits it) and defeat the
    "optional" intent — so the pipeline calls it defensively via getattr/callable.
    """

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the k most relevant chunks for a text query."""
        ...
