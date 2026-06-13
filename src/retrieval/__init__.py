"""Retrieval package — one file per strategy (same convention as chunking/).

Layout:
  base.py               shared contract: Retriever interface (+ optional index_corpus hook)
  dense_retriever.py    DenseRetriever  (type "dense")  — vector search
  bm25_retriever.py     BM25Retriever   (type "bm25")   — sparse/lexical (rank-bm25)
  hybrid_retriever.py   HybridRetriever (type "hybrid") — dense + BM25 via weighted RRF

To ADD a strategy:
  1. create  src/retrieval/<name>_retriever.py  with  @register("retriever", "<name>")
  2. add one import line below so its decorator runs on import

Safe to import eagerly (no heavy deps at import time — rank-bm25 is pure-Python and
lazily imported; the embedder the dense/hybrid retrievers use is constructed elsewhere).
"""

from .base import Retriever  # noqa: F401 — shared contract

from . import dense_retriever   # noqa: F401 — registers DenseRetriever ("dense")
from . import bm25_retriever    # noqa: F401 — registers BM25Retriever ("bm25")
from . import hybrid_retriever  # noqa: F401 — registers HybridRetriever ("hybrid")

from .dense_retriever import DenseRetriever    # noqa: F401 — convenience re-exports
from .bm25_retriever import BM25Retriever      # noqa: F401
from .hybrid_retriever import HybridRetriever  # noqa: F401
