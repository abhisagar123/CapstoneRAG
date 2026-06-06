"""Retrieval package — one file per strategy (same convention as chunking/).

Layout:
  base.py              shared contract: Retriever interface
  dense_retriever.py   DenseRetriever (type "dense")

To ADD a strategy (e.g. BM25 / hybrid):
  1. create  src/retrieval/bm25_retriever.py  with  @register("retriever", "bm25")
  2. add one import line below so its decorator runs on import

Safe to import eagerly (no heavy deps at import time; the embedder it uses is
constructed elsewhere).
"""

from .base import Retriever  # noqa: F401 — shared contract

from . import dense_retriever  # noqa: F401 — registers DenseRetriever ("dense")

from .dense_retriever import DenseRetriever  # noqa: F401 — convenience re-export
