"""Indexing package — one file per backend (same convention as chunking/).

Layout:
  base.py          shared contract: Index interface + RetrievedChunk result type
  faiss_index.py   FaissIndex (type "faiss")

To ADD a backend (e.g. Chroma):
  1. create  src/indexing/chroma_index.py  with  @register("index", "chroma")
  2. add one import line below so its decorator runs on import

Safe to import eagerly: faiss_index imports `faiss` lazily INSIDE __init__, so
importing this package does NOT pull in faiss/numpy heavy work — it only files
the class into the registry. (Contrast embeddings/, where the torch import is at
construction and we gate registration behind load_embedders().)
"""

from .base import Index, RetrievedChunk  # noqa: F401 — shared contract

from . import faiss_index  # noqa: F401 — registers FaissIndex ("faiss")

from .faiss_index import FaissIndex  # noqa: F401 — convenience re-export
