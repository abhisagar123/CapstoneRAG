"""Reranking package — one file per strategy.

Layout:
  base.py                    Reranker interface
  noop_reranker.py           NoOpReranker (type "none")          — light, no model
  cross_encoder_reranker.py  CrossEncoderReranker (type          — heavy (torch)
                             "cross_encoder")

HEAVY-DEP SPLIT (same idea as embeddings/):
- The light NoOpReranker is imported here, so it registers on `import src`.
- The CrossEncoderReranker pulls in torch, so it is NOT imported here. Call
  load_rerankers() once before building it. This keeps `import src` free of the
  ML stack while still letting the no-op (ablation control) work out of the box.
"""

from .base import Reranker  # noqa: F401 — light contract

from . import noop_reranker  # noqa: F401 — registers NoOpReranker ("none"), no torch
from .noop_reranker import NoOpReranker  # noqa: F401


def load_rerankers() -> None:
    """Import the heavy reranker(s) so they register. Call before building one.

    Pays the torch / sentence-transformers import cost only when invoked.
    """
    from . import cross_encoder_reranker  # noqa: F401 — registers CrossEncoderReranker
