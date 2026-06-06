"""Embeddings package — one file per strategy (model family).

Layout:
  base.py                            shared contract: Embedder interface
  sentence_transformer_embedder.py   SentenceTransformerEmbedder (types
                                     "sentence_transformer" / "minilm")

⚠️ HEAVY-DEPENDENCY EXCEPTION to the usual package convention:
Unlike the chunking package, we do NOT import the implementation here, because
it pulls in torch + sentence-transformers. Eagerly importing it would mean
`import src.embeddings` (or `import src`) drags in the whole ML stack — breaking
the lightweight-import rule (AI_CONTEXT §18 brick-2 gotcha).

Instead, the implementation self-registers when its module is explicitly
imported. Call `load_embedders()` once before you `build("embedder", ...)`, or
import the module directly. This keeps the registry lazy: you only pay the
torch import cost when you actually need an embedder.
"""

from .base import Embedder  # noqa: F401 — light contract only (no torch)


def load_embedders() -> None:
    """Import embedder implementations so they register. Call before building one.

    Pays the torch/sentence-transformers import cost only when invoked — never
    at package-import time.
    """
    from . import sentence_transformer_embedder  # noqa: F401 — registers on import
