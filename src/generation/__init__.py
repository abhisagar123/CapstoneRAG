"""Generation package — one file per backend.

Layout:
  base.py            Generator interface
  echo_generator.py  EchoGenerator (type "echo")          — light, no model
  hf_generator.py    HuggingFaceGenerator (type "hf")      — heavy (transformers), Colab

HEAVY-DEP SPLIT (same idea as embeddings/ and reranking/):
- EchoGenerator is imported here, so it registers on `import src` (lets the full
  pipeline run end-to-end locally with no model).
- HuggingFaceGenerator pulls in transformers/torch, so it is NOT imported here.
  Call load_generators() once before building it (done on Colab, per policy).
"""

from .base import Generator  # noqa: F401 — light contract

from . import echo_generator  # noqa: F401 — registers EchoGenerator ("echo"), no model
from .echo_generator import EchoGenerator  # noqa: F401


def load_generators() -> None:
    """Import the heavy generator(s) so they register. Call before building "hf".

    Pays the transformers/torch import cost only when invoked (on Colab).
    """
    from . import hf_generator  # noqa: F401 — registers HuggingFaceGenerator ("hf")
