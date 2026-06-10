"""Generation package — one file per backend.

Layout:
  base.py              Generator interface
  echo_generator.py    EchoGenerator (type "echo")     — light, no model
  hf_generator.py      HuggingFaceGenerator (type "hf") — heavy (transformers), in-process / Colab
  ollama_generator.py  OllamaGenerator (type "ollama")  — local Ollama server (Mac), light (httpx)

HEAVY-DEP SPLIT (same idea as embeddings/ and reranking/):
- EchoGenerator is imported here, so it registers on `import src` (lets the full
  pipeline run end-to-end locally with no model).
- HuggingFaceGenerator pulls in transformers/torch, so it is NOT imported here.
- OllamaGenerator is light (httpx → local server) but registered in load_generators()
  too, so all real backends are discovered in one place.
Call load_generators() once before building "hf" or "ollama".
"""

from .base import Generator  # noqa: F401 — light contract

from . import echo_generator  # noqa: F401 — registers EchoGenerator ("echo"), no model
from .echo_generator import EchoGenerator  # noqa: F401


def load_generators() -> None:
    """Import the real generator(s) so they register. Call before building "hf"/"ollama".

    "hf" pays the transformers/torch import cost; "ollama" is light (local server).
    """
    from . import hf_generator      # noqa: F401 — registers HuggingFaceGenerator ("hf")
    from . import ollama_generator  # noqa: F401 — registers OllamaGenerator ("ollama")
