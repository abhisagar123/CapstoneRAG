"""Chunking package — one file per strategy.

Layout:
  base.py                     shared contract: Chunk dataclass + Chunker interface + helpers
  fixed_chunker.py            FixedChunker            (type "fixed")    — fixed word window
  noop_chunker.py             NoOpChunker             (type "none")     — whole doc = one chunk
  paragraph_group_chunker.py  ParagraphGroupChunker   (type "pgc")      — structure-aware (paper's PGC)
  sac_chunker.py              SACChunker              (type "sac")      — fixed-char + prepended summary (LegalBench-RAG)
  semantic_chunker.py         SemanticChunker         (type "semantic") — cut where MEANING shifts (HEAVY)
  (planned: llm_chunker.py    LLMChunker              (type "llm")      — cut where an LLM says (HEAVY))

Two tiers (mirrors embeddings/ and generation/):
  * LIGHT strategies (pure Python — fixed/none/pgc) self-register on `import src`.
  * HEAVY strategies (semantic needs an embedding model = torch; a future llm
    chunker would need a running Ollama server) register only via load_chunkers(),
    so `import src` stays free of the ML stack (AI_CONTEXT §18 brick-2/3 rule).

To ADD a strategy:
  1. create  src/chunking/<name>_chunker.py  with  @register("chunker", "<name>")  on the class
  2. add one import line below (in the eager block if light, in load_chunkers() if heavy)
That's it — no other file changes; it becomes selectable via config `type: <name>`.

The imports below are REQUIRED: @register only runs when a module is imported,
so each strategy file must be imported for it to appear in the registry
(see AI_CONTEXT §18 brick-2 gotcha).
"""

from .base import Chunk, Chunker, resolve_doc_ids  # noqa: F401  — shared contract re-exported

# LIGHT strategies — self-register on import (pure Python, no heavy deps).
from . import fixed_chunker            # noqa: F401  — registers FixedChunker ("fixed")
from . import noop_chunker             # noqa: F401  — registers NoOpChunker ("none")
from . import paragraph_group_chunker  # noqa: F401  — registers ParagraphGroupChunker ("pgc")
from . import sac_chunker              # noqa: F401  — registers SACChunker ("sac")

# Convenience re-exports (so `from src.chunking import FixedChunker` still works).
from .fixed_chunker import FixedChunker                      # noqa: F401
from .noop_chunker import NoOpChunker                        # noqa: F401
from .paragraph_group_chunker import ParagraphGroupChunker   # noqa: F401
from .sac_chunker import SACChunker                          # noqa: F401


def load_chunkers() -> None:
    """Import the HEAVY chunker(s) so they register. Call before building
    "semantic" (loads an embedding model = torch).

    Light chunkers (fixed/none/pgc) are already registered by `import src`; this
    only adds the resource-hungry ones, keeping `import src` torch-free.
    """
    from . import semantic_chunker  # noqa: F401 — registers SemanticChunker ("semantic")
