"""Chunking package — one file per strategy.

Layout:
  base.py            shared contract: Chunk dataclass + Chunker interface + helpers
  fixed_chunker.py   FixedChunker  (type "fixed")
  noop_chunker.py    NoOpChunker   (type "none")

To ADD a strategy (e.g. a teammate's PGC chunker):
  1. create  src/chunking/pgc_chunker.py  with  @register("chunker", "pgc")  on the class
  2. add one import line below so its decorator runs on import
That's it — no other file changes; it becomes selectable via config `type: pgc`.

The imports below are REQUIRED: @register only runs when a module is imported,
so each strategy file must be imported here for it to appear in the registry
(see AI_CONTEXT §18 brick-2 gotcha).
"""

from .base import Chunk, Chunker, resolve_doc_ids  # noqa: F401  — shared contract re-exported

# Import each strategy so it self-registers. Add new strategies here.
from . import fixed_chunker  # noqa: F401  — registers FixedChunker ("fixed")
from . import noop_chunker   # noqa: F401  — registers NoOpChunker ("none")

# Convenience re-exports (so `from src.chunking import FixedChunker` still works).
from .fixed_chunker import FixedChunker  # noqa: F401
from .noop_chunker import NoOpChunker    # noqa: F401
