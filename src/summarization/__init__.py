"""Summarization (context compression) package — one file per strategy.

The pipeline stage between repacking and the prompt: it trims each retrieved
chunk down to the sentences most relevant to the query, handing the generator
less but cleaner context. This is the paper's "Summarization / Recomp" module
(Best Practices §3.7) — the one stage of its recommended recipe we hadn't built.
We implement the EXTRACTIVE flavour only (select sentences verbatim, never
rewrite) so it cannot introduce ungrounded text. See base.py for the full rationale.

Layout:
  base.py                              contract: Summarizer + select_top_indices() helper
  noop_summarizer.py                   NoOpSummarizer       (type "none")    — passthrough baseline
  lexical_summarizer.py                LexicalSummarizer    (type "lexical") — query-word overlap, NO model
  extractive_embedding_summarizer.py   ExtractiveEmbeddingSummarizer
                                       (type "extractive_embedding") — cosine(query, sentence) (HEAVY)

Two tiers (mirrors chunking/ and embeddings/):
  * LIGHT strategies (pure Python — none/lexical) self-register on `import src`.
  * HEAVY strategy (extractive_embedding needs an embedding model = torch)
    registers only via load_summarizers(), so `import src` stays free of the ML
    stack (AI_CONTEXT §18 brick-2/3 rule).

To ADD a strategy:
  1. create src/summarization/<name>_summarizer.py with @register("summarizer", "<name>")
  2. add one import line below (eager block if light, load_summarizers() if heavy)
"""

from .base import Summarizer, select_top_indices, keep_sentences  # noqa: F401 — shared contract

# LIGHT strategies — self-register on import (pure Python, no heavy deps).
from . import noop_summarizer     # noqa: F401 — registers NoOpSummarizer ("none")
from . import lexical_summarizer  # noqa: F401 — registers LexicalSummarizer ("lexical")

# Convenience re-exports.
from .noop_summarizer import NoOpSummarizer        # noqa: F401
from .lexical_summarizer import LexicalSummarizer  # noqa: F401


def load_summarizers() -> None:
    """Import the HEAVY summarizer(s) so they register. Call before building
    "extractive_embedding" (loads an embedding model = torch).

    Light summarizers (none/lexical) are already registered by `import src`; this
    only adds the resource-hungry one, keeping `import src` torch-free.
    """
    from . import extractive_embedding_summarizer  # noqa: F401 — registers "extractive_embedding"
