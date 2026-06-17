"""Summarization (context compression) — shared contract + a pure helper.

A summarizer sits between repacking and the prompt builder. It does NOT reorder
(that's the repacker) and does NOT re-rank chunks (that's the reranker). It
COMPRESSES: within the chunks we already chose, it keeps the sentences most
relevant to the query and drops the rest — shrinking the context the generator
must read.

Why this stage exists (the bottleneck it targets):
  Our retrieved top-5 is only ~58% relevant text (pooled Exp 1) — every chunk
  carries off-topic sentences. The small generator wades through that padding and
  grounds poorly (the adherence gap, Exp 11/12). Compression hands it LESS but
  CLEANER text. Crucially this is NOT top_n (which drops whole chunks and lost
  needed facts → adherence fell, Exp 2): we keep EVERY chunk and trim only the
  irrelevant SENTENCES inside it (min_keep ≥ 1), so no chunk is ever emptied.

  This is the paper's "Summarization / Recomp" module (Best Practices §3.7), the
  one stage of its recommended recipe we had never built. We implement the
  EXTRACTIVE flavour only — it selects existing sentences verbatim, so it can
  only delete text, never invent it → it cannot hurt adherence by hallucinating
  (the abstractive flavour rewrites and is deliberately omitted).

Contract: compress(query, chunks) -> chunks. Returns NEW RetrievedChunk objects
with shortened text, preserving score/rank/doc_id/chunk_id so the downstream
prompt builder + the runner's segmentation (which reads chunk texts) still work.

base.py holds ONLY the contract + a pure selection helper, so every strategy
(none / lexical / extractive_embedding) honours one interface — same base.py
pattern as chunking/, repacking/, etc.
"""

import math
from typing import Protocol, runtime_checkable

from ..indexing import RetrievedChunk


@runtime_checkable
class Summarizer(Protocol):
    """The contract every summarizer honours."""

    def compress(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Trim each chunk's text to its query-most-relevant sentences.
        Input chunks are the final post-rerank/repack set; output is the same
        chunks with shortened `.chunk.text` (order preserved)."""
        ...


def select_top_indices(scores: list[float], ratio: float, min_keep: int = 1) -> list[int]:
    """Pick which sentences to KEEP, returned in ORIGINAL reading order.

    Pure function (no model, no I/O) — the heart of every extractive strategy, so
    it is unit-tested directly. Given a per-sentence relevance `scores` list:
      * keep the top `ratio` fraction by score (ratio=0.5 → keep the better half),
      * but never fewer than `min_keep` (so a chunk is never fully emptied),
      * never more than all of them,
      * ties + selection resolved by score desc, then the kept set is RE-SORTED to
        original order so the surviving sentences still read in sequence.

    e.g. scores=[0.9,0.1,0.8,0.2], ratio=0.5 -> keep 2 best (idx 0,2) -> [0, 2].
    """
    n = len(scores)
    if n == 0:
        return []
    k = max(min_keep, math.ceil(ratio * n))
    k = min(k, n)                                   # can't keep more than we have
    if k >= n:
        return list(range(n))                       # keep everything (ratio≈1) — cheap path
    # indices of the k highest scores (stable: ties keep earlier sentence first)
    best = sorted(range(n), key=lambda i: (-scores[i], i))[:k]
    return sorted(best)                             # back to reading order


def keep_sentences(chunk_text: str, sentences: list[str], keep_idx: list[int]) -> str:
    """Join the kept sentences back into one text. If everything was somehow
    dropped (shouldn't happen with min_keep≥1) fall back to the original text so a
    chunk is never blanked out."""
    if not keep_idx:
        return chunk_text
    return " ".join(sentences[i] for i in keep_idx)
