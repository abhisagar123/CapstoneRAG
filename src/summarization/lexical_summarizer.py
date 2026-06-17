"""LexicalSummarizer — keep the sentences that share the most words with the query.

The MODEL-FREE compression baseline (no torch, no embedder). It scores each
sentence by how many distinct query words it contains (a tiny BM25-flavoured
overlap, length-normalized so long sentences don't win by sheer size), keeps the
top `ratio` fraction per chunk, and drops the rest.

Why it exists: it's the cheap control for the embedding compressor. If
lexical-overlap compression moves TRACe as much as the embedding version, then
the expensive sentence-embedding step isn't buying us anything — exactly the kind
of "is the heavy component worth it?" check we ran for embedders (BGE/mxbai were
null vs MiniLM). One-variable-honest: same select-top-sentences mechanism as the
embedding arm, only the SCORING differs (word overlap vs cosine).

Light (pure Python + the regex splitter) → self-registers on `import src`.
Registered as summarizer type "lexical".
"""

import re

from ..registry import register
from ..indexing import RetrievedChunk
from .base import select_top_indices, keep_sentences

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercased distinct word tokens (set → presence, not frequency)."""
    return set(_WORD.findall(text.lower()))


@register("summarizer", "lexical")
class LexicalSummarizer:
    def __init__(self, ratio: float = 0.5, min_keep: int = 1):
        """ratio: fraction of each chunk's sentences to KEEP (0.5 → the better half).
        min_keep: never trim a chunk below this many sentences (≥1 → never empty)."""
        if not (0.0 < ratio <= 1.0):
            raise ValueError("ratio must be in (0, 1]")
        if min_keep < 1:
            raise ValueError("min_keep must be >= 1")
        self.ratio = ratio
        self.min_keep = min_keep
        self._splitter = None        # lazy regex splitter (no heavy deps)

    def _split(self, text: str) -> list[str]:
        if self._splitter is None:
            from ..segmentation.regex_splitter import RegexSplitter
            self._splitter = RegexSplitter()
        return self._splitter.split(text)

    def _score(self, sentences: list[str], q_tokens: set[str]) -> list[float]:
        """Overlap of each sentence's words with the query, length-normalized:
        |sentence ∩ query| / sqrt(|sentence|). sqrt keeps it from collapsing to
        "longest sentence wins" while still rewarding sentences dense in query terms."""
        scores = []
        for s in sentences:
            toks = _tokens(s)
            overlap = len(toks & q_tokens)
            scores.append(overlap / (len(toks) ** 0.5) if toks else 0.0)
        return scores

    def compress(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        q_tokens = _tokens(query)
        if not q_tokens:                              # empty query → nothing to score on
            return list(chunks)
        out = []
        for rc in chunks:
            sentences = self._split(rc.chunk.text)
            if len(sentences) <= self.min_keep:       # too short to trim → keep as-is
                out.append(rc)
                continue
            scores = self._score(sentences, q_tokens)
            keep = select_top_indices(scores, self.ratio, self.min_keep)
            new_text = keep_sentences(rc.chunk.text, sentences, keep)
            # Preserve provenance + score/rank; only the text shrinks. Chunk is frozen
            # → build a replacement via dataclasses.replace.
            import dataclasses
            new_chunk = dataclasses.replace(rc.chunk, text=new_text)
            out.append(dataclasses.replace(rc, chunk=new_chunk))
        return out
