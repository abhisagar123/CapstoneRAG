"""ParagraphGroupChunker (PGC) — structure-aware chunking: group adjacent paragraphs.

Where FixedChunker slices at an arbitrary WORD count (cutting mid-sentence / mid-idea),
PGC respects the document's own structure: it splits text into paragraphs (on blank
lines) and groups `paragraphs` adjacent ones per chunk, sliding with `overlap`
paragraphs of carry-over. So chunk boundaries fall at natural paragraph breaks, keeping
each chunk topically coherent.

This is the "Document Chunking Strategies" paper's BEST overall chunker (PGC; their
default paragraphs=2, overlap=1) and the paper's top pick for Legal. The hypothesis for
our pooled track: coherent paragraph-sized chunks raise the relevant-FRACTION (less
off-topic text bundled in) without over-fragmenting the way tiny fixed windows do.

Fallback: a document with no blank-line structure (one big block) becomes a single
paragraph → one chunk. (A future variant could fall back to sentence grouping; kept
simple here — that's a separate strategy file if we want it.)

Registered as chunker type "pgc":  {type: pgc, paragraphs: 2, overlap: 1}
"""

import re

from ..registry import register
from .base import Chunk, resolve_doc_ids

# A paragraph break = one or more blank lines (two+ newlines, allowing intervening
# whitespace). Falls back to single-newline splitting only if no blank-line breaks exist.
_BLANK_LINE = re.compile(r"\n\s*\n")


def _paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines; fall back to single newlines if the
    document has no blank-line structure; final fallback = the whole text as one para."""
    paras = [p.strip() for p in _BLANK_LINE.split(text) if p.strip()]
    if len(paras) <= 1:
        # no blank-line structure — try single newlines (many corpora use \n per para)
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras or ([text.strip()] if text.strip() else [])


@register("chunker", "pgc")
class ParagraphGroupChunker:
    def __init__(self, paragraphs: int = 2, overlap: int = 1):
        if paragraphs <= 0:
            raise ValueError("paragraphs must be positive")
        if not (0 <= overlap < paragraphs):
            raise ValueError("overlap must satisfy 0 <= overlap < paragraphs")
        self.paragraphs = paragraphs
        self.overlap = overlap

    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        ids = resolve_doc_ids(documents, doc_ids)
        step = self.paragraphs - self.overlap
        chunks: list[Chunk] = []
        for doc_id, text in zip(ids, documents):
            paras = _paragraphs(text)
            if not paras:
                continue                              # skip empty documents
            i = 0
            n = 0
            while i < len(paras):
                piece = "\n\n".join(paras[i:i + self.paragraphs])
                chunks.append(Chunk(text=piece, doc_id=doc_id, chunk_id=f"{doc_id}-{n}"))
                n += 1
                if i + self.paragraphs >= len(paras):  # last group covered the tail; stop
                    break
                i += step
        return chunks
