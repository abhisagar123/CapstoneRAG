"""Chunker — slices documents into retrievable pieces (offline index step 1).

Chunking is "step 1" of building the index: a long document is cut into smaller
pieces ("chunks") that each later become a searchable vector. Piece size is a
real lever (too big → retrieved blobs are mostly irrelevant; too small → a single
idea gets split across pieces), so it's a per-domain knob we tune later.

Design (matches LLD §2/§3, registry per LLD §7):
  Chunk        an immutable record: text + provenance (doc_id, chunk_id) + meta
  Chunker      the INTERFACE (contract): .chunk(documents) -> list[Chunk]
  FixedChunker baseline impl: fixed-size sliding window measured in WORDS, with overlap
  NoOpChunker  "no chunking" impl: one chunk per whole document (ablation arm + baseline)

Swap by config: `chunker: {type: fixed, size: 512, overlap: 50}` or `{type: none}`.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .registry import register


@dataclass(frozen=True)
class Chunk:
    text: str               # the chunk's text
    doc_id: str             # which source document it came from (e.g. "0")
    chunk_id: str           # unique id within the index (e.g. "0-2" = doc 0, chunk 2)
    meta: dict = field(default_factory=dict)   # room for later extras (char span, parent id, ...)


@runtime_checkable
class Chunker(Protocol):
    """The contract every chunker honours (the 'socket shape')."""
    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        ...


def _doc_ids(documents: list[str], doc_ids: list[str] | None) -> list[str]:
    """Default each document's id to its index ('0','1',...) unless ids are given."""
    if doc_ids is None:
        return [str(i) for i in range(len(documents))]
    if len(doc_ids) != len(documents):
        raise ValueError(f"doc_ids length {len(doc_ids)} != documents length {len(documents)}")
    return doc_ids


@register("chunker", "fixed")
class FixedChunker:
    """Baseline: slide a fixed-size window of `size` WORDS across each document,
    stepping forward by (size - overlap) words so consecutive chunks share
    `overlap` words (so an idea straddling a boundary isn't cleanly severed).

    A document shorter than `size` yields a single chunk (the whole document).
    """
    def __init__(self, size: int = 512, overlap: int = 50):
        if size <= 0:
            raise ValueError("size must be positive")
        if not (0 <= overlap < size):
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        ids = _doc_ids(documents, doc_ids)
        step = self.size - self.overlap
        chunks: list[Chunk] = []
        for doc_id, text in zip(ids, documents):
            words = text.split()
            if not words:
                continue                         # skip empty documents
            i = 0
            n = 0
            while i < len(words):
                piece = " ".join(words[i:i + self.size])
                chunks.append(Chunk(text=piece, doc_id=doc_id, chunk_id=f"{doc_id}-{n}"))
                n += 1
                if i + self.size >= len(words):  # last window covered the tail; stop
                    break
                i += step
        return chunks


@register("chunker", "none")
class NoOpChunker:
    """'No chunking': return each whole document as a single chunk.

    Doubles as (a) the no-chunking ablation arm and (b) the identity baseline
    that smarter chunkers are compared against. Empty documents are skipped.
    """
    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        ids = _doc_ids(documents, doc_ids)
        chunks: list[Chunk] = []
        for doc_id, text in zip(ids, documents):
            if text.strip():
                chunks.append(Chunk(text=text, doc_id=doc_id, chunk_id=f"{doc_id}-0"))
        return chunks
