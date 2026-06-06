"""FixedChunker — baseline fixed-size sliding-window chunker (measured in WORDS).

Slides a window of `size` words across each document, stepping forward by
(size - overlap) words so consecutive chunks share `overlap` words — this keeps
an idea that straddles a boundary from being cleanly severed. A document shorter
than `size` yields a single chunk (the whole document).

Registered as chunker type "fixed":  {type: fixed, size: 512, overlap: 50}
"""

from ..registry import register
from .base import Chunk, resolve_doc_ids


@register("chunker", "fixed")
class FixedChunker:
    def __init__(self, size: int = 512, overlap: int = 50):
        if size <= 0:
            raise ValueError("size must be positive")
        if not (0 <= overlap < size):
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        self.size = size
        self.overlap = overlap

    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        ids = resolve_doc_ids(documents, doc_ids)
        step = self.size - self.overlap
        chunks: list[Chunk] = []
        for doc_id, text in zip(ids, documents):
            words = text.split()
            if not words:
                continue                          # skip empty documents
            i = 0
            n = 0
            while i < len(words):
                piece = " ".join(words[i:i + self.size])
                chunks.append(Chunk(text=piece, doc_id=doc_id, chunk_id=f"{doc_id}-{n}"))
                n += 1
                if i + self.size >= len(words):   # last window covered the tail; stop
                    break
                i += step
        return chunks
