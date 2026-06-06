"""NoOpChunker — "no chunking": each whole document becomes a single chunk.

Doubles as (a) the no-chunking ablation arm and (b) the identity baseline that
smarter chunkers are compared against. Empty documents are skipped.

Registered as chunker type "none":  {type: none}
"""

from ..registry import register
from .base import Chunk, resolve_doc_ids


@register("chunker", "none")
class NoOpChunker:
    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        ids = resolve_doc_ids(documents, doc_ids)
        chunks: list[Chunk] = []
        for doc_id, text in zip(ids, documents):
            if text.strip():
                chunks.append(Chunk(text=text, doc_id=doc_id, chunk_id=f"{doc_id}-0"))
        return chunks
