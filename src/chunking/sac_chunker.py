"""SACChunker — Summary-Augmented Chunking (Reuter et al. NLLP 2025, §3.2).

The published best-attested chunking config on LegalBench-RAG (which includes
CUAD). Two pieces:

  1. Slide a fixed CHARACTER window of `size` chars (default 500) across each
     document, stepping by (size - overlap) chars. This is the "naive" base
     chunker the paper found beats RCTS on CUAD top-k (P@1 9.27 vs 1.97).

  2. Prepend a `summary_chars`-character (default 150) document-level summary
     to EVERY chunk in that document, separated by a newline. The paper uses
     the first 150 chars of the contract text as a cheap proxy for a real
     summary — this single change roughly halves Document-level Retrieval
     Mismatch (~35% → 19.29%).

Why characters not words: the paper's optima are reported in characters
(200/500/800), and contracts have heavy boilerplate / capitalised headings that
make word counts a poor proxy for chunk size. FixedChunker (this framework's
existing one) is word-based; SAC keeps the paper's units.

Caveat the paper itself flags: BM25 hybrid weighting can over-match the
prepended summaries because identifiers leak across chunks. So SAC is most
clearly a win for DENSE-only retrieval — hybrid configs need their own ablation
(R3 in the Legal Phase A plan).

Registered as chunker type "sac":  {type: sac, size: 500, overlap: 0, summary_chars: 150}
"""

from ..registry import register
from .base import Chunk, resolve_doc_ids


@register("chunker", "sac")
class SACChunker:
    def __init__(self, size: int = 500, overlap: int = 0, summary_chars: int = 150):
        if size <= 0:
            raise ValueError("size must be positive")
        if not (0 <= overlap < size):
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        if summary_chars < 0:
            raise ValueError("summary_chars must be >= 0")
        self.size = size
        self.overlap = overlap
        self.summary_chars = summary_chars

    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        ids = resolve_doc_ids(documents, doc_ids)
        step = self.size - self.overlap
        chunks: list[Chunk] = []
        for doc_id, text in zip(ids, documents):
            text = text.strip()
            if not text:
                continue
            summary = text[: self.summary_chars].replace("\n", " ").strip()
            prefix = f"{summary}\n\n" if summary else ""
            i = 0
            n = 0
            while i < len(text):
                window = text[i : i + self.size]
                piece = f"{prefix}{window}"
                chunks.append(Chunk(text=piece, doc_id=doc_id, chunk_id=f"{doc_id}-{n}"))
                n += 1
                if i + self.size >= len(text):
                    break
                i += step
        return chunks
