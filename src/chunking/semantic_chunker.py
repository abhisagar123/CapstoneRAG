"""SemanticChunker — cut where the MEANING shifts, not at a fixed word count.

FixedChunker cuts every N words (blind to content); PGC cuts on blank lines
(structure, but still not meaning). This chunker cuts where the *topic* changes,
measured with embeddings:

    doc:  "The cat sat on the mat. It purred softly. Revenue rose 12%. Costs fell."
           └─ s0 ──────────────┘ └─ s1 ──────┘ └─ s2 ───────────┘ └─ s3 ───┘

    1. split into sentences (reuse the regex splitter — no new dependency)
    2. embed each sentence (reuse our normalized SentenceTransformerEmbedder)
    3. cosine similarity of each NEIGHBOUR pair:
          sim(s0,s1) ≈ 0.7   (both about the cat)
          sim(s1,s2) ≈ 0.1   ← big drop = topic shift = CUT HERE
          sim(s2,s3) ≈ 0.6   (both finance)
    4. chunks:  "The cat sat on the mat. It purred softly."   and
                "Revenue rose 12%. Costs fell."

The cat sentences and the finance sentences land in SEPARATE chunks. This targets
our pooled-track lever directly: TRACe relevance = relevant-sentences / total-
retrieved-sentences, so isolating relevant material into its own coherent chunk
should raise the relevant-FRACTION (the same mechanism that made PGC win
CustomerSupport in pooled Exp 7).

Two ways to turn the neighbour scores into cut points (config knob `breakpoint`):
  * "percentile" (default): cut at the biggest similarity DROPS within each
    document — gaps whose DISTANCE (1 - cosine) exceeds the `percentile`-th
    percentile of that document's gaps. Self-calibrating per document, so it is
    robust across domains and embedding models (this is LangChain's default).
  * "absolute": cut wherever neighbour cosine similarity falls below a fixed
    `threshold`. Simpler and predictable, but the right number drifts by model
    and domain. Kept as a built-in ablation arm to compare against percentile.

⚠️ HEAVY at run time: .chunk() loads an embedding model (torch). The MODULE
import is light (numpy only — torch is imported lazily on first .chunk()), but by
convention this chunker is registered via chunking.load_chunkers(), NOT on
`import src`, so the lightweight-import rule holds (AI_CONTEXT §18 brick-2/3).

Registered as chunker type "semantic":
    {type: semantic, breakpoint: percentile, percentile: 95}
    {type: semantic, breakpoint: absolute,   threshold: 0.5}
"""

import numpy as np

from ..registry import register
from .base import Chunk, resolve_doc_ids

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@register("chunker", "semantic")
class SemanticChunker:
    def __init__(self, model: str = DEFAULT_MODEL, breakpoint: str = "percentile",
                 percentile: float = 95.0, threshold: float = 0.5):
        if breakpoint not in ("percentile", "absolute"):
            raise ValueError("breakpoint must be 'percentile' or 'absolute'")
        if not (0.0 < percentile < 100.0):
            raise ValueError("percentile must be in (0, 100)")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold (a cosine similarity) must be in [0, 1]")
        self.model_name = model
        self.breakpoint = breakpoint
        self.percentile = percentile
        self.threshold = threshold
        self._embedder = None      # lazy — built on first .chunk() (pulls in torch)
        self._splitter = None      # lazy — regex sentence splitter

    def _ensure_loaded(self) -> None:
        """Build the embedder + splitter on first use (defers the torch import)."""
        if self._embedder is None:
            from ..embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
            # normalize=True (the default) → dot product == cosine similarity.
            self._embedder = SentenceTransformerEmbedder(model=self.model_name)
        if self._splitter is None:
            from ..segmentation.regex_splitter import RegexSplitter
            self._splitter = RegexSplitter()

    def _cut_after(self, sims: list[float]) -> set[int]:
        """Given neighbour cosine sims (length = #sentences - 1, sims[i] = sim of
        sentence i and i+1), return the set of indices i to CUT AFTER (start a new
        chunk at sentence i+1). Pure function — testable with no model."""
        if not sims:
            return set()
        if self.breakpoint == "percentile":
            distances = [1.0 - s for s in sims]                 # bigger = sharper topic shift
            thr = float(np.percentile(distances, self.percentile))
            # strict ">" so a uniformly-coherent document gets NO cut (stays one chunk)
            return {i for i, d in enumerate(distances) if d > thr}
        # absolute: cut where similarity dips below the fixed threshold
        return {i for i, s in enumerate(sims) if s < self.threshold}

    @staticmethod
    def _group(sentences: list[str], cut_after: set[int]) -> list[str]:
        """Join sentences into chunk texts, closing a chunk after each cut index."""
        pieces, current = [], []
        for i, sent in enumerate(sentences):
            current.append(sent)
            if i in cut_after:                  # boundary falls AFTER sentence i
                pieces.append(" ".join(current))
                current = []
        if current:                             # the tail (last sentence is never a cut index)
            pieces.append(" ".join(current))
        return pieces

    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        self._ensure_loaded()
        ids = resolve_doc_ids(documents, doc_ids)

        # Split every document into sentences and collect them into ONE flat list so
        # the whole batch is embedded in a single call (far faster than per-document
        # calls on a large pooled corpus). `spans` remembers each doc's slice.
        spans: list[tuple[str, list[str], int]] = []   # (doc_id, sentences, start_index)
        all_sentences: list[str] = []
        for doc_id, text in zip(ids, documents):
            if not text or not text.strip():
                continue                                # skip empty documents
            sentences = self._splitter.split(text)
            if not sentences:
                continue
            spans.append((doc_id, sentences, len(all_sentences)))
            all_sentences.extend(sentences)

        if not all_sentences:
            return []

        vectors = self._embedder.embed(all_sentences)          # (N, dim), normalized

        chunks: list[Chunk] = []
        for doc_id, sentences, start in spans:
            n = len(sentences)
            if n == 1:                                          # nothing to compare → one chunk
                chunks.append(Chunk(text=sentences[0], doc_id=doc_id, chunk_id=f"{doc_id}-0"))
                continue
            vecs = vectors[start:start + n]
            sims = [float(np.dot(vecs[i], vecs[i + 1])) for i in range(n - 1)]
            pieces = self._group(sentences, self._cut_after(sims))
            for j, piece in enumerate(pieces):
                chunks.append(Chunk(text=piece, doc_id=doc_id, chunk_id=f"{doc_id}-{j}"))
        return chunks
