"""ExtractiveEmbeddingSummarizer — keep the sentences whose MEANING matches the query.

The embedding flavour of context compression — our analog of the paper's Recomp
extractive compressor (Best Practices §3.7). For each retrieved chunk:

    1. split it into sentences (reuse the regex splitter — no new dependency)
    2. embed every sentence AND the query with our normalized MiniLM embedder
    3. score each sentence by cosine(sentence, query)   (dot product, vectors are L2-normed)
    4. keep the top `ratio` fraction (>= min_keep), drop the rest, in reading order

Example (query "What caused the revenue rise?"):

    chunk: "The office moved in May. Revenue rose 12% on strong ad sales. Staff
            enjoyed the new cafe. Ad demand outpaced supply."
    cos to query:  0.07            0.81                       0.05         0.66
    ratio=0.5 -> keep the 2 best  ->  "Revenue rose 12% on strong ad sales.
                                       Ad demand outpaced supply."

The off-topic office/cafe sentences are dropped before the generator sees them.
Because it SELECTS existing sentences verbatim (never rewrites), it cannot
introduce ungrounded text — the property that makes it adherence-safe.

Mechanism vs the other arms (clean one-variable story):
  * vs `lexical`: identical select-top-sentences logic; only SCORING differs
    (cosine meaning-match vs word overlap) — isolates "does semantic scoring beat
    lexical?".
  * vs `top_n` (Exp 2, which hurt adherence): top_n drops whole CHUNKS; this drops
    irrelevant SENTENCES while keeping >=1 per chunk — no chunk's facts are lost.

⚠️ HEAVY at run time: .compress() loads an embedding model (torch). The MODULE
import is light (numpy only); by convention this is registered via
summarization.load_summarizers(), NOT on `import src`, so the lightweight-import
rule holds (AI_CONTEXT §18 brick-2/3) — same treatment as the semantic chunker.

Registered as summarizer type "extractive_embedding":
    {type: extractive_embedding, ratio: 0.5}
"""

import dataclasses

import numpy as np

from ..registry import register
from ..indexing import RetrievedChunk
from .base import select_top_indices, keep_sentences

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@register("summarizer", "extractive_embedding")
class ExtractiveEmbeddingSummarizer:
    def __init__(self, model: str = DEFAULT_MODEL, ratio: float = 0.5, min_keep: int = 1):
        """ratio: fraction of each chunk's sentences to KEEP. min_keep: floor per
        chunk (>=1 → never empty a chunk). model: the sentence embedder (MiniLM,
        matching our retrieval embedder so scores live in the same space)."""
        if not (0.0 < ratio <= 1.0):
            raise ValueError("ratio must be in (0, 1]")
        if min_keep < 1:
            raise ValueError("min_keep must be >= 1")
        self.model_name = model
        self.ratio = ratio
        self.min_keep = min_keep
        self._embedder = None        # lazy — built on first .compress() (pulls in torch)
        self._splitter = None        # lazy — regex sentence splitter

    def _ensure_loaded(self) -> None:
        """Build the embedder + splitter on first use (defers the torch import)."""
        if self._embedder is None:
            from ..embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
            self._embedder = SentenceTransformerEmbedder(model=self.model_name)   # normalize=True
        if self._splitter is None:
            from ..segmentation.regex_splitter import RegexSplitter
            self._splitter = RegexSplitter()

    def compress(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        self._ensure_loaded()

        # Split each chunk into sentences once. Chunks too short to trim are passed
        # through untouched; the rest contribute their sentences to ONE flat list so
        # the whole set (+ the query) is embedded in a single batched call (fast).
        per_chunk_sentences: list[list[str] | None] = []   # None = pass through unchanged
        all_sentences: list[str] = []
        spans: list[tuple[int, int]] = []                  # (start, count) into all_sentences, per trimmable chunk
        for rc in chunks:
            sents = self._splitter.split(rc.chunk.text)
            if len(sents) <= self.min_keep:
                per_chunk_sentences.append(None)
                spans.append((0, 0))
            else:
                per_chunk_sentences.append(sents)
                spans.append((len(all_sentences), len(sents)))
                all_sentences.extend(sents)

        if not all_sentences:                              # nothing trimmable → unchanged
            return list(chunks)

        # Embed query + all sentences together; cosine = dot product (all normalized).
        vectors = self._embedder.embed([query] + all_sentences)
        q_vec = vectors[0]
        sent_vecs = vectors[1:]

        out = []
        for rc, sents, (start, count) in zip(chunks, per_chunk_sentences, spans):
            if sents is None:                              # short chunk → keep as-is
                out.append(rc)
                continue
            scores = [float(np.dot(q_vec, sent_vecs[start + j])) for j in range(count)]
            keep = select_top_indices(scores, self.ratio, self.min_keep)
            new_text = keep_sentences(rc.chunk.text, sents, keep)
            new_chunk = dataclasses.replace(rc.chunk, text=new_text)
            out.append(dataclasses.replace(rc, chunk=new_chunk))
        return out
