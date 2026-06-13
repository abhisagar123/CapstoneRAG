"""Pipeline — assemble components from a config and run the RAG chain.

This is the conductor: it reads a validated PipelineConfig, instantiates each
swappable component via the registry (no hard-coded class names), and runs them
in order. Two phases, mirroring real RAG:

  index_documents(docs)  OFFLINE (once): chunk -> embed -> index
  answer(query)          ONLINE (per question):
                         retrieve -> rerank -> repack -> prompt -> generate

It returns {answer, sources, context} and STOPS there — segmenting + judging +
TRACe scoring is the ExperimentRunner's job (clean separation; the judge is
deferred). See AI_CONTEXT §11 / LLD §4.

Assembly notes (why this isn't just "build each stage from config"):
  - Most stages are built straight from config: build(kind, type, params).
  - The RETRIEVER is special — it wraps the already-built embedder + index, so
    we construct it directly rather than via a config `type` alone.
  - k (retriever) and top_n (reranker) are RUNTIME args to retrieve()/rerank(),
    not constructor params — we read them from the config's params here.
  - Optional stages (reranker/repacker) may be None in the config → skipped.
"""

from .registry import build
from .retrieval import DenseRetriever, BM25Retriever, HybridRetriever


# How many candidates to retrieve / keep, if the config doesn't say.
DEFAULT_K = 20
DEFAULT_TOP_N = 5


class Pipeline:
    def __init__(self, config):
        self.config = config

        c = config

        # Runtime knobs (k, top_n) live in config params but are CALL-TIME args to
        # retrieve()/rerank(), NOT constructor args — pull them out first so they
        # aren't passed to component __init__ (which would error).
        self.k = c.retriever.params.get("k", DEFAULT_K)
        self.top_n = (c.reranker.params.get("top_n", DEFAULT_TOP_N)
                      if c.reranker else DEFAULT_TOP_N)
        reranker_ctor_params = ({k: v for k, v in c.reranker.params.items() if k != "top_n"}
                                if c.reranker else {})

        # --- build the swappable components from config (registry lookups) ---
        self.chunker = build("chunker", c.chunker.type, c.chunker.params)
        self.embedder = build("embedder", c.embedder.type, c.embedder.params)
        # Index needs the embedding dim; pass it alongside the config's params.
        index_params = {**c.index.params, "dim": self.embedder.dim}
        self.index = build("index", c.index.type, index_params)
        self.prompt_builder = build("prompt", c.prompt.type, c.prompt.params)
        self.generator = build("generator", c.generator.type, c.generator.params)

        # Optional stages → None means "skip this stage".
        self.reranker = build("reranker", c.reranker.type, reranker_ctor_params) if c.reranker else None
        self.repacker = build("repacker", c.repacker.type, c.repacker.params) if c.repacker else None

        # Retriever is config-driven. It wraps the built embedder + index (it's NOT a
        # bare registry build() like other stages, because it needs those objects), so a
        # small factory maps retriever.type -> the right wrapper. `k` is a call-time knob
        # (handled above) and is excluded from the ctor params.
        self._retriever_ctor_params = {k: v for k, v in c.retriever.params.items() if k != "k"}
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        """Construct the retriever for this config's `retriever.type`, wrapping the
        already-built embedder + index. dense/hybrid need the vector index; bm25 is
        sparse (its corpus is supplied later via index_corpus). Defaults to dense."""
        rtype = self.config.retriever.type
        p = self._retriever_ctor_params
        if rtype == "bm25":
            return BM25Retriever()
        if rtype == "hybrid":
            return HybridRetriever(embedder=self.embedder, index=self.index, **p)
        return DenseRetriever(embedder=self.embedder, index=self.index)   # "dense" default

    # ---------------- OFFLINE: build the index ----------------

    def reset_index(self) -> None:
        """Replace the index with a fresh empty one (same type/params).

        Needed for `corpus_mode: per_example`, where each question is answered
        against only its OWN documents — so the index is cleared between examples.
        """
        index_params = {**self.config.index.params, "dim": self.embedder.dim}
        self.index = build("index", self.config.index.type, index_params)
        self.retriever = self._build_retriever()

    def index_documents(self, documents: list[str], doc_ids: list[str] | None = None) -> int:
        """Chunk the documents, embed the chunks, add them to the index.
        Returns the number of chunks indexed."""
        chunks = self.chunker.chunk(documents, doc_ids=doc_ids)
        if not chunks:
            return 0
        vectors = self.embedder.embed([ch.text for ch in chunks])
        self.index.add(chunks, vectors)
        # Sparse / hybrid retrievers build their OWN term index over the chunk texts;
        # give them the chunks now (dense provides a no-op / lacks the method).
        index_corpus = getattr(self.retriever, "index_corpus", None)
        if callable(index_corpus):
            index_corpus(chunks)
        return len(chunks)

    # ---------------- ONLINE: answer a query ----------------

    def answer(self, query: str) -> dict:
        """Run the online chain and return {answer, sources, context}.

        sources  = the chunks actually fed to the generator (post rerank/repack)
        context  = the exact context text the prompt builder used
        """
        # 1. retrieve a wide net of candidates
        candidates = self.retriever.retrieve(query, k=self.k)

        # 2. rerank down to top_n (or skip → just truncate to top_n)
        if self.reranker is not None:
            chunks = self.reranker.rerank(query, candidates, top_n=self.top_n)
        else:
            chunks = candidates[: self.top_n]

        # 3. reorder for the prompt (or skip → keep order)
        if self.repacker is not None:
            chunks = self.repacker.pack(chunks)

        # 4. build the prompt, 5. generate
        prompt = self.prompt_builder.build(query, chunks)
        answer = self.generator.generate(prompt)

        return {
            "answer": answer,
            "sources": chunks,
            "context": "\n".join(rc.chunk.text for rc in chunks),
        }


def build_pipeline(config) -> Pipeline:
    """Factory: assemble a Pipeline from a (validated) PipelineConfig."""
    return Pipeline(config)
