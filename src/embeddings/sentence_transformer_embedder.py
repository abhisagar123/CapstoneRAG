"""SentenceTransformerEmbedder — wraps any sentence-transformers model.

One flexible class serves ALL sentence-transformers models: the model name is a
constructor parameter, so "MiniLM vs BGE vs E5" is a config value, not a new
class each time:
    {type: sentence_transformer, model: all-MiniLM-L6-v2}
    {type: sentence_transformer, model: BAAI/bge-base-en-v1.5}

Registered under both the generic type "sentence_transformer" and the short
alias "minilm" (the baseline default), so configs can say either.

Heavy deps (torch, sentence-transformers) are imported LAZILY inside __init__,
so importing this module is cheap and it is NOT imported by src/__init__.py
(keeps `import src` free of the GPU/ML stack — see AI_CONTEXT §18 brick-2 rule).
"""

from ..registry import register

# A small, fast, CPU-friendly default (≈90 MB, 384-dim). Strong models like
# BAAI/bge-base-en-v1.5 are selected by passing model=... in config.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@register("embedder", "sentence_transformer")
@register("embedder", "minilm")
class SentenceTransformerEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32,
                 normalize: bool = True):
        # Lazy import: only pay the torch/sentence-transformers cost when an
        # embedder is actually constructed, never just by importing the package.
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.batch_size = batch_size
        self.normalize = normalize          # unit-length vectors → cosine == dot product
        self._model = SentenceTransformer(model)
        # The dimension getter was renamed across sentence-transformers versions
        # (get_sentence_embedding_dimension → get_embedding_dimension); support both.
        get_dim = (getattr(self._model, "get_embedding_dimension", None)
                   or self._model.get_sentence_embedding_dimension)
        self._dim = get_dim()

    def embed(self, texts: list[str]):
        """Encode texts → numpy array of shape (len(texts), dim), float32."""
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    @property
    def dim(self) -> int:
        return self._dim
