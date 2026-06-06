"""CapstoneRAG source package.

Importing this package registers the light, dependency-free components into the
registry (so `build("chunker", "fixed")` works after just `import src`). The
`@register` decorators only run when their module is imported — hence these.

IMPORTANT: only import components whose IMPORT is light here. Embeddings is the
exception: its torch import happens at *construction*, gated behind
`load_embedders()`, so it is NOT imported here (importing it is fine, but we keep
the rule simple). indexing/ imports faiss lazily inside FaissIndex.__init__, so
importing the package is light and safe here.
"""

from . import chunking  # noqa: F401  — registers FixedChunker, NoOpChunker
from . import indexing   # noqa: F401  — registers FaissIndex (faiss import is lazy)
from . import retrieval  # noqa: F401  — registers DenseRetriever
