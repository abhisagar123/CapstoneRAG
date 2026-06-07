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

from . import chunking   # noqa: F401  — registers FixedChunker, NoOpChunker
from . import indexing    # noqa: F401  — registers FaissIndex (faiss import is lazy)
from . import retrieval   # noqa: F401  — registers DenseRetriever
from . import repacking   # noqa: F401  — registers forward / reverse / sides (pure Python)
from . import reranking    # noqa: F401  — registers NoOpReranker; CrossEncoder gated behind load_rerankers()
from . import prompting    # noqa: F401  — registers grounded / minimal prompt builders (pure Python)
from . import generation   # noqa: F401  — registers EchoGenerator; HuggingFaceGenerator gated behind load_generators()
from . import segmentation  # noqa: F401  — registers RegexSplitter; NltkSplitter gated behind load_nltk_splitter()
from . import judge          # noqa: F401  — registers FakeJudge; HuggingFaceJudge gated behind load_judges()
