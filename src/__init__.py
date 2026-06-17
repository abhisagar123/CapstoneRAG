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

# ── OpenMP dual-runtime guard (macOS) — MUST be set before faiss / torch load ──────
# faiss-cpu and PyTorch each bundle their OWN OpenMP runtime (libomp.dylib). When both
# get loaded into one process — which the pooled pipeline does (faiss index + torch
# embedder live together) — the duplicate-OpenMP guard aborts the process:
#   "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized"
# observed as a hard SEGFAULT during a faiss search after the torch embedder had loaded
# (first hit on the large pooled PGC index, 14,968 chunks, 14 Jun). KMP_DUPLICATE_LIB_OK
# tolerates the duplicate; OMP_NUM_THREADS=1 keeps the two runtimes from fighting over
# threads (negligible cost — exact faiss search + a tiny MiniLM embedder). setdefault so
# an explicit user/env override still wins. This is the documented workaround for a
# library-linkage conflict, NOT a logic bug in our code.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

from . import chunking   # noqa: F401  — registers FixedChunker, NoOpChunker
from . import indexing    # noqa: F401  — registers FaissIndex (faiss import is lazy)
from . import retrieval   # noqa: F401  — registers DenseRetriever
from . import repacking   # noqa: F401  — registers forward / reverse / sides (pure Python)
from . import summarization  # noqa: F401  — registers none / lexical; extractive_embedding gated behind load_summarizers()
from . import reranking    # noqa: F401  — registers NoOpReranker; CrossEncoder gated behind load_rerankers()
from . import prompting    # noqa: F401  — registers grounded / minimal prompt builders (pure Python)
from . import generation   # noqa: F401  — registers EchoGenerator; HuggingFaceGenerator gated behind load_generators()
from . import segmentation  # noqa: F401  — registers RegexSplitter; NltkSplitter gated behind load_nltk_splitter()
from . import judge          # noqa: F401  — registers FakeJudge; HuggingFaceJudge gated behind load_judges()
