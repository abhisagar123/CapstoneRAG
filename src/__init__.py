"""CapstoneRAG source package.

Importing this package registers the light, dependency-free components into the
registry (so `build("chunker", "fixed")` works after just `import src`). The
`@register` decorators only run when their module is imported — hence these.

IMPORTANT: only import components with NO heavy/optional deps here (no torch,
sentence-transformers, faiss, ...). Heavy components (embeddings, generator)
self-register when their own module is first imported, so importing `src` never
drags in the GPU stack.
"""

from . import chunking  # noqa: F401  — registers FixedChunker, NoOpChunker
