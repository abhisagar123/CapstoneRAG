"""ForwardRepacker — keep best→worst order (the naive baseline / identity).

Input is assumed ranked best→worst (rank 0 first); forward leaves it unchanged.
This is the baseline that reverse/sides are compared against.

Registered as repacker type "forward".
"""

from ..registry import register
from ..indexing import RetrievedChunk


@register("repacker", "forward")
class ForwardRepacker:
    def pack(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(chunks)
