"""SidesRepacker — strongest chunks at the two ENDS, weakest in the middle.

The middle of the context is the low-attention trough ("lost in the middle"),
so we place the best chunks at both ends and bury the weak ones in the middle.
e.g. ranks [0,1,2,3,4] → [0,2,4,3,1]: even ranks fill front-to-back, odd ranks
fill the tail in reverse.

Registered as repacker type "sides".
"""

from ..registry import register
from ..indexing import RetrievedChunk


@register("repacker", "sides")
class SidesRepacker:
    def pack(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        left, right = [], []
        for i, c in enumerate(chunks):          # chunks are best→worst
            (left if i % 2 == 0 else right).append(c)
        return left + right[::-1]               # strong at front, strong at back
