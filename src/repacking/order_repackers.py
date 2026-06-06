"""Order repackers — forward / reverse / sides (pure reordering, no model).

All three take chunks ranked best→worst (rank 0 first) and return a reordering.
They are tiny and tightly related (variations on ordering), so they share one
file. Each is the identity-or-permutation of the input — same chunks, new order.

Registered as repacker types "forward", "reverse", "sides".
"""

from ..registry import register
from ..indexing import RetrievedChunk


@register("repacker", "forward")
class ForwardRepacker:
    """Keep best→worst order (rank 0 first). The naive baseline / identity."""

    def pack(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(chunks)


@register("repacker", "reverse")
class ReverseRepacker:
    """Best→worst reversed, so the MOST relevant chunk is LAST — i.e. nearest the
    question at the end of the prompt, where the LLM attends most. (Mitigates
    'lost in the middle'; the Best-Practices paper found this helps.)"""

    def pack(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(reversed(chunks))


@register("repacker", "sides")
class SidesRepacker:
    """Put the strongest chunks at the two ENDS and the weakest in the middle
    (the low-attention trough). e.g. ranks [0,1,2,3,4] → [0,2,4,3,1]:
    even ranks fill front-to-back, odd ranks fill the tail in reverse."""

    def pack(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        left, right = [], []
        for i, c in enumerate(chunks):          # chunks are best→worst
            (left if i % 2 == 0 else right).append(c)
        return left + right[::-1]               # strong at front, strong at back
