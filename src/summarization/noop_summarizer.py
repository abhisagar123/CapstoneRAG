"""NoOpSummarizer — pass the chunks through unchanged (the baseline).

This is the identity stage: no compression at all, so a config with
`summarizer: {type: none}` behaves exactly like the pre-summarization pipeline.
It's the apples-to-apples control every compression arm is compared against
(same role as ForwardRepacker for repacking, or NoOpChunker for chunking).

Registered as summarizer type "none".
"""

from ..registry import register
from ..indexing import RetrievedChunk


@register("summarizer", "none")
class NoOpSummarizer:
    def compress(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(chunks)
