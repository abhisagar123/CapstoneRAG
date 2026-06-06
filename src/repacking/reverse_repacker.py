"""ReverseRepacker — best→worst reversed, so the MOST relevant chunk is LAST.

Putting the strongest chunk at the END of the prompt places it nearest the
question, where the LLM attends most — mitigating "lost in the middle". The
Best-Practices-for-RAG paper found this ordering helps.

Registered as repacker type "reverse".
"""

from ..registry import register
from ..indexing import RetrievedChunk


@register("repacker", "reverse")
class ReverseRepacker:
    def pack(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(reversed(chunks))
