"""Chunking — shared contract (data structure + interface).

Every chunking strategy lives in its own file in this package and imports from
here. This file holds ONLY the shared pieces, so all strategies honour one
contract:

  Chunk    an immutable record: text + provenance (doc_id, chunk_id) + meta
  Chunker  the INTERFACE (the 'socket shape'): .chunk(documents) -> list[Chunk]

A "chunk" is a retrievable piece of a document; each later becomes a searchable
vector. See LLD §2/§3.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Chunk:
    text: str               # the chunk's text
    doc_id: str             # which source document it came from (e.g. "0")
    chunk_id: str           # unique id within the index (e.g. "0-2" = doc 0, chunk 2)
    meta: dict = field(default_factory=dict)   # room for later extras (char span, parent id, ...)


@runtime_checkable
class Chunker(Protocol):
    """The contract every chunking strategy honours."""
    def chunk(self, documents: list[str], doc_ids: list[str] | None = None) -> list[Chunk]:
        ...


def resolve_doc_ids(documents: list[str], doc_ids: list[str] | None) -> list[str]:
    """Shared helper: default each document's id to its index ('0','1',...)
    unless explicit ids are given. Validates length when given."""
    if doc_ids is None:
        return [str(i) for i in range(len(documents))]
    if len(doc_ids) != len(documents):
        raise ValueError(f"doc_ids length {len(doc_ids)} != documents length {len(documents)}")
    return doc_ids
