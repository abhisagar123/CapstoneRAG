"""Segmentation — the bridge from pipeline output to the TRACe evaluator.

The TRACe judge expects context + answer split into KEYED sentences, exactly like
RAGBench's reference fields (confirmed in AI_CONTEXT §8):
    documents_sentences : [[['0a', text], ['0b', text]], [['1a', text], ...]]  # [doc][sent][key,text]
    response_sentences  : [['a', text], ['b', text], ...]                       # answer sentences
Our pipeline produces RAW text, so this module creates those keys for our own
output — making it interchangeable with reference data for judging.

TWO layers:
  SentenceSplitter  — SWAPPABLE strategy: raw text -> list[str] of sentences
                      (regex / nltk / ...). This is what we A/B in experiments.
  OutputSegmenter   — FIXED keyer: uses a splitter, then assigns RAGBench keys
                      ('{doc}{letter}' for docs, plain letters for the answer).
                      The key SCHEME is defined by RAGBench, so it does not vary.
"""

import string
from typing import Protocol, runtime_checkable


@runtime_checkable
class SentenceSplitter(Protocol):
    """The contract every sentence-splitting strategy honours."""

    def split(self, text: str) -> list[str]:
        """Split a blob of text into a list of sentence strings (no empties)."""
        ...


def _letters(n: int) -> list[str]:
    """Sentence-letter keys: a, b, ..., z, aa, ab, ... (handles >26 sentences)."""
    out, i = [], 0
    while len(out) < n:
        i += 1
        # base-26 style: 1->a, 26->z, 27->aa, ...
        s, x = "", i
        while x > 0:
            x, r = divmod(x - 1, 26)
            s = string.ascii_lowercase[r] + s
        out.append(s)
    return out


class OutputSegmenter:
    """Fixed keyer: split context chunks + the answer into RAGBench-keyed sentences.

    Takes a SentenceSplitter (the swappable part) at construction.
    """

    def __init__(self, splitter: SentenceSplitter):
        self.splitter = splitter

    def segment_documents(self, doc_texts: list[str]) -> list[list[list[str]]]:
        """Each document -> list of [key, sentence]; key = '{doc_index}{letter}'.
        e.g. doc 0 -> [['0a', ...], ['0b', ...]], doc 1 -> [['1a', ...], ...]."""
        out = []
        for doc_idx, text in enumerate(doc_texts):
            sentences = self.splitter.split(text)
            keys = _letters(len(sentences))
            out.append([[f"{doc_idx}{ltr}", s] for ltr, s in zip(keys, sentences)])
        return out

    def segment_response(self, answer: str) -> list[list[str]]:
        """Answer -> list of [key, sentence] with plain-letter keys: a, b, c, ..."""
        sentences = self.splitter.split(answer)
        keys = _letters(len(sentences))
        return [[ltr, s] for ltr, s in zip(keys, sentences)]

    def segment(self, doc_texts: list[str], answer: str) -> dict:
        """Full bridge output: the two keyed fields, matching RAGBench's schema."""
        return {
            "documents_sentences": self.segment_documents(doc_texts),
            "response_sentences": self.segment_response(answer),
        }
