"""Segmentation package — the pipeline→evaluator bridge.

Layout:
  base.py             SentenceSplitter interface + OutputSegmenter (the keyer)
  regex_splitter.py   RegexSplitter (type "regex") — baseline, zero deps
  nltk_splitter.py    NltkSplitter  (type "nltk")  — punkt; needs one-time data download

The regex splitter registers on `import src` (pure Python). The NLTK splitter is
gated behind load_nltk_splitter() because constructing it may trigger a one-time
punkt data download — we don't want that to happen merely from importing src.
"""

from .base import SentenceSplitter, OutputSegmenter  # noqa: F401 — contract + keyer

from . import regex_splitter  # noqa: F401 — registers "regex" (no deps)
from .regex_splitter import RegexSplitter  # noqa: F401


def load_nltk_splitter() -> None:
    """Register the NLTK splitter (and ensure punkt data on first construction)."""
    from . import nltk_splitter  # noqa: F401 — registers "nltk"
