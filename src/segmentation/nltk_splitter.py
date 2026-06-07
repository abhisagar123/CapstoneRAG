"""NltkSplitter — sentence splitter using NLTK's trained punkt tokenizer.

punkt is an unsupervised statistical sentence tokenizer (a data file, NOT an LLM
— fine to run locally). It handles abbreviations far better than the regex
baseline ("Dr. Smith", "Section 8.2", "$1.5M" stay intact).

Needs the one-time 'punkt'/'punkt_tab' data download, so this module is NOT
imported by src/__init__ — call load_nltk_splitter() once to register it (which
also ensures the data is present). Registered as splitter type "nltk".
"""

from ..registry import register


@register("splitter", "nltk")
class NltkSplitter:
    def __init__(self):
        import nltk
        # Ensure punkt data is available (download once; quiet if already there).
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        self._tokenize = nltk.sent_tokenize

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [s.strip() for s in self._tokenize(text.strip()) if s.strip()]
