"""RegexSplitter — baseline sentence splitter (zero dependencies).

Splits on . ! ? when followed by whitespace and a capital letter / digit. Simple,
transparent, no deps — good enough for a baseline. It WILL mis-split some
abbreviations ("Dr. Smith", "Section 8.2"); that's exactly why NLTK exists as an
alternative strategy to compare against. Registered as splitter type "regex".
"""

import re

from ..registry import register

# Split point: a sentence-ender (.!?), optional closing quote/paren, whitespace,
# then a capital letter or digit (a likely new-sentence start).
_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=[A-Z0-9])')


@register("splitter", "regex")
class RegexSplitter:
    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        parts = _BOUNDARY.split(text.strip())
        return [p.strip() for p in parts if p.strip()]   # drop empties
