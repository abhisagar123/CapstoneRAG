"""Query classifier package — one file per strategy (no model).

Layout:
  base.py                     QueryClassifier interface + ClassificationResult
  always_retrieve.py          AlwaysRetrieveClassifier (type "always_retrieve")
  heuristic.py                HeuristicClassifier (type "heuristic")

To ADD a strategy: new file with @register("query_classifier", "<name>") + an
import below.

Safe to import eagerly (pure Python, no heavy deps), so src/__init__.py imports
this package to register the strategies.
"""

from .base import ClassificationResult, QueryClassifier  # noqa: F401 — shared contract

from . import always_retrieve  # noqa: F401 — registers "always_retrieve"
from . import heuristic        # noqa: F401 — registers "heuristic"

from .always_retrieve import AlwaysRetrieveClassifier  # noqa: F401
from .heuristic import HeuristicClassifier             # noqa: F401
