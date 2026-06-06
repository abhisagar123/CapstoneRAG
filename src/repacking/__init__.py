"""Repacking package — chunk-ordering strategies (no model).

Layout:
  base.py             Repacker interface
  order_repackers.py  ForwardRepacker / ReverseRepacker / SidesRepacker

Safe to import eagerly (pure Python, no heavy deps), so src/__init__.py imports
it to register the strategies.
"""

from .base import Repacker  # noqa: F401 — shared contract

from . import order_repackers  # noqa: F401 — registers forward / reverse / sides

from .order_repackers import (  # noqa: F401 — convenience re-exports
    ForwardRepacker, ReverseRepacker, SidesRepacker,
)
