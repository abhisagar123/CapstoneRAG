"""Repacking package — one file per ordering strategy (no model).

Layout:
  base.py               Repacker interface
  forward_repacker.py   ForwardRepacker  (type "forward", identity baseline)
  reverse_repacker.py   ReverseRepacker  (type "reverse", best-last)
  sides_repacker.py     SidesRepacker    (type "sides", strong at both ends)

To ADD a strategy: new file with @register("repacker", "<name>") + one import below.

Safe to import eagerly (pure Python, no heavy deps), so src/__init__.py imports
it to register the strategies.
"""

from .base import Repacker  # noqa: F401 — shared contract

# Import each strategy so it self-registers.
from . import forward_repacker  # noqa: F401 — registers "forward"
from . import reverse_repacker  # noqa: F401 — registers "reverse"
from . import sides_repacker    # noqa: F401 — registers "sides"

# Convenience re-exports.
from .forward_repacker import ForwardRepacker  # noqa: F401
from .reverse_repacker import ReverseRepacker  # noqa: F401
from .sides_repacker import SidesRepacker      # noqa: F401
