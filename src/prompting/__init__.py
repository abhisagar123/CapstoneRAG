"""Prompting package — one file per prompt strategy (no model, no heavy deps).

Layout:
  base.py                       PromptBuilder interface + format_chunks() helper
  grounded_prompt_builder.py    GroundedPromptBuilder (type "grounded") — baseline
  minimal_prompt_builder.py     MinimalPromptBuilder  (type "minimal")  — contrast arm

To ADD a variant: new file with @register("prompt", "<name>") + one import below.

Safe to import eagerly (pure Python string assembly), so src/__init__.py imports
it to register the strategies.
"""

from .base import PromptBuilder, format_chunks  # noqa: F401 — shared contract + helper

from . import grounded_prompt_builder  # noqa: F401 — registers "grounded"
from . import minimal_prompt_builder   # noqa: F401 — registers "minimal"

from .grounded_prompt_builder import GroundedPromptBuilder  # noqa: F401
from .minimal_prompt_builder import MinimalPromptBuilder    # noqa: F401
