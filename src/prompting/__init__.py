"""Prompting package — one file per prompt strategy (no model, no heavy deps).

Layout:
  base.py                              PromptBuilder interface + format_chunks() helper
  grounded_prompt_builder.py           GroundedPromptBuilder (type "grounded") — baseline
  minimal_prompt_builder.py            MinimalPromptBuilder  (type "minimal")  — contrast arm
  grounded_complete_prompt_builder.py  GroundedCompletePromptBuilder ("grounded_complete")
                                         — grounding + completeness push (Exp 4)
  extractive_prompt_builder.py         ExtractivePromptBuilder ("extractive")
                                         — grounding + stick-to-source-wording (Exp 6D)
  grounded_coverage_prompt_builder.py  GroundedCoveragePromptBuilder ("grounded_coverage")
                                         — grounding + PROCEDURAL coverage push (Exp G)
  grounded_fewshot_prompt_builder.py   GroundedFewshotPromptBuilder ("grounded_fewshot")
                                         — grounding + few-shot DEMONSTRATIONS (Exp H)

To ADD a variant: new file with @register("prompt", "<name>") + one import below.

Safe to import eagerly (pure Python string assembly), so src/__init__.py imports
it to register the strategies.
"""

from .base import PromptBuilder, format_chunks  # noqa: F401 — shared contract + helper

from . import grounded_prompt_builder           # noqa: F401 — registers "grounded"
from . import minimal_prompt_builder            # noqa: F401 — registers "minimal"
from . import grounded_complete_prompt_builder  # noqa: F401 — registers "grounded_complete"
from . import extractive_prompt_builder         # noqa: F401 — registers "extractive"
from . import grounded_coverage_prompt_builder  # noqa: F401 — registers "grounded_coverage"
from . import grounded_fewshot_prompt_builder   # noqa: F401 — registers "grounded_fewshot"

from .grounded_prompt_builder import GroundedPromptBuilder                  # noqa: F401
from .minimal_prompt_builder import MinimalPromptBuilder                    # noqa: F401
from .grounded_complete_prompt_builder import GroundedCompletePromptBuilder  # noqa: F401
from .extractive_prompt_builder import ExtractivePromptBuilder              # noqa: F401
from .grounded_coverage_prompt_builder import GroundedCoveragePromptBuilder  # noqa: F401
from .grounded_fewshot_prompt_builder import GroundedFewshotPromptBuilder    # noqa: F401
