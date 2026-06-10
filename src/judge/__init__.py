"""Judge package — produces TRACe labels (the evaluator's hard half).

Layout:
  base.py        Judge interface + EXACT Appendix-7.4 prompt + JSON parsing +
                 scores_from_label() adapter (judge JSON -> trace.py math)
  fake_judge.py  FakeJudge (type "fake") — deterministic, no model (tests/wiring)
  hf_judge.py    HuggingFaceJudge (type "hf") — real OSS model, heavy, COLAB

HEAVY-DEP SPLIT (same as embeddings/generation):
- FakeJudge registers on `import src` (light).
- HuggingFaceJudge pulls in transformers/torch → NOT imported here; call
  load_judges() before building it (on Colab).
"""

from .base import (  # noqa: F401 — shared contract + helpers
    Judge, build_prompt, parse_label_json, scores_from_label,
    APPENDIX_7_4_PROMPT, CONSERVATIVE_ADDENDUM,
)

from . import fake_judge  # noqa: F401 — registers FakeJudge ("fake"), no model
from .fake_judge import FakeJudge  # noqa: F401


def load_judges() -> None:
    """Register the heavy judge(s) so they can be built. Call before building "hf"."""
    from . import hf_judge  # noqa: F401 — registers HuggingFaceJudge ("hf")
