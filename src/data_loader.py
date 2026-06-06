"""Data loader — the single front door to the RAGBench dataset.

Every downstream component (chunker, retriever, evaluator) consumes RAGBench
examples. Rather than scatter `load_dataset("rungalileo/ragbench", ...)` and the
domain→config knowledge across many files, all of it lives here.

Key idea: we think in DOMAINS (5 of them), but the dataset is published as
CONFIGS (12 sub-datasets). One domain may bundle several configs, e.g.
Biomedical = pubmedqa + covidqa. This module is the one place that knows the map
(reused from evaluator.validate, the single source of truth) and hides it behind
`load_domain("Biomedical", split="test", n=50)`.

CPU-only, no API key. `datasets` is imported lazily so importing this module is
cheap and never fails if the heavy library is absent.
"""

import random

# Single source of truth for the domain→config map (defined in the evaluator).
from .evaluator.validate import DOMAIN_CONFIGS, REPO

DOMAINS = list(DOMAIN_CONFIGS)                       # ["Biomedical", "GenKnowledge", ...]
VALID_SPLITS = ("train", "validation", "test")       # split discipline: nothing else allowed
CONFIG_TO_DOMAIN = {c: d for d, cs in DOMAIN_CONFIGS.items() for c in cs}


def configs_for(domain: str) -> list[str]:
    """'Legal' -> ['cuad'];  'Biomedical' -> ['pubmedqa', 'covidqa']."""
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain {domain!r}. Valid: {DOMAINS}")
    return list(DOMAIN_CONFIGS[domain])


def _check_split(split: str) -> None:
    """Guard rail: tune on validation, report on test, never tune on test."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}. Valid: {list(VALID_SPLITS)}")


def load_config(config: str, split: str = "test", n: int | None = None,
                seed: int = 42) -> list[dict]:
    """Load one specific sub-dataset config (e.g. just 'cuad').

    Each returned example is the raw RAGBench dict plus a `_config` tag so we
    always know which sub-dataset it came from (used for per-source analysis even
    when a domain is sampled as a whole). If `n` is given, take a seeded random
    sample of up to n examples.
    """
    from datasets import load_dataset

    _check_split(split)
    ds = load_dataset(REPO, config, split=split)

    idx = list(range(len(ds)))
    if n is not None and n < len(idx):
        idx = random.Random(seed).sample(idx, n)        # seeded → reproducible

    out = []
    for i in idx:
        ex = dict(ds[i])            # copy so we can safely tag it
        ex["_config"] = config
        out.append(ex)
    return out


def load_domain(domain: str, split: str = "test", n: int | None = None,
                seed: int = 42) -> list[dict]:
    """Load all examples for a domain+split, pooling its sub-datasets.

    `n` is the total per DOMAIN (not per config): we pool every sub-dataset's
    examples, then take a single seeded sample of up to n from the pool. Each
    example keeps its `_config` tag, so per-sub-dataset breakdowns are still
    possible in analysis even though the sample isn't balanced by source.

    `n=None` returns the full split (what final reported numbers use).
    """
    _check_split(split)
    pool: list[dict] = []
    for config in configs_for(domain):
        pool.extend(load_config(config, split=split, n=None))   # full configs, then sample the pool

    if n is not None and n < len(pool):
        pool = random.Random(seed).sample(pool, n)
    return pool
