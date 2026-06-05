"""Validate the math-half TRACe evaluator against RAGBench's shipped reference scores.

"Validation" = grade our calculator at scale. For every example we recompute the
four TRACe scores from the gold sentence labels (via
trace.score_from_reference_labels) and compare to the dataset's reference values:

  * the 3 fraction metrics (relevance/utilization/completeness) are floats, graded
    with RMSE ("typical gap"; 0 = identical), the worst single gap, and how many
    rows matched exactly.
  * adherence is a yes/no, graded with plain accuracy (% of rows correct).

If the math half is right, RMSE should be ~0 everywhere. Any domain that is off
tells us something real (as the adherence bug did on 5 Jun).
"""

from .trace import score_from_reference_labels

REPO = "rungalileo/ragbench"

# All 12 configs grouped by the 5 RAGBench domains (confirmed in AI_CONTEXT.md §8).
DOMAIN_CONFIGS = {
    "Biomedical": ["pubmedqa", "covidqa"],
    "GenKnowledge": ["hotpotqa", "msmarco", "hagrid", "expertqa"],
    "Legal": ["cuad"],
    "CustomerSupport": ["techqa", "emanual", "delucionqa"],
    "Finance": ["finqa", "tatqa"],
}

# Map each metric to the dataset field holding its reference value.
REFERENCE_FIELD = {
    "relevance": "relevance_score",
    "utilization": "utilization_score",
    "completeness": "completeness_score",
    "adherence": "adherence_score",
}
FRACTION_METRICS = ["relevance", "utilization", "completeness"]


def rmse(ours, theirs) -> float:
    """Typical gap between two equal-length lists of numbers. 0 means identical.

    (Root Mean Square Error: square each gap so large errors stand out, average
    them, then square-root back to the original units.)
    """
    if not ours:
        return 0.0
    squared = [(a - b) ** 2 for a, b in zip(ours, theirs)]
    return (sum(squared) / len(squared)) ** 0.5


def validate_config(config, n=300):
    """Score n examples of one config from gold labels; compare to reference.

    Returns a per-metric report card dict. Imports `datasets` lazily so importing
    this module never requires the heavy library to be installed.
    """
    from datasets import load_dataset

    ds = load_dataset(REPO, config, split="test")
    n = min(n, len(ds))
    ds = ds.select(range(n))

    ours = {m: [] for m in REFERENCE_FIELD}
    theirs = {m: [] for m in REFERENCE_FIELD}
    for ex in ds:
        scores = score_from_reference_labels(ex)
        for m in REFERENCE_FIELD:
            ours[m].append(scores[m])
            theirs[m].append(ex[REFERENCE_FIELD[m]])

    report = {"config": config, "n": n}
    for m in FRACTION_METRICS:
        gaps = [abs(a - b) for a, b in zip(ours[m], theirs[m])]
        report[m] = {
            "rmse": rmse(ours[m], theirs[m]),
            "max_gap": max(gaps) if gaps else 0.0,
            "exact": sum(g < 1e-9 for g in gaps),
        }
    adh_correct = sum(a == b for a, b in zip(ours["adherence"], theirs["adherence"]))
    report["adherence"] = {"accuracy": adh_correct / n if n else 0.0, "correct": adh_correct}
    return report


def validate_all(n=300):
    """Validate every config in every domain; return a flat list of report cards."""
    reports = []
    for domain, configs in DOMAIN_CONFIGS.items():
        for config in configs:
            r = validate_config(config, n=n)
            r["domain"] = domain
            reports.append(r)
    return reports
