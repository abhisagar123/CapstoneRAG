"""Validate a JUDGE against RAGBench reference scores (§9.4).

The judge defines ground truth, so before trusting its scores we measure how well
they agree with the dataset's shipped reference scores. For labelled RAGBench
examples, we:
  1. feed the example's OWN keyed sentences (documents_sentences / response_sentences
     ship in the dataset) to the judge,
  2. map the judge's labels -> TRACe scores (scores_from_label),
  3. compare to the reference scores: RMSE on the 3 fractions, accuracy on adherence.

Good agreement (low RMSE, high adherence accuracy) => the judge is trustworthy for
scoring our own pipeline. This mirrors evaluator/validate.py (which validated the
MATH half); here we validate the JUDGE half. Runs on Colab (needs the model).
"""

from ..judge.base import scores_from_label
from .validate import rmse, REPO

FRACTIONS = ["relevance", "utilization", "completeness"]
REFERENCE_FIELD = {"relevance": "relevance_score", "utilization": "utilization_score",
                   "completeness": "completeness_score", "adherence": "adherence_score"}


def _keyed_from_example(ex) -> dict:
    """RAGBench ships documents_sentences/response_sentences already keyed — reuse them
    directly so the judge labels the same sentences the reference scores were built on."""
    return {"documents_sentences": ex["documents_sentences"],
            "response_sentences": ex["response_sentences"]}


def validate_judge(judge, config: str, n: int = 50, split: str = "test") -> dict:
    """Run `judge` on n examples of a config; compare its scores to the reference.

    Returns a report: RMSE + mean abs error per fraction, accuracy for adherence,
    and n. Imports `datasets` lazily.
    """
    from datasets import load_dataset

    ds = load_dataset(REPO, config, split=split)
    n = min(n, len(ds))
    ds = ds.select(range(n))

    ours = {m: [] for m in REFERENCE_FIELD}
    ref = {m: [] for m in REFERENCE_FIELD}
    for ex in ds:
        keyed = _keyed_from_example(ex)
        label = judge.label(ex["question"], keyed)
        s = scores_from_label(keyed, label)
        for m in REFERENCE_FIELD:
            ours[m].append(s[m])
            ref[m].append(ex[REFERENCE_FIELD[m]])

    report = {"config": config, "n": n}
    for m in FRACTIONS:
        gaps = [abs(a - b) for a, b in zip(ours[m], ref[m])]
        report[m] = {"rmse": rmse(ours[m], ref[m]),
                     "mean_abs_err": (sum(gaps) / len(gaps)) if gaps else 0.0}
    adh_correct = sum(bool(a) == bool(b) for a, b in zip(ours["adherence"], ref["adherence"]))
    report["adherence"] = {"accuracy": adh_correct / n if n else 0.0}
    return report
