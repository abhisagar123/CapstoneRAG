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
    plus `n` (examples attempted), `n_scored`, and `n_failed`. Imports `datasets` lazily.

    ROBUSTNESS: OSS judges sometimes emit JSON we can't parse even after a retry. One
    such example must NOT kill the whole config — we skip it, count it in `n_failed`,
    and score the rest. The failure RATE is itself a judge-quality signal (a model that
    fails often is a worse choice), so we surface it rather than hiding it.
    """
    from datasets import load_dataset

    ds = load_dataset(REPO, config, split=split)
    n = min(n, len(ds))
    ds = ds.select(range(n))

    ours = {m: [] for m in REFERENCE_FIELD}
    ref = {m: [] for m in REFERENCE_FIELD}
    n_failed = 0
    for ex in ds:
        keyed = _keyed_from_example(ex)
        try:
            label = judge.label(ex["question"], keyed)
            s = scores_from_label(keyed, label)
        except (ValueError, KeyError) as e:
            n_failed += 1                              # unparseable/malformed -> skip, count
            print(f"  [skip] {config}: judge failed on one example ({e})")
            continue
        for m in REFERENCE_FIELD:
            ours[m].append(s[m])
            ref[m].append(ex[REFERENCE_FIELD[m]])

    n_scored = n - n_failed
    report = {"config": config, "n": n, "n_scored": n_scored, "n_failed": n_failed}
    for m in FRACTIONS:
        diffs = [a - b for a, b in zip(ours[m], ref[m])]          # SIGNED: + => judge over-marks
        gaps = [abs(d) for d in diffs]
        report[m] = {"rmse": rmse(ours[m], ref[m]),
                     "mean_abs_err": (sum(gaps) / len(gaps)) if gaps else 0.0,
                     # signed_err tells DIRECTION: if |signed_err| ~ mean_abs_err the bias is
                     # systematic (a prompt tweak can fix it); if signed_err ~ 0 but mae is high,
                     # the errors are random (the model just isn't capable -> needs a stronger judge).
                     "signed_err": (sum(diffs) / len(diffs)) if diffs else 0.0}
    # Accuracy is over SCORED examples, not all n (skipped ones aren't "wrong").
    # Also split adherence mistakes by DIRECTION (judge bool vs reference bool):
    #   over_flag  = judge said NOT supported (False) but reference said supported (True)
    #                -> the judge is too harsh / over-flags hallucination
    #   under_flag = judge said supported (True) but reference said NOT supported (False)
    #                -> the judge misses real hallucination
    adh_correct = sum(bool(a) == bool(b) for a, b in zip(ours["adherence"], ref["adherence"]))
    over_flag = sum((not bool(a)) and bool(b) for a, b in zip(ours["adherence"], ref["adherence"]))
    under_flag = sum(bool(a) and (not bool(b)) for a, b in zip(ours["adherence"], ref["adherence"]))
    report["adherence"] = {"accuracy": adh_correct / n_scored if n_scored else 0.0,
                           "over_flag": over_flag, "under_flag": under_flag}
    return report
