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


def _judge_one_example(judge, ex, config):
    """Judge a single example. Returns (scores_dict | None, reference_dict).

    None scores => the judge failed on this example (unparseable JSON after retry);
    the caller counts it in n_failed and skips it. This is the unit of parallel work —
    it has NO shared state, so judging examples concurrently is safe; each returns its
    own (scores, reference) pair and the caller collects them IN ORDER.
    """
    keyed = _keyed_from_example(ex)
    reference = {m: ex[REFERENCE_FIELD[m]] for m in REFERENCE_FIELD}
    try:
        label = judge.label(ex["question"], keyed)
        scores = scores_from_label(keyed, label)
    except (ValueError, KeyError) as e:
        print(f"  [skip] {config}: judge failed on one example ({e})")
        return None, reference
    return scores, reference


def validate_judge(judge, config: str, n: int = 50, split: str = "test",
                   workers: int = 1) -> dict:
    """Run `judge` on n examples of a config; compare its scores to the reference.

    Returns a report: RMSE + mean abs error per fraction, accuracy for adherence,
    plus `n` (examples attempted), `n_scored`, and `n_failed`. Imports `datasets` lazily.

    ROBUSTNESS: OSS judges sometimes emit JSON we can't parse even after a retry. One
    such example must NOT kill the whole config — we skip it, count it in `n_failed`,
    and score the rest. The failure RATE is itself a judge-quality signal (a model that
    fails often is a worse choice), so we surface it rather than hiding it.

    PARALLELISM: `workers` > 1 judges the n examples CONCURRENTLY (one Ollama server with
    GPU headroom batches them — ~2x faster at workers=3). This is ONLY parallel within
    THIS config; the caller (run_validation_sweep) still does domains sequentially and
    writes the CSV from one thread. Order is preserved (executor.map), so ours[i] still
    pairs with ref[i] — the result is identical to the serial version, just faster.
    """
    from datasets import load_dataset

    ds = load_dataset(REPO, config, split=split)
    n = min(n, len(ds))
    examples = [ds[i] for i in range(n)]

    # Judge every example -> ordered list of (scores|None, reference). Same work whether
    # serial (workers=1) or concurrent; map() preserves input order either way.
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda ex: _judge_one_example(judge, ex, config), examples))
    else:
        results = [_judge_one_example(judge, ex, config) for ex in examples]

    # Collect IN ORDER. Skipped (None) examples are counted, not scored.
    ours = {m: [] for m in REFERENCE_FIELD}
    ref = {m: [] for m in REFERENCE_FIELD}
    n_failed = 0
    for scores, reference in results:
        if scores is None:
            n_failed += 1
            continue
        for m in REFERENCE_FIELD:
            ours[m].append(scores[m])
            ref[m].append(reference[m])

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


# ── the resumable sweep + CSV writer (shared by nb03 AND scripts/run_judge_validation.py) ──
# Kept here so the notebook and the local script call ONE implementation and can't drift.

SWEEP_COLS = ["model", "prompt_variant", "domain", "config", "n", "n_scored", "n_failed",
              "relevance_rmse", "relevance_mae", "relevance_signed",
              "utilization_rmse", "utilization_mae", "utilization_signed",
              "completeness_rmse", "completeness_mae", "completeness_signed",
              "adherence_acc", "adherence_over_flag", "adherence_under_flag"]


def sweep_csv_path(model: str, prompt_variant: str, n: int) -> str:
    """results/judge_validation__{model-slug}__{variant}__n{N}.csv — one file per
    (model, variant, N) so parallel runs never collide and the verdict cell can glob+merge."""
    import re
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()
    return f"results/judge_validation__{slug}__{prompt_variant}__n{n}.csv"


def _flatten_report(report: dict, model: str, prompt_variant: str, domain: str) -> dict:
    r = {"model": model, "prompt_variant": prompt_variant, "domain": domain,
         "config": report["config"], "n": report["n"],
         "n_scored": report["n_scored"], "n_failed": report["n_failed"]}
    for m in FRACTIONS:
        r[f"{m}_rmse"] = report[m]["rmse"]
        r[f"{m}_mae"] = report[m]["mean_abs_err"]
        r[f"{m}_signed"] = report[m]["signed_err"]          # + => judge OVER-marks vs reference
    r["adherence_acc"] = report["adherence"]["accuracy"]
    r["adherence_over_flag"] = report["adherence"]["over_flag"]
    r["adherence_under_flag"] = report["adherence"]["under_flag"]
    return r


def run_validation_sweep(judge, model: str, configs, *, n: int = 50,
                         conservative: bool = False, split: str = "test",
                         workers: int = 1) -> str:
    """Validate `judge` over (domain, config) pairs and write a RESUMABLE CSV.

    `configs` = list of (domain, config) tuples. `model` is the model NAME (for the
    filename + a CSV column); `conservative` only labels the variant here (the judge
    object already carries the actual flag). Rows are written + flushed per config and
    already-done (model, config) pairs are SKIPPED, so a re-run resumes. Returns the
    CSV path. This is the ONE implementation nb03 and the local script share.

    `workers` is passed to validate_judge -> judges examples WITHIN each config
    concurrently (~2x at 3). Domains stay SEQUENTIAL and the CSV is written from this
    one thread, so domains can never interleave — only same-config examples run in
    parallel, and their results are recombined in order before the row is written.
    """
    import os
    import csv

    prompt_variant = "conservative" if conservative else "baseline"
    out = sweep_csv_path(model, prompt_variant, n)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    for domain, config in configs:
        # Recompute `done` from the CURRENT on-disk file each iteration (not once up
        # front) so resume is correct even if the file changed between configs.
        done = _read_done(out)
        if (model, config) in done:
            print(f"skip (done): {model} / {config}")
            continue
        print(f"judging {n}x {domain}/{config}  [{model} / {prompt_variant}] "
              f"(workers={workers}) ...", flush=True)
        report = validate_judge(judge, config, n=n, split=split, workers=workers)
        # Re-open in append mode PER ROW (open -> write -> close). A long run that held
        # one handle open got orphaned when the file was swapped mid-run (git/IDE replaced
        # it -> writes went to a dead inode). Re-opening always targets the live file.
        _append_row(out, _flatten_report(report, model, prompt_variant, domain))
        rel = report["relevance"]
        print(f"   -> rel_rmse={rel['rmse']:.3f} (signed {rel['signed_err']:+.3f})  "
              f"adh_acc={report['adherence']['accuracy']:.2f}  "
              f"(scored {report['n_scored']}/{report['n']}, {report['n_failed']} unparseable)")
    print(f"done. wrote {out}")
    return out


def _read_done(path: str) -> set:
    """The (model, config) pairs already in the CSV — for resume. Empty if no file."""
    import os
    import csv
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["model"], r["config"]) for r in csv.DictReader(f)}


def _append_row(path: str, row: dict) -> None:
    """Append ONE row, opening + closing the file each call (survives a mid-run file
    swap). Writes the header first if the file doesn't exist yet."""
    import os
    import csv
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
