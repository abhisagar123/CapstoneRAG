"""Compare OUR pipeline's TRACe scores to RAGBench's shipped REFERENCE scores.

This is the second of two comparisons against the reference scores — and it is a
DIFFERENT comparison from judge-validation, which is the easy thing to confuse it
with. Keep them straight:

  * judge-validation (evaluator/judge_validate.py): feed the judge the SAME inputs
    the reference was built on (the example's own gold documents + gold answer) and
    ask "does OUR judge agree with GPT-4?". That CALIBRATES the ruler.

  * THIS module: take the scores OUR pipeline earned (results/per_example/ragbench_matrix.csv —
    our answer, over our retrieved context) and lay them next to the reference
    scores the dataset ships, per domain. That MEASURES our system with the
    calibrated ruler, and exposes WHERE we lag → which component to improve.

Why this is cheap and exact:
  - The matrix sampled its examples deterministically: load_domain(domain, "test",
    n=N, seed=42). We reproduce that EXACT sample here, so the reference baseline is
    a fair head-to-head — the same questions, not a different slice.
  - The reference TRACe scores ship as plain FIELDS on every example
    (relevance_score, utilization_score, completeness_score, adherence_score). We
    proved in notebook 02 that these equal our own math to RMSE 0, so averaging the
    shipped fields IS the reference — no LLM, no Ollama, instant and fully local.

The symmetry worth stating in the report — both numbers end in the same trace.py math:
    OUR number:        our answer + our context  -> judge labels     -> TRACe
    REFERENCE number:  gold answer + gold context -> gold labels      -> TRACe
The only things that differ are (a) what got retrieved/generated and (b) gold-labels
vs judge-labels. That is exactly what the gap exposes.

How to read each metric's gap (gap = ours - reference) — they are NOT symmetric:
  * relevance   — CLEANEST retriever signal. Reference = relevant-density of the raw
                  document pile; ours = relevant-density of what our retriever
                  surfaced. A good retriever FILTERS junk, so it should push ours
                  ABOVE reference. Ours < ref => the retriever is dropping relevant
                  material. (Caveat: ours is judged by the 8b judge, which mildly
                  over-marks relevance ~+0.09 — so a small positive gap may be judge
                  bias, not pure retrieval gain. The bias is ~constant across configs,
                  so config-vs-config comparison stays valid.)
  * adherence   — reference = a real system's (gpt-3.5) faithfulness rate; matching or
                  beating it is the goal. Points at the prompt + generator.
  * utilization — mixes retrieval AND generation; triangulate with the other two.
  * completeness— a derived ratio; noisiest; read last.
"""

from .validate import REFERENCE_FIELD, FRACTION_METRICS
from ..data_loader import load_domain

# The four TRACe metrics, fractions first then the boolean adherence.
METRICS = FRACTION_METRICS + ["adherence"]


def _mean_reference_fields(examples) -> dict:
    """Average the shipped reference score FIELDS over a list of examples.

    Each example is a raw RAGBench dict, so it carries `relevance_score`,
    `utilization_score`, `completeness_score` (floats) and `adherence_score` (bool).
    `float(bool)` is 0/1, so the adherence mean is the reference adherence RATE —
    the same shape as how the matrix averages OUR pipeline's adherence booleans.

    Returns {metric: mean, ...} plus `_n_ref` = how many examples were averaged.
    """
    means = {}
    for metric in METRICS:
        field = REFERENCE_FIELD[metric]
        vals = [float(ex[field]) for ex in examples]      # bool -> 0/1 for adherence
        means[metric] = sum(vals) / len(vals) if vals else float("nan")
    means["_n_ref"] = len(examples)
    return means


def reference_means(domain: str, n: int | None = None, seed: int = 42,
                    split: str = "test") -> dict:
    """Mean shipped reference TRACe over the SAME examples the matrix sampled.

    Reproduces the matrix's sampling exactly (load_domain with the same n + seed=42),
    so this is a fair head-to-head, not a different example set. Pass `n=None` for the
    whole-split baseline (a more stable yardstick — still instant, the fields are free).

    Imports `datasets` lazily (inside load_domain), so importing this module is cheap.
    """
    return _mean_reference_fields(load_domain(domain, split=split, n=n, seed=seed))


def _to_float(value):
    """Parse a matrix-CSV score cell to float, or None if blank/unparseable.

    Matrix scores arrive as strings; a blank cell ("") means scoring was pending
    (no judge) — those rows have no gap to compute, so we return None and the caller
    leaves the gap blank rather than crashing.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_comparison(matrix_csv: str, *, seed: int = 42, split: str = "test",
                     match_n: bool = True, reference_fn=reference_means) -> list[dict]:
    """Join each matrix row with the reference means for its (domain, n).

    Returns a list of dicts: every original matrix column, plus `{metric}_ref` and
    `{metric}_gap` (gap = ours - reference) for each of the four TRACe metrics.

    - `match_n=True` (default): the reference is averaged over the SAME N examples the
      matrix ran for that domain (fair head-to-head). `match_n=False`: over the whole
      test split (stable yardstick).
    - `reference_fn` is injectable so the offline test can supply fixed means without
      touching the network/dataset.

    The reference means are CACHED per (domain, n): every strategy in a domain shares
    the same baseline, so we load each domain's examples once, not once per config.
    """
    import csv

    with open(matrix_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    ref_cache: dict = {}
    out: list[dict] = []
    for r in rows:
        domain = r["domain"]
        n = int(r["n"]) if match_n else None
        key = (domain, n)
        if key not in ref_cache:
            ref_cache[key] = reference_fn(domain, n=n, seed=seed, split=split)
        ref = ref_cache[key]

        row = dict(r)                                       # keep all original columns
        for m in METRICS:
            ours = _to_float(r.get(m))
            row[f"{m}_ref"] = ref[m]
            row[f"{m}_gap"] = (ours - ref[m]) if ours is not None else None
        out.append(row)
    return out


def comparison_fieldnames(rows: list[dict]) -> list[str]:
    """Column order for the output CSV: the matrix's own columns, then for each metric
    its `_ref` and `_gap` neighbours (so each metric's three numbers sit together)."""
    if not rows:
        return []
    base = [k for k in rows[0] if not (k.endswith("_ref") or k.endswith("_gap"))]
    extra = []
    for m in METRICS:
        extra += [f"{m}_ref", f"{m}_gap"]
    return base + extra


def write_comparison_csv(rows: list[dict], path: str) -> None:
    """Write the joined comparison rows to a CSV (the report artifact)."""
    import csv
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=comparison_fieldnames(rows))
        w.writeheader()
        w.writerows(rows)


# ── per-metric gap summary: BIAS + SPREAD (deliberately NOT RMSE) ───────────────────
# Across the comparison's cells (one per config×domain) each metric has a set of gaps
# (ours - reference). We summarise that set with TWO numbers, not one:
#   bias   = signed mean of the gaps  -> WHICH WAY and HOW FAR we sit vs the benchmark.
#   spread = std-dev of the gaps      -> how CONSISTENT that offset is across configs.
#
# Why not RMSE (the question everyone asks): RMSE measures distance-from-a-bullseye and
# THROWS THE SIGN AWAY. But here the sign IS the finding — relevance ABOVE reference is a
# retriever WIN, adherence BELOW is the real hallucination gap — so an "error" magnitude
# would mislabel a win as a defect. And because RMSE^2 = bias^2 + spread^2, RMSE MERGES the
# two things we need apart: e.g. utilization has bias~=0 but spread~=0.07 ("dead-on average
# but config-dependent"), a completely different story from relevance's steady +0.10 offset
# — RMSE collapses both to ~0.07 and hides which is which. (RMSE is the RIGHT tool for JUDGE
# VALIDATION, where ours and the reference score the SAME inputs so a per-example error is
# real; THAT is a different comparison — see evaluator/judge_validate.py.)

def _mean_std(values: list[float]) -> tuple[float, float]:
    """Population mean and std-dev of a non-empty list (std=0 for a single value)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5


def summarize_gaps(rows: list[dict]) -> list[dict]:
    """Collapse the per-cell comparison into one bias+spread row PER metric.

    `rows` are build_comparison() output. For each TRACe metric we gather every cell's
    `{metric}_gap` (skipping blanks — a pending/unscored cell has no gap), then report:
        metric, n_cells, bias (signed mean gap), spread (std of gaps),
        gap_min, gap_max  (the signed range — so a one-sided offset is obvious at a glance).
    Metrics with no scored cells are omitted. Deliberately NO rmse column — see the note
    above; a bare "RMSE" here would be mistaken for the judge-validation error metric.
    """
    summary = []
    for m in METRICS:
        gaps = [r[f"{m}_gap"] for r in rows
                if r.get(f"{m}_gap") is not None and _to_float(r[f"{m}_gap"]) is not None]
        gaps = [float(g) for g in gaps]
        if not gaps:
            continue
        bias, spread = _mean_std(gaps)
        summary.append({"metric": m, "n_cells": len(gaps),
                        "bias": bias, "spread": spread,
                        "gap_min": min(gaps), "gap_max": max(gaps)})
    return summary


SUMMARY_COLS = ["metric", "n_cells", "bias", "spread", "gap_min", "gap_max"]


def write_summary_csv(summary: list[dict], path: str) -> None:
    """Write the per-metric bias+spread summary to its own CSV (the headline artifact —
    the one number-per-metric a reader scans before the per-cell detail)."""
    import csv
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(summary)


def _strategy_label(row: dict) -> str:
    """Short strategy name for a chart axis: prefer the YAML filename without its
    extension (e.g. 'grounded_rerank'), fall back to the config_id."""
    name = row.get("config_name") or row.get("config_id") or "?"
    return name[:-5] if name.endswith(".yaml") else name


def plot_comparison(rows: list[dict], out_dir: str, *, baseline_label: str = "") -> list[str]:
    """Grouped bar charts — ours vs reference — one PNG per metric, faceted by domain.

    For each TRACe metric we draw a figure whose subplots are the domains; within a
    subplot the x-axis is the strategy (config) and each strategy shows two bars: our
    pipeline's score and the dataset's reference score, side by side. The visual gap
    between the pair IS the finding. Returns the list of written PNG paths.

    matplotlib is imported HERE (lazily), not at module top, so importing compare.py
    never drags in matplotlib — same heavy-dep discipline the rest of the codebase uses.
    Pending rows (blank score -> {metric} not numeric) are dropped from that metric's plot.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")                      # headless: write files, never open a window
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    domains = sorted({r["domain"] for r in rows})
    written = []

    for metric in METRICS:
        ref_col, gap_col = f"{metric}_ref", f"{metric}_gap"
        fig, axes = plt.subplots(1, len(domains), figsize=(5.5 * len(domains), 4.2),
                                 squeeze=False)
        for ax, domain in zip(axes[0], domains):
            drows = [r for r in rows if r["domain"] == domain and _to_float(r.get(metric)) is not None]
            labels = [_strategy_label(r) for r in drows]
            ours = [_to_float(r[metric]) for r in drows]
            ref = [r.get(ref_col) for r in drows]
            x = range(len(labels))
            width = 0.38
            ax.bar([i - width / 2 for i in x], ours, width, label="ours", color="#4C72B0")
            ax.bar([i + width / 2 for i in x], ref, width, label="reference", color="#C44E52")
            ax.set_title(domain)
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
        sub = f"  ({baseline_label})" if baseline_label else ""
        fig.suptitle(f"{metric}: ours vs RAGBench reference{sub}", fontweight="bold")
        fig.tight_layout()
        path = os.path.join(out_dir, f"compare_{metric}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


# ── matrix-only plots (no reference) — for comparing configs AMONG each other ───────
# Used by the pooled track, where there is no reference comparison: each domain gets one
# chart with the configs on the x-axis and all 4 TRACe metrics grouped per config, so you
# can read each config's full profile and compare configs side by side.

def plot_matrix(rows: list[dict], out_dir: str, *, title_prefix: str = "") -> list[str]:
    """Grouped bar charts of a results matrix — configs compared AMONG each other.

    One PNG per domain; x-axis = config (strategy), and each config shows its four TRACe
    metrics (relevance/utilization/completeness/adherence) as grouped bars. No reference —
    this is for tracks (e.g. pooled) where configs are compared to each other, not to the
    RAGBench reference. Returns the written PNG paths.

    `rows` are raw matrix-CSV dicts (string scores). matplotlib is imported lazily here,
    same heavy-dep discipline as plot_comparison.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")                      # headless
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    domains = sorted({r["domain"] for r in rows})
    colors = {"relevance": "#4C72B0", "utilization": "#DD8452",
              "completeness": "#55A868", "adherence": "#C44E52"}
    written = []

    for domain in domains:
        drows = [r for r in rows if r["domain"] == domain]
        labels = [_strategy_label(r) for r in drows]
        x = range(len(labels))
        n_metrics = len(METRICS)
        width = 0.8 / n_metrics                 # bars share each config's slot
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 4.6))
        for j, metric in enumerate(METRICS):
            vals = [(_to_float(r.get(metric)) or 0.0) for r in drows]
            offsets = [i + (j - (n_metrics - 1) / 2) * width for i in x]
            ax.bar(offsets, vals, width, label=metric, color=colors[metric])
        ax.set_title(f"{title_prefix}{domain}", fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("TRACe score")
        ax.legend(fontsize=8, ncol=4, loc="upper right")
        fig.tight_layout()
        safe_dom = domain.lower().replace(" ", "_")
        path = os.path.join(out_dir, f"matrix_{safe_dom}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def load_matrix_rows(*csv_paths: str) -> list[dict]:
    """Read + concatenate one or more matrix CSVs into a list of row dicts.

    Pooled experiments are split across files (separate --out per parallel run), so the
    plot/analysis layer needs to merge them. De-dupes on (config_name, domain), keeping the
    last occurrence, so re-running a config supersedes its older row."""
    import csv

    merged: dict = {}
    for path in csv_paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                merged[(r.get("config_name") or r.get("config_id"), r["domain"])] = r
    return list(merged.values())
