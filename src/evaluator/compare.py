"""Compare OUR pipeline's TRACe scores to RAGBench's shipped REFERENCE scores.

This is the second of two comparisons against the reference scores — and it is a
DIFFERENT comparison from judge-validation, which is the easy thing to confuse it
with. Keep them straight:

  * judge-validation (evaluator/judge_validate.py): feed the judge the SAME inputs
    the reference was built on (the example's own gold documents + gold answer) and
    ask "does OUR judge agree with GPT-4?". That CALIBRATES the ruler.

  * THIS module: take the scores OUR pipeline earned (results/ragbench_matrix.csv —
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
