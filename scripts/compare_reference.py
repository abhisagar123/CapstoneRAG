"""Compare our pipeline's matrix scores to RAGBench's shipped REFERENCE scores.

The problem statement asks us to compare our computed TRACe scores against the
dataset's reference scores, analyse WHY they differ, and optimise. This is the
producer of that comparison: it reads results/ragbench_matrix.csv (our pipeline's
scores) and joins each row with the reference TRACe averaged over the SAME examples,
then writes results/reference_comparison.csv and prints per-metric gap tables.

This needs NO model and NO Ollama — the reference scores ship as fields on every
example and we proved (notebook 02) they equal our own math at RMSE 0, so the
reference baseline is just averaging those fields. It is instant and fully local;
it only reads the dataset (cached after the matrix run already pulled it).

Run:
    .venv-eda/bin/python scripts/compare_reference.py
    .venv-eda/bin/python scripts/compare_reference.py --full-split   # whole-test baseline
    .venv-eda/bin/python scripts/compare_reference.py --matrix results/ragbench_matrix.csv

gap = ours - reference. How to read each metric (printed below the tables too):
  relevance   -> retriever. A good retriever FILTERS junk, so positive is expected/good
                 (mind the ~+0.09 judge over-mark bias); NEGATIVE = retriever dropping
                 relevant material -> experiment on retrieval (k, embedder, hybrid, rerank).
  adherence   -> prompt + generator faithfulness; reference is a real system's rate, so
                 match-or-beat is the goal. Below -> grounding prompt / bigger generator.
  utilization -> retrieval AND generation mixed; triangulate.
  completeness-> derived ratio, noisiest; read last.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MATRIX_CSV = "results/per_example/ragbench_matrix.csv"
OUT_CSV = "results/per_example/reference_comparison.csv"

from src.evaluator.compare import METRICS  # noqa: E402  (after sys.path insert)


def _fmt_gap(v) -> str:
    """Signed gap with a + on positives; blank cell stays blank."""
    return "" if v is None else f"{v:+.3f}"


def _print_tables(rows, baseline_label: str):
    """Per-metric tables: ours | ref | gap, one row per (config_name, domain)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    label = "config_name" if "config_name" in df.columns else "config_id"
    print(f"\nReference comparison  (gap = ours - reference; baseline = {baseline_label})\n")

    for m in METRICS:
        ours_col, ref_col, gap_col = m, f"{m}_ref", f"{m}_gap"
        if ref_col not in df.columns:
            continue
        view = df[[label, "domain", ours_col, ref_col, gap_col]].copy()
        view.columns = [label, "domain", "ours", "reference", "gap"]
        for c in ("ours", "reference"):
            view[c] = pd.to_numeric(view[c], errors="coerce").round(3)
        view["gap"] = view["gap"].map(_fmt_gap)
        print(f"--- {m} ---")
        print(view.to_string(index=False))
        print()


def _print_summary(summary):
    """Per-metric BIAS + SPREAD table — the headline before the per-cell detail."""
    if not summary:
        print("(no scored cells yet — nothing to summarise)\n")
        return
    print("Per-metric gap summary  (across all config×domain cells)\n")
    print(f'{"metric":<13}{"bias":>9}{"spread":>9}{"range":>20}{"  cells":>8}')
    print("-" * 59)
    for s in summary:
        rng = f"[{s['gap_min']:+.3f}, {s['gap_max']:+.3f}]"
        print(f"{s['metric']:<13}{s['bias']:>+9.3f}{s['spread']:>9.3f}{rng:>20}{s['n_cells']:>8}")
    print()
    print("  bias   = signed mean gap (ours - reference): WHICH WAY + HOW FAR vs the benchmark.")
    print("  spread = std of the gap across configs: how CONSISTENT that offset is.")
    print("  Deliberately NOT RMSE: RMSE drops the SIGN (it would score our relevance WIN as")
    print("  'error'), and RMSE^2 = bias^2 + spread^2 MERGES the two numbers above into one,")
    print("  hiding e.g. 'dead-on average but config-dependent' (utilization) vs a steady")
    print("  offset (relevance). RMSE is the right tool for JUDGE validation (same inputs →")
    print("  real per-example error), not for this system-vs-benchmark comparison.\n")


def _print_reading_notes():
    print("How to read the gaps:")
    print("  relevance    retriever signal (cleanest). A good retriever filters junk -> ours")
    print("               should sit ABOVE reference. NEGATIVE => retriever dropping relevant")
    print("               material -> try bigger k / better embedder / hybrid / reranker.")
    print("               (Caveat: our relevance is judged by llama3.1:8b, which mildly")
    print("               over-marks ~+0.09 — a small positive gap may be judge bias, not gain.")
    print("               The bias is ~constant across configs, so config-vs-config stays valid.)")
    print("  adherence    prompt + generator faithfulness. Reference = a real system's rate;")
    print("               match-or-beat is the goal. Below => grounding prompt / bigger generator.")
    print("  utilization  retrieval AND generation mixed — triangulate with the other two.")
    print("  completeness derived ratio, noisiest — read last.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=MATRIX_CSV, help="our pipeline's matrix CSV")
    ap.add_argument("--out", default=OUT_CSV, help="where to write the comparison CSV")
    ap.add_argument("--full-split", action="store_true",
                    help="baseline over the WHOLE test split (stable yardstick) instead of "
                         "the same N examples the matrix ran (fair head-to-head, default)")
    ap.add_argument("--seed", type=int, default=42, help="must match the matrix's sampling seed")
    ap.add_argument("--plot", action="store_true",
                    help="also write grouped bar charts (ours vs reference) to the fig dir "
                         "(needs matplotlib)")
    ap.add_argument("--fig-dir", default="results/per_example/figures", help="where --plot writes PNGs")
    args = ap.parse_args()

    if not os.path.exists(args.matrix):
        print(f"no matrix at {args.matrix} — run scripts/run_matrix.py first.")
        return

    from src.evaluator.compare import (build_comparison, write_comparison_csv,
                                        summarize_gaps, write_summary_csv)

    match_n = not args.full_split
    baseline_label = "matched-N (same examples)" if match_n else "full test split"
    print(f"matrix={args.matrix}  baseline={baseline_label}  seed={args.seed}")
    rows = build_comparison(args.matrix, seed=args.seed, match_n=match_n)
    write_comparison_csv(rows, args.out)
    summary = summarize_gaps(rows)
    summary_path = args.out.replace(".csv", "_summary.csv")
    write_summary_csv(summary, summary_path)
    _print_summary(summary)
    _print_tables(rows, baseline_label)
    _print_reading_notes()
    print(f"\nwrote {args.out}  ({len(rows)} rows)")
    print(f"wrote {summary_path}  ({len(summary)} metrics)")

    if args.plot:
        from src.evaluator.compare import plot_comparison
        pngs = plot_comparison(rows, args.fig_dir, baseline_label=baseline_label)
        print("wrote charts:")
        for p in pngs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
