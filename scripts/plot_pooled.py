"""Plot the pooled experiment matrix — configs compared AMONG each other.

The pooled track has no reference comparison (our retrieval universe != the reference's),
so this charts pooled configs against EACH OTHER: one PNG per domain, x-axis = config, four
grouped TRACe bars per config. Merges every results/pooled/*.csv (pooled runs are split
across files, one per parallel --out) into one view.

Run:
    .venv-eda/bin/python scripts/plot_pooled.py
    .venv-eda/bin/python scripts/plot_pooled.py --fig-dir results/pooled/figures
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/pooled/*.csv",
                    help="pooled matrix CSVs to merge + plot (default: all results/pooled/*.csv)")
    ap.add_argument("--fig-dir", default="results/pooled/figures", help="where PNGs are written")
    args = ap.parse_args()

    from src.evaluator.compare import load_matrix_rows, plot_matrix

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"no pooled CSVs match {args.glob} — run a pooled experiment first.")
        return
    print(f"merging {len(paths)} pooled CSV(s): {[os.path.basename(p) for p in paths]}")
    rows = load_matrix_rows(*paths)
    n_configs = len({r.get("config_name") or r.get("config_id") for r in rows})
    print(f"{len(rows)} rows across {n_configs} configs")
    pngs = plot_matrix(rows, args.fig_dir, title_prefix="pooled — ")
    print("wrote charts:")
    for p in pngs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
