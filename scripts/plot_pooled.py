"""Plot the pooled experiment matrix — configs compared AMONG each other.

The pooled track has no reference comparison (our retrieval universe != the reference's),
so this charts pooled configs against EACH OTHER: one PNG per domain, x-axis = config, four
grouped TRACe bars per config. Merges every results/pooled/*.csv (pooled runs are split
across files, one per parallel --out) into one view.

⚠️ GENERATOR-MODEL COLLISION (why --exclude defaults to the 8B files): the merge keys on
(config_name, domain), but the generator MODEL is not a column — only the stage type
(`ollama`) is. So a config run on a bigger generator (e.g. llama3.1:8b, written to a
`*_llama8b.csv` file) has the SAME config_name as its 3B run and would silently overwrite
it in the merge. We therefore keep model-variant files OUT of the main (3B) charts by
default and chart them separately (see --only).

Run:
    .venv-eda/bin/python scripts/plot_pooled.py                          # main 3B track
    .venv-eda/bin/python scripts/plot_pooled.py --fig-dir results/pooled/figures
    .venv-eda/bin/python scripts/plot_pooled.py --only _llama8b \
        --fig-dir results/pooled/figures-8b                              # the 8B-only chart
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Filename substrings that encode a NON-default generator model. The merge key doesn't
# carry the model (see module docstring), so these must be charted on their own — left in
# the main merge they'd overwrite the 3B row for the same config_name. Add new model
# suffixes here (e.g. "_gemma9b") as they're introduced.
MODEL_VARIANT_MARKERS = ("_llama8b", "_gemma9b")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/pooled/*.csv",
                    help="pooled matrix CSVs to merge + plot (default: all results/pooled/*.csv)")
    ap.add_argument("--fig-dir", default="results/pooled/figures", help="where PNGs are written")
    ap.add_argument("--exclude", nargs="*", default=list(MODEL_VARIANT_MARKERS),
                    help="skip CSVs whose filename contains any of these substrings "
                         "(default: the model-variant files, so the main charts are the 3B track).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="chart ONLY CSVs whose filename contains one of these substrings "
                         "(overrides --exclude). Use for the dedicated 8B chart: --only _llama8b.")
    args = ap.parse_args()

    from src.evaluator.compare import load_matrix_rows, plot_matrix

    paths = sorted(glob.glob(args.glob))
    if args.only:
        paths = [p for p in paths if any(s in os.path.basename(p) for s in args.only)]
    else:
        paths = [p for p in paths
                 if not any(ex and ex in os.path.basename(p) for ex in args.exclude)]
    if not paths:
        sel = f"--only {args.only}" if args.only else args.glob
        print(f"no pooled CSVs match {sel} — run a pooled experiment first.")
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
