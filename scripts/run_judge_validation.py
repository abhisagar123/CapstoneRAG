"""Local judge-validation runner — the terminal twin of notebooks/03_judge_validation.ipynb.

Same job, same output: validate a judge against RAGBench reference scores over one
representative config per domain, writing a resumable CSV. It calls the SAME shared
helper the notebook calls (evaluator.judge_validate.run_validation_sweep), so the two
can't drift — the notebook stays the Colab-portable artifact; this is the convenient
local path (no Jupyter needed).

Policy: open-source models locally, NOT Chinese models (no Qwen). Default = local Ollama.

Prereqs (local Ollama path):
    open the Ollama app (or `ollama serve`), then `ollama pull llama3.1:8b`

Run:
    .venv-eda/bin/python scripts/run_judge_validation.py
    .venv-eda/bin/python scripts/run_judge_validation.py --model mistral --conservative
    .venv-eda/bin/python scripts/run_judge_validation.py --backend hf --model meta-llama/Llama-3.1-8B-Instruct

Then read all results (this run + any others) with the verdict cell in nb03, or:
    .venv-eda/bin/python scripts/run_judge_validation.py --verdict
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One representative config per domain (matches nb03). cuad omitted by default —
# long contracts are slow / may OOM the hf path; pass --with-cuad to include it.
DEFAULT_CONFIGS = [
    ("Biomedical", "covidqa"),
    ("GenKnowledge", "hotpotqa"),
    ("CustomerSupport", "emanual"),
    ("Finance", "tatqa"),
]


def build_judge(backend: str, model: str, conservative: bool):
    import src  # noqa: F401 — light registry
    from src.judge import load_judges
    from src.registry import build
    load_judges()
    if backend == "ollama":
        return build("judge", "ollama", {"model": model, "conservative": conservative})
    return build("judge", "hf", {"model": model, "load_in_4bit": True,
                                 "max_new_tokens": 1536, "conservative": conservative})


def print_verdict():
    """Glob every judge_validation CSV and print the per-(model, variant) summary —
    the same view as nb03's verdict cell, for a terminal run."""
    import pandas as pd
    frames = []
    for path in sorted(glob.glob("results/validation/judge_validation*.csv")):
        d = pd.read_csv(path)
        if "prompt_variant" not in d.columns:
            d["prompt_variant"] = "baseline"
        frames.append(d)
    if not frames:
        print("no results/validation/judge_validation*.csv yet — run a sweep first.")
        return
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["model", "prompt_variant", "config"], keep="last")
    summary = df.groupby(["model", "prompt_variant"]).agg(
        configs=("config", "nunique"),
        relevance_rmse=("relevance_rmse", "mean"),
        relevance_signed=("relevance_signed", "mean"),
        utilization_rmse=("utilization_rmse", "mean"),
        completeness_rmse=("completeness_rmse", "mean"),
        adherence_acc=("adherence_acc", "mean"),
    ).round(3)
    print("Mean judge<->reference agreement per (model, prompt_variant):\n")
    print(summary)
    print("\n  lower rmse + |signed| = better; conservative vs baseline shows if the steer helped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    ap.add_argument("--model", default="llama3.1:8b", help="judge model (NON-Chinese)")
    ap.add_argument("--n", type=int, default=50, help="examples per config")
    ap.add_argument("--conservative", action="store_true", help="append the conservative steer")
    ap.add_argument("--with-cuad", action="store_true", help="also validate Legal/cuad (slow)")
    ap.add_argument("--workers", type=int, default=3,
                    help="judge examples concurrently within each config (~2x at 3; 1 = serial)")
    ap.add_argument("--verdict", action="store_true", help="just print the merged summary and exit")
    args = ap.parse_args()

    if args.verdict:
        print_verdict()
        return

    from src.evaluator.judge_validate import run_validation_sweep
    configs = list(DEFAULT_CONFIGS)
    if args.with_cuad:
        configs.insert(2, ("Legal", "cuad"))

    print(f"backend={args.backend}  model={args.model}  N={args.n}  workers={args.workers}  "
          f"conservative={args.conservative}  configs={[c for _, c in configs]}\n")
    judge = build_judge(args.backend, args.model, args.conservative)
    run_validation_sweep(judge, args.model, configs, n=args.n,
                         conservative=args.conservative, workers=args.workers)
    print("\n--- verdict so far ---")
    print_verdict()


if __name__ == "__main__":
    main()
