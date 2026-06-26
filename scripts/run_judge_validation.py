"""Local judge-validation runner — the terminal twin of notebooks/03_judge_validation.ipynb.

Same job, same output: validate a judge against RAGBench reference scores over one
representative config per domain, writing a resumable CSV. It calls the SAME shared
helper the notebook calls (evaluator.judge_validate.run_validation_sweep), so the two
can't drift — the notebook stays the Colab-portable artifact; this is the convenient
local path (no Jupyter needed).

Policy: open-source / open-WEIGHT models, NOT Chinese (no Qwen). Default = local Ollama.

Prereqs (local Ollama path):
    open the Ollama app (or `ollama serve`), then `ollama pull llama3.1:8b`
Prereqs (Groq path — a BIGGER judge for the re-validation, hosted/fast):
    export GROQ_API_KEY=...   (console.groq.com; read lazily, never committed)

Run:
    .venv-eda/bin/python scripts/run_judge_validation.py
    .venv-eda/bin/python scripts/run_judge_validation.py --model mistral --conservative
    .venv-eda/bin/python scripts/run_judge_validation.py --backend hf --model meta-llama/Llama-3.1-8B-Instruct
    .venv-eda/bin/python scripts/run_judge_validation.py --backend groq   # bigger-judge re-validation (70B)

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


# Per-backend default judge model, so --backend groq doesn't inherit the Ollama tag.
DEFAULT_MODEL = {
    "ollama": "llama3.1:8b",                  # the LOCKED local scoring judge (baseline)
    "hf":     "meta-llama/Llama-3.1-8B-Instruct",
    "groq":   "llama-3.3-70b-versatile",      # the BIGGER judge for the re-validation
}


def build_judge(backend: str, model: str, conservative: bool, rpm: float = 0.0):
    import src  # noqa: F401 — light registry
    from src.judge import load_judges
    from src.registry import build
    load_judges()
    if backend == "ollama":
        return build("judge", "ollama", {"model": model, "conservative": conservative})
    if backend == "groq":
        # GroqJudge reads GROQ_API_KEY lazily; no num_ctx (Groq sizes context server-side).
        # requests_per_minute paces sends UNDER Groq's free-tier tokens/min budget.
        return build("judge", "groq", {"model": model, "conservative": conservative,
                                       "requests_per_minute": rpm})
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
    ap.add_argument("--backend", default="ollama", choices=["ollama", "hf", "groq"])
    ap.add_argument("--model", default=None,
                    help="judge model (NON-Chinese); defaults per-backend if omitted")
    ap.add_argument("--n", type=int, default=50, help="examples per config")
    ap.add_argument("--conservative", action="store_true", help="append the conservative steer")
    ap.add_argument("--with-cuad", action="store_true", help="also validate Legal/cuad (slow)")
    ap.add_argument("--workers", type=int, default=None,
                    help="judge examples concurrently within each config (default 3; "
                         "FORCED to 1 for --backend groq unless overridden — concurrency "
                         "bursts past Groq's tokens/min budget)")
    ap.add_argument("--rpm", type=float, default=None,
                    help="groq only: max requests/minute (proactive pacing under the TPM "
                         "budget). Default 5 (≈ safe for ~12k tokens/min at ~2k-tok prompts).")
    ap.add_argument("--verdict", action="store_true", help="just print the merged summary and exit")
    args = ap.parse_args()

    if args.verdict:
        print_verdict()
        return

    model = args.model or DEFAULT_MODEL[args.backend]   # per-backend default if not given

    # Groq's free tier limits TOKENS/min, and concurrency bursts past it -> for groq,
    # default to SERIAL (workers=1) + proactive pacing (rpm). Other backends keep workers=3.
    is_groq = args.backend == "groq"
    workers = args.workers if args.workers is not None else (1 if is_groq else 3)
    rpm = (args.rpm if args.rpm is not None else 5.0) if is_groq else 0.0

    from src.evaluator.judge_validate import run_validation_sweep
    configs = list(DEFAULT_CONFIGS)
    if args.with_cuad:
        configs.insert(2, ("Legal", "cuad"))

    pacing = f"  rpm={rpm}" if is_groq else ""
    print(f"backend={args.backend}  model={model}  N={args.n}  workers={workers}{pacing}  "
          f"conservative={args.conservative}  configs={[c for _, c in configs]}\n")
    judge = build_judge(args.backend, model, args.conservative, rpm=rpm)
    run_validation_sweep(judge, model, configs, n=args.n,
                         conservative=args.conservative, workers=workers)
    print("\n--- verdict so far ---")
    print_verdict()


if __name__ == "__main__":
    main()
