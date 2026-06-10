"""Local results-matrix runner — the terminal twin of notebooks/04_run_matrix.ipynb.

THE HEADLINE TASK-1 DELIVERABLE: run our RAG pipeline over (config × domain), score
each answer with the chosen judge, and write the strategy × domain TRACe matrix. Calls
the SAME src helpers the notebook uses (build_grid + run_named_matrix), so the two
can't drift; the notebook stays Colab-portable, this is the convenient local path.

Policy: open-source models locally, NOT Chinese (no Qwen). Default = local Ollama.

Prereqs (local Ollama path): open the Ollama app, then pull BOTH models:
    ollama pull llama3.1:8b      # the judge (scores our answers)
    ollama pull llama3.2:3b      # the generator (writes our answers)

Run:
    .venv-eda/bin/python scripts/run_matrix.py
    .venv-eda/bin/python scripts/run_matrix.py --domains GenKnowledge CustomerSupport --n 10
    .venv-eda/bin/python scripts/run_matrix.py --verdict          # just print the pivot

Reads configs/*.yaml as the strategy grid. Resumable + file-swap-safe.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_CSV = "results/ragbench_matrix.csv"


def build_judge(backend, model):
    from src.judge import load_judges
    from src.registry import build
    load_judges()
    if backend == "ollama":
        # OllamaJudge takes no max_new_tokens (output is bounded; num_ctx sizes the window).
        return build("judge", "ollama", {"model": model})
    return build("judge", "hf", {"model": model, "load_in_4bit": True, "max_new_tokens": 1536})


def print_verdict():
    """Pivot the matrix CSV to strategy × domain, per metric (same view as nb04)."""
    import pandas as pd
    if not os.path.exists(OUT_CSV):
        print(f"no {OUT_CSV} yet — run the matrix first.")
        return
    df = pd.read_csv(OUT_CSV)
    print("Raw rows:\n")
    cols = ["config_name", "domain", "n", "n_scored", "relevance", "utilization",
            "completeness", "adherence", "scoring"]
    print(df[cols].round(3).to_string(index=False))
    print("\nStrategy × domain, per metric:")
    for metric in ["relevance", "utilization", "completeness", "adherence"]:
        print(f"\n--- {metric} ---")
        print(df.pivot_table(index="config_name", columns="domain", values=metric).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    ap.add_argument("--judge-model", default="llama3.1:8b")
    ap.add_argument("--gen-model", default="llama3.2:3b")
    ap.add_argument("--domains", nargs="+", default=["GenKnowledge", "CustomerSupport"])
    ap.add_argument("--n", type=int, default=10, help="examples per (config, domain)")
    ap.add_argument("--workers", type=int, default=1, help="(reserved) per-example concurrency")
    ap.add_argument("--verdict", action="store_true", help="just print the matrix pivot and exit")
    args = ap.parse_args()

    if args.verdict:
        print_verdict()
        return

    import yaml
    import src  # noqa: F401
    from src.embeddings import load_embedders
    from src.generation import load_generators
    from src.reranking import load_rerankers
    from src.data_loader import load_domain
    from src.segmentation import OutputSegmenter, RegexSplitter
    from src.runner import build_grid, run_named_matrix
    load_embedders(); load_generators(); load_rerankers()

    judge = build_judge(args.backend, args.judge_model)
    gen_override = ({"type": "ollama", "model": args.gen_model, "max_new_tokens": 256}
                    if args.backend == "ollama"
                    else {"type": "hf", "model": args.gen_model, "load_in_4bit": True})

    raw = {os.path.basename(p): yaml.safe_load(open(p)) for p in sorted(glob.glob("configs/*.yaml"))}
    grid = build_grid(raw, args.domains, generator_override=gen_override)
    print(f"backend={args.backend}  judge={args.judge_model}  gen={args.gen_model}  "
          f"domains={args.domains}  N={args.n}  -> {len(grid)} (config x domain) runs\n")

    seg = OutputSegmenter(RegexSplitter())
    cache = {}
    def examples_for(cfg):
        if cfg.domain not in cache:
            cache[cfg.domain] = load_domain(cfg.domain, split="test", n=args.n, seed=42)
        return cache[cfg.domain]

    run_named_matrix(grid, examples_for, OUT_CSV, segmenter=seg, judge=judge)
    print("\n--- matrix so far ---")
    print_verdict()


if __name__ == "__main__":
    main()
