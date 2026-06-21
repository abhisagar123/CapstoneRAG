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

OUT_CSV = "results/per_example/ragbench_matrix.csv"


def build_judge(backend, model):
    from src.judge import load_judges
    from src.registry import build
    load_judges()
    if backend == "ollama":
        # OllamaJudge takes no max_new_tokens (output is bounded; num_ctx sizes the window).
        return build("judge", "ollama", {"model": model})
    if backend == "groq":
        # GroqJudge: paid; default model llama-3.3-70b-versatile. Reads GROQ_API_KEY from env.
        return build("judge", "groq", {"model": model})
    return build("judge", "hf", {"model": model, "load_in_4bit": True, "max_new_tokens": 1536})


def print_verdict(out_csv=OUT_CSV):
    """Pivot the matrix CSV to strategy × domain, per metric (same view as nb04)."""
    import pandas as pd
    if not os.path.exists(out_csv):
        print(f"no {out_csv} yet — run the matrix first.")
        return
    df = pd.read_csv(out_csv)
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
    ap.add_argument("--backend", default="ollama", choices=["ollama", "hf", "groq"],
                    help="judge + generator backend. 'groq' => GroqJudge + GroqGenerator (paid; "
                         "needs GROQ_API_KEY env). 'ollama' / 'hf' are local.")
    ap.add_argument("--judge-model", default="llama3.1:8b",
                    help="judge model id. For --backend groq use 'llama-3.3-70b-versatile'.")
    ap.add_argument("--gen-model", default="llama3.2:3b")
    ap.add_argument("--domains", nargs="+", default=["GenKnowledge", "CustomerSupport"])
    ap.add_argument("--n", type=int, default=10, help="examples per (config, domain)")
    ap.add_argument("--out", default=OUT_CSV,
                    help="output matrix CSV (use a distinct file for a different N so the "
                         "resumable skip-logic doesn't treat an N=10 run as already done)")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="generator answer-length cap (Ollama num_predict). Raise to test whether "
                         "terse answers are TRUNCATED vs the model being terse (targets completeness). "
                         "NOTE: changing this keeps the same config_id, so run such configs into a "
                         "SEPARATE --out file (config_id is built from stage TYPES, not params).")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="only run config files whose name contains one of these substrings "
                         "(e.g. --configs grounded_norerank). Default: all configs/*.yaml.")
    ap.add_argument("--workers", type=int, default=1, help="(reserved) per-example concurrency")
    ap.add_argument("--verdict", action="store_true", help="just print the matrix pivot and exit")
    args = ap.parse_args()

    if args.verdict:
        print_verdict(args.out)
        return

    import yaml
    import src  # noqa: F401
    from src.chunking import load_chunkers
    from src.embeddings import load_embedders
    from src.generation import load_generators
    from src.reranking import load_rerankers
    from src.summarization import load_summarizers
    from src.data_loader import load_domain
    from src.segmentation import OutputSegmenter, RegexSplitter
    from src.runner import build_grid, run_named_matrix
    load_chunkers(); load_embedders(); load_generators(); load_rerankers(); load_summarizers()

    judge = build_judge(args.backend, args.judge_model)
    if args.backend == "ollama":
        gen_override = {"type": "ollama", "model": args.gen_model, "max_new_tokens": args.max_new_tokens}
    elif args.backend == "groq":
        # Paid path — pair Groq judge with Groq generator. Same env var (GROQ_API_KEY).
        gen_override = {"type": "groq", "model": args.gen_model, "max_new_tokens": args.max_new_tokens}
    else:
        gen_override = {"type": "hf", "model": args.gen_model, "load_in_4bit": True,
                        "max_new_tokens": args.max_new_tokens}

    raw = {os.path.basename(p): yaml.safe_load(open(p)) for p in sorted(glob.glob("configs/*.yaml"))}
    if args.configs:                                 # filter to a subset by filename substring
        raw = {name: c for name, c in raw.items() if any(s in name for s in args.configs)}
        if not raw:
            print(f"no configs match {args.configs} — nothing to run.")
            return
    grid = build_grid(raw, args.domains, generator_override=gen_override)
    print(f"backend={args.backend}  judge={args.judge_model}  gen={args.gen_model}  "
          f"max_new_tokens={args.max_new_tokens}  domains={args.domains}  N={args.n}  "
          f"configs={list(raw)}  -> {len(grid)} (config x domain) runs\n")

    seg = OutputSegmenter(RegexSplitter())
    cache = {}
    def examples_for(cfg):
        if cfg.domain not in cache:
            cache[cfg.domain] = load_domain(cfg.domain, split="test", n=args.n, seed=42)
        return cache[cfg.domain]

    # POOLED corpus: a config with index.corpus_mode == "pooled" retrieves from the WHOLE
    # domain corpus (the full test split's documents), not just each question's own docs.
    # Loaded once per domain and cached. Per_example configs ignore this (corpus_for→None path).
    corpus_cache = {}
    def corpus_for(cfg):
        if cfg.index.params.get("corpus_mode") != "pooled":
            return None
        if cfg.domain not in corpus_cache:
            full = load_domain(cfg.domain, split="test", n=None)      # FULL split = full corpus
            corpus_cache[cfg.domain] = [d for ex in full for d in ex["documents"]]
        return corpus_cache[cfg.domain]

    run_named_matrix(grid, examples_for, args.out, segmenter=seg, judge=judge, corpus_for=corpus_for)
    print("\n--- matrix so far ---")
    print_verdict(args.out)


if __name__ == "__main__":
    main()
