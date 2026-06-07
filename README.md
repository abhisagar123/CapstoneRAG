# CapstoneRAG — Real World RAG System (Group 16)

This is the Capstone project of the AI/ML course (IIIT-Hyderabad / TalentSprint PGCP).

A modular, domain-aware **Retrieval-Augmented Generation** system with a strong emphasis on
**self-implemented evaluation** — the graded core of the project. Two tasks:

1. **RAGBench** — a configurable RAG pipeline over 5 industry domains, evaluated per component
   with the **TRACe** metrics (uTilization, Relevance, Adherence, Completeness), compared against
   the dataset's reference scores.
2. **RGB** — generation-robustness evaluation of several open-source LLMs across 4 abilities
   (noise robustness, negative rejection, information integration, counterfactual robustness).

**Constraints:** open-source models only for retrieval/embeddings/generation; a hosted LLM is
allowed **only as an evaluation judge**; metrics are implemented from the papers, not pulled from a
ready-made library.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Repo, EDA, schema confirmation | ✅ done |
| 1a | TRACe evaluator — **math half** (the 4 metrics from gold labels) | ✅ done & validated |
| 1b | TRACe evaluator — **LLM-judge half** (produces labels for our own pipeline) | ✅ built (OSS judge on Colab; validate next) |
| 1c–2 | Modular RAG pipeline (built brick by brick) + per-domain ablations | 🚧 in progress |
| 3 | RGB robustness evaluation | ⬜ |
| 4 | Gradio demo + report | ⬜ |

**RAG pipeline bricks** (built one at a time, each tested): data loader ✅ · chunker + registry ✅ ·
embedder ✅ · index + retriever ✅ (FAISS dense) · reranker + repacker ✅ · prompt builder ✅ ·
generator ✅ (hf for Colab + echo for tests) · output segmenter ✅ (regex + nltk) · pipeline wiring + experiment runner ✅.

**🎉 The RAG pipeline + evaluator are complete** — a config (YAML) assembles all components and `run_matrix()`
runs configs × domains into a resumable results CSV, and the TRACe **judge** (strong OSS model on Colab via the
exact RAGBench Appendix-7.4 prompt; OpenAI can drop in later) produces the labels that fill in real scores.
Remaining: validate the judge vs reference scores → run the RAGBench results matrix → RGB (Task 2) → demo.

> **Note:** per a runtime constraint, all model inference (embedder, reranker, generator) runs on **Google
> Colab**; the pure-Python stages (loader, chunker, indexing, repacking, prompting, evaluator) and their offline
> tests run locally. Model-dependent checks run on Colab via a runner notebook.

**Validated so far:** our TRACe metric implementation reproduces RAGBench's shipped reference
scores **exactly** (RMSE = 0; adherence 100%) across all 12 sub-datasets / 5 domains
(3,165 examples sampled). See `results/evaluator_validation.csv`.

## Repository layout

```
src/
  data_loader.py   # load a RAGBench domain/split; domain → config map; sampling
  registry.py      # swap components by config string (@register + build factory)
  chunking/        # Chunker interface + strategies (fixed, noop) — one file each
  embeddings/      # Embedder interface + sentence-transformers impl (lazy-loaded)
  indexing/        # Index interface + FAISS backend (exact search); per-example or pooled
  retrieval/       # Retriever interface + dense retriever: question → nearest chunks
  reranking/       # Reranker interface + cross-encoder (accurate re-score) + noop
  repacking/       # Repacker interface + chunk-ordering strategies (forward/reverse/sides)
  prompting/       # PromptBuilder interface + grounded/minimal prompt variants
  generation/      # Generator interface + HuggingFace LLM (Colab) + echo (tests)
  segmentation/    # SentenceSplitter (regex/nltk) + OutputSegmenter: pipeline→evaluator bridge
  config.py        # YAML experiment config: load + validate against the registry
  pipeline.py      # Pipeline: assemble components from a config; index + answer
  runner.py        # ExperimentRunner: run configs × domains → resumable results CSV
  judge/           # TRACe judge: Appendix-7.4 prompt → R/U labels (OSS on Colab + fake for tests)
  evaluator/
    trace.py       # the 4 TRACe metrics (relevance, utilization, completeness, adherence)
    validate.py    # validates trace.py against RAGBench reference scores (RMSE / accuracy)
tests/             # lightweight assert scripts (python tests/check_*.py)
notebooks/
  01_eda.ipynb                  # dataset schema confirmation + chunking-lever analysis
  02_evaluator_validation.ipynb # proves the math-half evaluator reproduces reference scores
results/
  evaluator_validation.csv      # the validation report card
docs/
  PROJECT_PLAN.md  HLD.md  LLD.md   # plan + high-/low-level design
papers/                         # the 3 core + 1 supplementary reference papers (PDFs)
```

## Quickstart

```bash
# Python 3.12 recommended (3.14 lags some wheels). CPU is enough for the evaluator.
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validate the evaluator against RAGBench reference scores:
python -c "from src.evaluator.validate import validate_all; \
import pprint; pprint.pp(validate_all(n=50))"
```

Notebooks run on Google Colab (the project's primary compute); the evaluator math-half also runs
locally on CPU with no API key.

## Team

Group 16 — AI/ML PGCP capstone (IIIT-Hyderabad / TalentSprint).
