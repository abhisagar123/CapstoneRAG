# CapstoneRAG — Real World RAG System (Group 16)

This is the Capstone project of the AI/ML course (IIIT-Hyderabad / TalentSprint PGCP).

A modular, domain-aware **Retrieval-Augmented Generation** system with a strong emphasis on
**self-implemented evaluation** — the graded core of the project. Two tasks:

1. **RAGBench** — a configurable RAG pipeline over 5 industry domains, evaluated per component
   with the **TRACe** metrics (uTilization, Relevance, Adherence, Completeness), compared against
   the dataset's reference scores.
2. **RGB** — generation-robustness evaluation of several open-source LLMs across 4 abilities
   (noise robustness, negative rejection, information integration, counterfactual robustness).

**Constraints:** **open-weight models only** for retrieval / embeddings / generation — the rule is
about model *licensing*, not where the model runs, so an open-weight model served by a hosted API
(e.g. **Groq**) is allowed; closed / proprietary models (GPT-4 / Claude / Gemini) are barred from
generation and permitted **only as an evaluation judge**. No Chinese-origin models anywhere. Metrics
are implemented from the papers, not pulled from a ready-made library.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Repo, EDA, schema confirmation | ✅ done |
| 1a | TRACe evaluator — **math half** (the 4 metrics from gold labels) | ✅ done & validated |
| 1b | TRACe evaluator — **LLM-judge half** (labels our own pipeline) | ✅ built; judge locked (`llama3.1:8b`, local); bigger judges (incl. Groq) used to *validate* that choice |
| 1c | Modular RAG pipeline (built brick by brick) + config + experiment runner | ✅ built |
| 2 | Strategy × domain results matrix + per-domain ablations + reference-score comparison | 🚧 N=50 matrix on 2 domains; 5 experiments done (see `docs/EXPERIMENTS.md`); more domains next |
| 3 | RGB robustness evaluation | ⬜ |
| 4 | Gradio demo + report | ⬜ |

> Open-source models run **locally via Ollama** (judge `llama3.1:8b`, generator `llama3.2:3b`) — except
> Chinese models. The `hf`/Colab path is retained as an alternative. See `docs/EXPERIMENTS.md` for results and
> `docs/PIPELINE_WALKTHROUGH.md` for an end-to-end trace.

**RAG pipeline bricks** (built one at a time, each tested): data loader ✅ · chunker + registry ✅ ·
embedder ✅ · index + retriever ✅ (FAISS dense) · reranker + repacker ✅ · prompt builder ✅ ·
generator ✅ (ollama local · groq hosted · hf for Colab · echo for tests) · output segmenter ✅ (regex + nltk) · pipeline wiring + experiment runner ✅.

**🎉 The RAG pipeline + evaluator are complete** — a config (YAML in `configs/`) assembles all components and
`run_matrix()` runs configs × domains into a resumable results CSV, and the TRACe **judge** (`llama3.1:8b` via the
exact RAGBench Appendix-7.4 prompt; OpenAI can drop in later via the same interface) produces the labels that fill
in real scores. Both a terminal runner (`scripts/run_matrix.py`) and a Colab notebook (`04_run_matrix.ipynb`) drive
this from the same shared helpers. **Done so far:** judge selected by agreement with reference scores; an N=50
matrix on 2 domains; the pipeline's scores compared to RAGBench's reference scores
(`scripts/compare_reference.py`); 5 documented experiments. **Remaining:** more domains → RGB (Task 2) → demo.

> **Compute (revised 10 Jun 2026):** open-source models run **locally via Ollama** on the dev Mac — **except
> Chinese models**. The in-process `transformers` path (`type: hf`) remains as a Colab alternative. Pure-Python
> stages (loader, chunker, indexing, repacking, prompting, evaluator) and all offline tests run locally with no GPU.

**Validated so far:** our TRACe metric implementation reproduces RAGBench's shipped reference
scores **exactly** (RMSE = 0; adherence 100%) across all 12 sub-datasets / 5 domains
(3,165 examples sampled). See `results/validation/evaluator_validation.csv`.

## Repository layout

```
src/
  data_loader.py   # load a RAGBench domain/split; domain → config map; sampling
  registry.py      # swap components by config string (@register + build factory)
  chunking/        # Chunker interface + strategies (fixed, noop, semantic, SAC, paragraph-group)
  embeddings/      # Embedder interface + sentence-transformers impl (lazy-loaded)
  indexing/        # Index interface + FAISS backend (exact search); per-example or pooled
  retrieval/       # Retriever interface + dense / BM25 / hybrid (weighted RRF) retrievers
  reranking/       # Reranker interface + cross-encoder (accurate re-score) + noop
  repacking/       # Repacker interface + chunk-ordering strategies (forward/reverse/sides)
  summarization/   # Summarizer interface + lexical / extractive-embedding (for SAC chunking) + noop
  prompting/       # PromptBuilder interface + grounded/minimal/grounded_complete/extractive variants
  generation/      # Generator interface + Ollama (local) + Groq (hosted API) + HuggingFace (Colab) + echo (tests)
  segmentation/    # SentenceSplitter (regex/nltk) + OutputSegmenter: pipeline→evaluator bridge
  query_classifier/ # optional Stage-0 retrieve-or-not gate (always_retrieve / heuristic) — off by default
  config.py        # YAML experiment config: load + validate against the registry
  pipeline.py      # Pipeline: assemble components from a config; index + answer
  runner.py        # ExperimentRunner: run configs × domains → resumable results CSV
  cost_tracker.py  # Groq $-accounting + hard-stop budget tripwire (paid backend only)
  telemetry.py     # optional per-example JSONL traces (best-effort; never crashes a run)
  judge/           # TRACe judge: Appendix-7.4 prompt → R/U labels (ollama local · groq hosted · hf Colab · fake for tests)
  evaluator/
    trace.py         # the 4 TRACe metrics (relevance, utilization, completeness, adherence)
    validate.py      # validates trace.py (math half) against RAGBench reference scores
    judge_validate.py # validates the judge half: its scores vs the reference (RMSE / accuracy)
    compare.py       # OUR matrix scores vs reference: per-cell signed gaps + bias/spread summary + charts
configs/             # experiment definitions (one YAML = one pipeline setup) + README
scripts/             # terminal tools: run_matrix / compare_reference / run_judge_validation / plot_pooled
tests/               # lightweight assert scripts (python tests/check_*.py)
notebooks/
  01_eda.ipynb                  # dataset schema confirmation + chunking-lever analysis
  02_evaluator_validation.ipynb # proves the math-half evaluator reproduces reference scores
  03_judge_validation.ipynb     # Colab: pick the judge by agreement with reference scores
  04_run_matrix.ipynb           # Colab: run configs × domains → strategy × domain matrix
results/                        # grouped by purpose (validation / per-example / pooled tracks)
  validation/                   # evaluator + judge validation report cards (vs reference scores)
    evaluator_validation.csv    #   math-half: reproduces reference scores (RMSE=0)
    judge_validation__*.csv     #   judge-half: each candidate judge's agreement with reference
  per_example/                  # per-example track (each Q answered against its own docs)
    ragbench_matrix*.csv        #   STEP 1 output: the strategy × domain results matrix (N=10, N=50, …)
    reference_comparison*.csv   #   STEP 2 output: per-cell ours vs reference + signed gap
    reference_comparison*_summary.csv  #   STEP 2 headline: per-metric bias + spread
    figures*/                   #   ours-vs-reference bar charts
  pooled/                       # pooled track (Q retrieves from the whole domain corpus — real RAG)
    ragbench_matrix_n50_pooled*.csv   # baseline + per-lever variants (one --out file each)
    figures/                          # configs-vs-each-other charts (scripts/plot_pooled.py)
  telemetry/                    # optional per-example JSONL traces + Groq cost tally (paid runs)
docs/
  PROJECT_PLAN.md  HLD.md  LLD.md   # plan + high-/low-level design
  PIPELINE_WALKTHROUGH.md           # one question traced through every stage, end to end
  EXPERIMENTS.md                    # per-example experiment log + TRACe findings & reasoning
  POOLED_EXPERIMENTS.md             # pooled-corpus experiment log (separate track)
papers/                         # the 3 core + 1 supplementary reference papers (PDFs)
```

## Quickstart (setup)

```bash
# Python 3.12 recommended (3.14 lags some wheels). CPU is enough for the evaluator.
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Sanity check 1 — the evaluator math reproduces RAGBench reference scores (no model, instant):
python -c "from src.evaluator.validate import validate_all; \
import pprint; pprint.pp(validate_all(n=50))"

# Sanity check 2 — the offline test suite (no model, no network):
python tests/check_compare.py        # the reference-comparison layer
python tests/check_judge.py          # the judge (parsing + backends)
python tests/check_generation.py     # the generators
```

> **Which Python?** The heavy stack (torch / sentence-transformers / faiss / datasets) lives in a
> 3.12 venv. In this repo that venv is `.venv-eda`, so the commands below call
> `.venv-eda/bin/python` directly. If you made your own venv above, just use `python` instead.

## Running an experiment — the TWO steps

This is the part most people miss: **a run and its reference-comparison are two separate steps.**
`run_matrix.py` produces only *raw* TRACe scores; the comparison-to-reference is a second script
you run afterwards. The runner does **not** call it for you (by design — the comparison is free and
instant, so you can re-slice it many ways without re-running the expensive pipeline).

```
 STEP 1 — run the pipeline (needs a model)         STEP 2 — compare to reference (no model, instant)
 ┌───────────────────────────────┐                 ┌─────────────────────────────────────┐
 │ scripts/run_matrix.py          │  writes         │ scripts/compare_reference.py         │  writes
 │ pipeline → judge → TRACe        │ ───────────────▶│ averages RAGBench's shipped ref      │ ───────▶ reference_comparison*.csv
 │                                │ ragbench_       │ score fields over the SAME examples, │          reference_comparison*_summary.csv
 │                                │ matrix*.csv     │ then ours − reference per cell       │
 └───────────────────────────────┘ (RAW scores)    └─────────────────────────────────────┘ (the ANALYSIS)
```

**Step 1 — run the matrix** (configs × domains → one row per pair, resumable):

```bash
# Local Ollama path (default). First: open the Ollama app, then pull both models:
#   ollama pull llama3.1:8b      # the judge (scores answers)   — the LOCKED scoring judge
#   ollama pull llama3.2:3b      # the generator (writes answers)
.venv-eda/bin/python scripts/run_matrix.py --domains GenKnowledge CustomerSupport --n 50 \
  --out results/per_example/ragbench_matrix_n50.csv

.venv-eda/bin/python scripts/run_matrix.py --verdict --out results/per_example/ragbench_matrix_n50.csv  # just print the pivot
```

Useful flags: `--configs grounded_norerank` (run only matching config files), `--n` (examples per
cell), `--max-new-tokens` (generator answer-length cap). **Use a distinct `--out` per N** — the
resume-logic skips already-done `(config, domain)` rows, so an N=10 file would block an N=50 re-run.

**Step 2 — compare to the reference scores:**

```bash
.venv-eda/bin/python scripts/compare_reference.py \
  --matrix results/per_example/ragbench_matrix_n50.csv \
  --out results/per_example/reference_comparison_n50.csv --plot
```

This writes two files and prints both: the **per-cell** comparison (`reference_comparison*.csv`:
`ours | reference | gap` for every config×domain) and a **per-metric summary**
(`*_summary.csv`: bias + spread per metric — see methodology below). `--plot` also writes
ours-vs-reference bar charts. It needs no model — it just averages reference fields the dataset
already ships (we proved in notebook 02 these equal our own math at RMSE 0).

### How to read the comparison (gap = ours − reference)

`gap` is **signed on purpose** — the direction *is* the finding, so we never collapse it to a
magnitude:

- **relevance** — retriever signal. A good retriever filters junk, so ours **above** reference is
  expected/good. *Caveat:* the 8B judge over-marks relevance ~+0.09, so a small positive gap may be
  judge bias, not pure gain (the bias is ~constant across configs, so config-vs-config stays valid).
- **adherence** — prompt + generator faithfulness. Reference is a real system's rate; **match-or-beat**
  is the goal. Below → grounding prompt / bigger generator. *This is the trustworthy faithfulness signal.*
- **utilization** — mixes retrieval AND generation; triangulate with the other two.
- **completeness** — a derived ratio, noisiest; read last.

### Why the summary is bias + spread, not RMSE

The per-metric summary reports **bias** (signed mean gap — which way and how far we sit vs the
benchmark) and **spread** (std of the gap across configs — how consistent that offset is). It does
**not** report RMSE, on purpose:

1. **RMSE drops the sign.** It measures distance-from-a-bullseye, but here being *above* reference on
   relevance is a *win* — RMSE would score that win as "error".
2. **RMSE merges two facts into one.** Since `RMSE² = bias² + spread²`, a single RMSE can't tell
   "consistent +0.10 offset" (relevance) from "dead-on average but config-dependent" (utilization,
   bias≈0 yet spread≈0.07). Bias+spread keeps them apart.
3. **It's the wrong RMSE for this comparison.** RMSE *is* the right tool for **judge validation**
   (`run_judge_validation.py`), where ours and the reference score the *same inputs* so a per-example
   error is real. Comparison-to-reference scores *different* inputs (our retrieved context + our
   answer vs gold), so there is no per-example "error" to RMSE — only a system-vs-benchmark gap.

## Other entry points

- **Judge validation** — is a candidate judge accurate vs the GPT-4 gold labels? (This is the
  comparison that *does* use RMSE.)
  ```bash
  .venv-eda/bin/python scripts/run_judge_validation.py                 # local Ollama (the locked 8B)
  .venv-eda/bin/python scripts/run_judge_validation.py --backend groq  # a bigger hosted judge (70B), to re-validate the choice
  .venv-eda/bin/python scripts/run_judge_validation.py --verdict       # print the merged summary
  ```
- **Pooled-track plots** — compare configs against each other (no reference) for pooled runs:
  ```bash
  .venv-eda/bin/python scripts/plot_pooled.py
  ```

## Model backends (all open-weight; no Chinese-origin models)

Set via the config's `generator`/`judge` type, or `--backend` on the scripts:

| backend | where it runs | how to enable | notes |
|---|---|---|---|
| `ollama` | local Ollama server | `ollama pull <model>` | default; non-Chinese tags (`llama3.1:8b`, `mistral`, `gemma2:9b`) |
| `groq` | hosted API | `export GROQ_API_KEY=...` | bigger open-weight models (e.g. `llama-3.3-70b-versatile`) at high speed; **paid** |
| `hf` | in-process transformers | Colab GPU | the original Colab path |
| `echo` / `fake` | none | — | no-model stubs for tests |

> **Groq cost cap.** The `groq` backend is metered. `src/cost_tracker.py` tallies every paid call to
> `results/telemetry/legal/_cost_running.json` and **hard-aborts** the run if cumulative spend crosses
> `HARD_CAP_USD` (currently $4.50). Re-verify Groq's prices in `cost_tracker.PRICES` before a big run,
> and override the state-file location with `CAPSTONERAG_COST_FILE` if you want a per-experiment tally.
> `GROQ_API_KEY` is read from the environment and **must never be committed** (the repo is public).

## Team

Group 16 — AI/ML PGCP capstone (IIIT-Hyderabad / TalentSprint).
