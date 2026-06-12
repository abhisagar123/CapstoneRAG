# `configs/` — experiment definitions

**An experiment = a config.** Each `.yaml` here is one full RAG pipeline setup. The
runner (`src/runner.py`) loops these configs over the chosen RAGBench domains and writes
one results-matrix row per `(config, domain)` — that matrix is the headline Task-1
deliverable.

## The starter grid (4 configs)

A clean 2×2 that varies **one lever at a time**, holding everything else fixed:

|                      | reranker: `cross_encoder` | reranker: `none`          |
|----------------------|---------------------------|---------------------------|
| **prompt: grounded** | `grounded_rerank.yaml`    | `grounded_norerank.yaml`  |
| **prompt: minimal**  | `minimal_rerank.yaml`     | `minimal_norerank.yaml`   |

- **prompt** (`grounded` vs `minimal`) → the **Adherence** lever. `grounded` tells the
  model "answer ONLY from the context, else refuse"; `minimal` omits that. Comparing the
  two columns-of-prompt measures how much the instruction curbs hallucination.
- **reranker** (`cross_encoder` vs `none`) → the **Relevance** lever. The cross-encoder
  re-scores (query, chunk) pairs and keeps the best; `none` just takes the retriever's
  top-N. Comparing the two reranker columns measures whether reranking sharpens context.

Because only one component changes between any adjacent pair, a score difference is
attributable to that single component — the "change one thing at a time" rule.

## Added experiments (each = a one-variable swap off `grounded_norerank.yaml`)

Each compares head-to-head against `grounded_norerank.yaml` (or its rerank sibling) at the
same seed/N — only the named component changes (see `docs/EXPERIMENTS.md` Exp 4–7):

| config | changed component | lever tested |
|---|---|---|
| `grounded_complete_norerank.yaml` | prompt → `grounded_complete` | **Completeness** (a "be thorough" steer on top of grounding) |
| `grounded_bge_norerank.yaml`      | embedder → `bge-base-en-v1.5` | **Relevance/retrieval** (stronger embedder than MiniLM) |
| `grounded_complete_rerank.yaml`   | prompt + reranker ON | the prompt×reranker **synergy** (Exp 6B — best config so far) |
| `grounded_sides_norerank.yaml`    | repacker → `sides` | chunk ordering (Exp 6C — `reverse` won, `sides` parked) |
| `extractive_norerank.yaml`        | prompt → `extractive` | **Adherence** via "quote the source" (Exp 6D) |
| `grounded_pooled_norerank.yaml`   | index → `corpus_mode: pooled` | **real-RAG retrieval** (Exp 7 — search the whole domain, not just the question's docs) |

### ⚠️ Pooled vs per-example corpus mode (important)
- **`per_example`** (default for all the above except the pooled config): each question is
  answered against ONLY its own documents. This matches RAGBench's reference-score universe,
  so our scores are apples-to-apples vs the reference.
- **`pooled`**: the index is built ONCE from the WHOLE domain corpus and every question
  retrieves from it (real needle-in-haystack RAG). Our retrieval universe ≠ the reference's
  per-question docs, so relevance/utilization/completeness are valid as OUR-OWN scores but are
  **NOT apples-to-apples vs the reference** (adherence still is). All four are still computed;
  whether to run the reference comparison in pooled mode is a later call. Run pooled into a
  **separate `--out` file** (its scores aren't comparable to the per-example matrix). The runner
  loads the full domain corpus automatically for any config whose `corpus_mode` is `pooled`.

## Schema (every config has these keys)

```yaml
domain: Legal            # placeholder — the runner overrides this per domain
chunker:   { type: fixed, size: 512, overlap: 50 }     # size/overlap in WORDS
embedder:  { type: minilm }                            # all-MiniLM-L6-v2 (gated: load_embedders())
index:     { type: faiss, corpus_mode: per_example }   # per_example | pooled
retriever: { type: dense, k: 20 }                      # k = candidates to retrieve (call-time)
reranker:  { type: cross_encoder, top_n: 5 }           # or { type: none, top_n: 5 }
repacker:  { type: reverse }                           # forward | reverse | sides
prompt:    { type: grounded }                          # grounded | minimal
generator: { type: ollama, model: llama3.1:8b, max_new_tokens: 256 }   # local (Ollama); or type: hf for Colab
splitter:  { type: regex }                             # how our output is split for the judge
seed: 42
```

### Gotchas (verified against the code)

- **Never put `dim` in the `index`** — the pipeline injects the embedding dimension at
  runtime from the embedder. Putting it here would be ignored at best.
- **`k` and `top_n` are call-time args**, not constructor params. They live in the
  retriever/reranker `params` but the pipeline pulls them out before building the stage.
- **`reranker` and `repacker` are optional** — set to `null` to skip the stage entirely.
  All other stages are required.
- **Generator backends:** `ollama` (local Ollama server — Mac; non-Chinese models like
  `llama3.1:8b`, `mistral`, `gemma2:9b`) or `hf` (in-process transformers; Colab GPU).
  `echo` is the no-model test stub.
- **Heavy/lazy types.** `minilm` (embedder), `hf` (generator), `cross_encoder` (reranker),
  and the real judges (`hf`, `ollama`) only appear in the registry after the matching
  `load_embedders()` / `load_generators()` / `load_rerankers()` / `load_judges()` call. So
  **validate configs only after those loads.** Light types (`fixed`, `faiss`, `dense`,
  `none`, `reverse`, `grounded`, `minimal`, `regex`, `echo`, `fake`) are on bare `import src`.

## Adding a config

1. Copy an existing `.yaml`, change the one component you want to ablate.
2. Name it after the lever(s) it varies (e.g. `grounded_bge_norerank.yaml` for the embedder swap).
3. That's it — **no list to edit.** The runner **globs `configs/*.yaml`**, so a new file is
   picked up automatically. Already-done `(config_name, domain)` rows in the output CSV are
   skipped, so re-running only executes the new file's cells. (Keep one-variable changes so
   any score difference is attributable to that single component.)

## How they're run

Either the terminal runner or the notebook — both call the **same** shared helpers
(`src/runner.py::build_grid` + `run_named_matrix`), so they can't drift:

```bash
.venv-eda/bin/python scripts/run_matrix.py --n 50 --domains GenKnowledge CustomerSupport \
  --out results/ragbench_matrix_n50.csv
```

or `notebooks/04_run_matrix.ipynb` (Colab-portable). Both load every `configs/*.yaml`, build
the chosen judge (`llama3.1:8b`), and write one row per `(config, domain)`. The run is
**resumable + file-swap-safe** — already-done pairs are skipped on re-run. Then compare to the
reference scores: `.venv-eda/bin/python scripts/compare_reference.py --matrix <that csv> --plot`.
