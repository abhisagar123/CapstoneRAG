# `configs/` — experiment definitions

**An experiment = a config.** Each `.yaml` here is one full RAG pipeline setup. The
runner (`src/runner.py::run_matrix`) loops these configs over the 5 RAGBench domains
and writes one results-matrix row per `(config, domain)` — that matrix is the headline
Task-1 deliverable.

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
generator: { type: hf, model: Qwen/Qwen2.5-3B-Instruct, load_in_4bit: false, max_new_tokens: 256 }
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
- **Heavy types register lazily.** `minilm` (embedder), `hf` (generator), `cross_encoder`
  (reranker) only appear in the registry after the matching `load_embedders()` /
  `load_generators()` / `load_rerankers()` call. So **validate configs only after those
  loads** (on Colab). Light types (`fixed`, `faiss`, `dense`, `none`, `reverse`,
  `grounded`, `minimal`, `regex`, `echo`, `fake`) are available on bare `import src`.

## Adding a config

1. Copy an existing `.yaml`, change the one component you want to ablate.
2. Name it after the lever(s) it varies (e.g. `bge_rerank.yaml` if you swap the embedder).
3. Add it to the runner notebook's config list. The matrix grows automatically.

## How they're run

See `notebooks/04_run_matrix.ipynb`: it loads these files, builds the chosen judge
(from notebook 03's validation), and calls `run_matrix(...)` → `results/ragbench_matrix.csv`.
The run is **resumable** — already-done `(config, domain)` pairs are skipped on re-run.
