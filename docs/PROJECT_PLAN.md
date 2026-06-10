# Real World RAG System — Execution Plan (Group 16)

A practical, followable plan for the capstone. Timeline ~9–10 weeks (start ~2 Jun, present early Aug),
3 members, weekends-focused, open-source models only (hosted LLM allowed **only as evaluation judge**),
Google Colab (free / Pro) for compute.

**Golden rule from the mentor:** *evaluation is the core of this project.* Build the evaluator early,
prove it against RAGBench's reference scores, then optimise the pipeline. Do RAGBench first, then RGB,
then bonus.

---

## 0. Guiding principles

- **Build the evaluator before chasing scores.** Nothing else can be measured until TRACe works.
- **Subset first, scale later.** Iterate on a few hundred examples; run the full set only when stable.
- **One common pipeline, per-domain configs.** Don't build five apps — build one configurable pipeline.
- **Change one component at a time.** That's the only way the results matrix means anything.
- **Discipline on splits.** Tune on validation, report final numbers on test. Never tune on test.
- **Commit often, small notebooks/modules.** Keep everything reproducible in the GitHub repo.

---

## 1. Recommended tech stack (decided defaults, so you don't stall on choices)

| Component | Default choice | Notes / alternatives to compare |
|---|---|---|
| Data | HuggingFace `datasets` → `rungalileo/ragbench` | 12 sub-datasets across 5 domains |
| Chunking | fixed-size (512 tok, 50 overlap) baseline | compare sliding-window, semantic |
| Embeddings | `BAAI/bge-base-en-v1.5` | compare `intfloat/e5-base-v2`, `all-MiniLM-L6-v2`; `jina-v2` for long docs |
| Vector DB | ChromaDB (dev) | FAISS for speed; try one alternative for the writeup |
| Sparse retrieval | `rank-bm25` | for hybrid |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast) | compare `castorini/monot5-base-msmarco` |
| Generator (open-source) | `Llama-3.1-8B-Instruct` or `Mistral-7B-Instruct` (local via Ollama on Mac, or 4-bit on Colab) | `Llama-3.2-3B` for fast dev. **Non-Chinese models only** (no Qwen) |
| Evaluation judge | strong non-Chinese OSS model (Appendix-7.4 prompt) — local Ollama or Colab | OpenAI can drop in later via the same interface |
| Demo | Gradio on Hugging Face Spaces | FastAPI optional |
| Orchestration | plain Python modules | LangChain/LlamaIndex optional, only if they save time |

---

## 2. Repository structure

```
rag-capstone/
├── README.md
├── requirements.txt
├── data/                  # cached/loaded subsets (gitignored if large)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_evaluator_validation.ipynb
│   ├── 03_baseline_rag.ipynb
│   ├── 04_experiments.ipynb
│   └── 05_rgb.ipynb
├── src/
│   ├── data_loader.py     # load ragbench, split handling
│   ├── chunking.py
│   ├── embeddings.py
│   ├── indexer.py         # vector DB + BM25
│   ├── retriever.py       # dense / sparse / hybrid + rerank
│   ├── generator.py       # open-source LLM wrapper
│   ├── pipeline.py        # end-to-end orchestration + per-domain config
│   ├── evaluator/
│   │   ├── trace.py       # the 4 TRACe metrics (CORE)
│   │   └── rgb.py         # 4 RGB ability metrics
│   └── config.py          # per-domain settings
├── results/               # CSVs / the strategy×domain matrix
└── app/                   # Gradio demo
```

---

## PHASE 0 — Foundations (Week 1)

**Goal:** repo, environment, data understanding.

- [ ] Create GitHub repo + shared Colab; agree on folder structure & branching.
- [ ] `requirements.txt`: `datasets transformers sentence-transformers rank-bm25 chromadb faiss-cpu bitsandbytes accelerate openai gradio pandas matplotlib`.
- [ ] Load RAGBench via `datasets.load_dataset("rungalileo/ragbench", <subset>)`; confirm the **domain → sub-dataset mapping** for real.
- [ ] **EDA (`01_eda.ipynb`)**: per domain — #examples, doc count/length distributions, fields present, the reference score fields (`relevance/utilization/adherence/completeness`), question & document types. Save a short summary.
- [ ] Read all 3 papers as a team; each person writes a ~½-page summary (this becomes the literature review / 3-paper summary deliverable).

**Deliverable:** working repo + EDA notebook + paper summaries.

---

## PHASE 1 — TRACe Evaluator + Baseline (Weeks 1–2)  ⭐ most important

**Goal:** a trustworthy evaluator, then a first end-to-end RAG number.

### 1a. Implement the TRACe evaluator (`src/evaluator/trace.py`)
- [ ] Read the metric definitions in the RAGBench paper carefully.
- [ ] Implement, given (question, retrieved context, generated answer):
  - [ ] **Context Relevance** — fraction of retrieved context relevant to the query.
  - [ ] **Context Utilization** — fraction of relevant context actually used in the answer.
  - [ ] **Adherence** — whether all answer claims are grounded in the context (hallucination check).
  - [ ] **Completeness** — whether the answer covers all relevant info in the context.
- [ ] Use `gpt-4o-mini` as the judge to label sentence-level relevance/support, then compute the ratios.

### 1b. Validate the evaluator (`02_evaluator_validation.ipynb`)
- [ ] Run your evaluator on a few hundred labelled RAGBench examples.
- [ ] Compare your scores vs the dataset's **reference scores** (correlation + error).
- [ ] Tune the judge prompts until agreement is reasonable. **Don't proceed until this holds.**

### 1c. Baseline RAG on ONE domain (`03_baseline_rag.ipynb`)
- [ ] Pick Customer Support (e.g. `emanual`/`techqa`). Build: fixed chunking → bge embeddings → Chroma → dense top-k → open-source LLM generation with a grounding prompt.
- [ ] Score all 4 metrics; compare to reference. This is your CP-1 / Review-1 baseline.

**Deliverable:** validated evaluator + baseline scores on one domain.

---

## PHASE 2 — Multi-domain + Component Experiments (Weeks 3–5)  →  Review #1

**Goal:** all five domains + the strategy×domain results matrix (the headline RAGBench output).

- [ ] Generalise the pipeline to load any domain via `config.py` (one pipeline, per-domain settings).
- [ ] Build the **pooled per-domain index** (retrieve from the whole domain corpus, not just one example's docs).
- [ ] Run the baseline across **all 5 domains**; record scores.
- [ ] Component ablations (change ONE at a time, per domain):
  - [ ] Chunking: fixed vs sliding-window vs semantic.
  - [ ] Embeddings: 2–3 models from MTEB.
  - [ ] Retrieval: dense vs BM25 vs hybrid (RRF).
  - [ ] Reranking: with vs without cross-encoder.
  - [ ] Context ordering: forward vs reverse.
  - [ ] (Optional) summarization of context before generation.
- [ ] Assemble the **results matrix** (`results/`): best config per domain + short reasoning for each.

**Deliverable (Review #1):** multi-domain results, ablation tables, and analysis of what worked where & why.

> Scope control: if time is tight, run the full ablation grid on 1–2 domains and a reduced grid on the rest.

---

## PHASE 3 — RGB Robustness Evaluation (Weeks 6–7)  →  Review #2

**Goal:** evaluate open-source LLMs on the 4 RGB abilities.

- [ ] Get RGB data (English split) from `github.com/chen700564/RGB`.
- [ ] Implement RGB metrics (`src/evaluator/rgb.py`):
  - [ ] Noise Robustness — accuracy (exact match).
  - [ ] Information Integration — accuracy (exact match).
  - [ ] Negative Rejection — rejection rate (model emits the instructed refusal).
  - [ ] Counterfactual Robustness — error detection rate & correction rate.
- [ ] Run **4–5 open-source LLMs** (non-Chinese: e.g. Llama-3.1-8B, Mistral-7B, Gemma-2-9B, Phi-3-mini) — only ~1000 questions, feasible locally via Ollama or in 4-bit on Colab.
- [ ] Tabulate results per ability per model + short analysis.

**Deliverable (Review #2):** RGB scores across models + analysis.

---

## PHASE 4 — Deployment + Report (Weeks 8–9)

**Goal:** demo + final deliverables.

- [ ] Build a **Gradio** app: domain dropdown → query box → answer + retrieved source passages.
- [ ] Wire it to the best per-domain config; deploy to **Hugging Face Spaces**.
- [ ] Clean up the GitHub repo: README, `requirements.txt`, reproducible scripts.
- [ ] Final **technical report** (covers everything: problem, papers, dataset/EDA, methodology, results matrix, RGB, challenges, deployment) + **presentation**.
- [ ] Rehearse the live demo.

**Deliverable:** deployed demo + report + slides + repo.

---

## Bonus (only if ahead of schedule)
- Domain-specific / novel chunking (e.g. biomedical) and measure gains.
- Improve RGB abilities via prompting/context formatting **without** changing the base LLM.
- Enterprise RAGBench: at least an extended analysis/feasibility writeup.

---

## Suggested work split (3 members, shared infra first)
- **Weeks 1–2 (together):** repo, data loader, EDA, and the TRACe evaluator — build these jointly since everything depends on them.
- **Weeks 3–5 (split by domain):** e.g. member 1 → Biomedical + General Knowledge; member 2 → Legal + Customer Support; member 3 → Finance + owns the shared pipeline/indexer. Each runs ablations on their domains into a shared results sheet.
- **Weeks 6–7:** one person leads RGB while the other two finish RAGBench analysis & the report draft.
- **Weeks 8–9 (together):** deployment, report, slides, demo rehearsal.

(Adjust freely — keep a shared sheet of who-owns-what so individual contributions are visible for grading.)

---

## Definition of done (deliverables checklist)
- [ ] GitHub repo (reproducible).
- [ ] Self-implemented TRACe evaluator, validated against reference scores.
- [ ] RAGBench results matrix across all 5 domains + analysis.
- [ ] RGB evaluation across 4–5 open-source LLMs + analysis.
- [ ] Deployed Gradio demo (domain selector).
- [ ] Technical report + presentation + 3-paper summaries.

---

## Practical tips / risks
- **Compute:** prefer local Ollama on the Mac (M3 Max/96 GB — no Colab limits) with a small non-Chinese model (e.g. `Llama-3.2-3B`) for fast iteration, scaling to 8B+ for final runs; Colab (4-bit) is the alternative. Save indexes/embeddings to Drive when on Colab.
- **Judge cost:** `gpt-4o-mini` is cheap, but cache judge outputs and evaluate on subsets to control spend.
- **Don't over-engineer retrieval setup:** start with per-example retrieval to validate the evaluator quickly, then move to the pooled per-domain index for the real results.
- **Reproducibility:** fix random seeds; log every config + its scores into `results/` as CSV.
```
