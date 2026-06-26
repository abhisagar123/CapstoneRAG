# Legal-domain RAG configuration search — experiment plan

Date: 2026-06-21 (revised after research-informed Phase A pivot)
Generator: **Groq `llama-3.3-70b-versatile`** (paid)
Scoring judge: **Groq `llama-3.3-70b-versatile`** (paid; user-requested upgrade from the local 8B doctrine — see § 1a)
Dataset: `rungalileo/ragbench` → `cuad` (Legal). Already cached at `~/.cache/huggingface/datasets/rungalileo___ragbench/cuad/`.
Goal: identify the configuration that maximizes the four TRACe metrics (Context Relevance, Context Utilization, Completeness, Adherence) on Legal.
Hard budget cap: **$5 total Groq spend.** (Plan estimates **~$4.10** under the revised slim-4 Phase A; details below.)

## 0a. Why the original 8-config Phase A was scrapped (2026-06-21)

A deep-research workflow (see `docs/LEGAL_RESEARCH_FINDINGS.md`) surveyed RAGBench, LegalBench-RAG, and the contemporary legal-RAG literature. Findings substantially contradicted the original 8-config blind sweep:

- **Chunk size**: published optimum is **500 chars no overlap**; A5 (1024) and A6 (256) are both contraindicated.
- **Hybrid retrieval**: dense-only **beats** BM25+dense weighted on LegalBench-RAG; A8 was the wrong default-on.
- **Cross-encoder rerank**: Cohere rerank *degraded* CUAD retrieval (P@1 9.27 → 1.97); A3/A4/A7/A8 all had it default-on without ablating against off.
- **Biggest missing lever**: **Summary-Augmented Chunking (SAC)** — prepending a 150-char doc summary to each 500-char chunk — halved Document-level Retrieval Mismatch (~35% → 19.29%) and was the single highest-leverage finding in the literature. The framework didn't implement it.

The revised Phase A is **4 research-anchored configs (R1-R4)** instead of 8 enumerated configs. Code added this session: `chunker.sac`, `embedder.gte_large`, `reranker.bge_large`. Higher N per config (50, not 15-25), same total cost.

---

## 1a. Doctrine deviation: Groq 70B as the SCORING judge

Framework default doctrine pins the scoring judge to local `llama3.1:8b` via Ollama (free, deterministic, validated). User requested that the judge be **configurable** and run on Groq `llama-3.3-70b-versatile` for this study, with the explicit understanding that:
- The Groq judge runs the SAME verbatim Appendix-7.4 prompt as the local judge (`src/judge/base.py`), so the math half is unchanged.
- Per-judge-call cost is ~$0.00148 (~1,850 in + ~500 out tokens), roughly 2.4× the generator's per-call cost. § 2 budgets both legs explicitly.
- The CLI flag is now `--backend groq --judge-model llama-3.3-70b-versatile` (patched in `scripts/run_matrix.py`). `--backend ollama --judge-model llama3.1:8b` still works — config-file YAML can pin either.
- Cross-judge agreement (Groq-70B vs local-8B) is recorded as a side artifact on a 30-example subsample in Phase A (~$0.04 extra), so the report can quote the divergence numerically.

---

## 0. Why a plan first, not "just sweep everything"

CUAD's structural shape (1 doc per question, ~25 K chars median per doc, max 250 K) is unlike the other RAGBench domains, so we cannot copy the GenKnowledge/CustomerSupport recipe. The experimentation must (a) respect that shape, (b) move ONE lever at a time so each delta is attributable, and (c) honour the $5 ceiling using a token-cost model worked out below.

---

## 1. How the framework reaches TRACe scores (1-page summary)

End-to-end path is the **runner.run_experiment()** flow:

```
load_domain("Legal", split="test", n=N)         # CUAD test split, seeded sample
  │
  ▼
for each example:
  pipeline.reset_index()                        # per_example mode
  pipeline.index_documents(ex["documents"])     # chunk → embed → faiss
  out = pipeline.answer(ex["question"])
      ├── (optional) query_classifier.classify  # PDF Stage 1 — off by default
      ├── retriever.retrieve(query, k)          # dense / hybrid / bm25
      ├── reranker.rerank(query, cands, top_n)  # optional, cross-encoder
      ├── repacker.pack(chunks)                 # forward / reverse / sides
      ├── summarizer.compress(query, chunks)    # PDF Stage 5 — off by default
      ├── prompt_builder.build(query, chunks)
      └── generator.generate(prompt)            # Groq 70B
  │
  ▼
keyed = OutputSegmenter.segment(out["sources"], out["answer"])
label = judge.label(question, keyed)            # Appendix-7.4 prompt → JSON labels
scores = scores_from_label(keyed, label)
     {relevance, utilization, completeness, adherence}
```

Important contract:
- **`relevance`** = |R| / T where R is the set of doc sentences the judge marks relevant, T the total doc sentences in the **retrieved** keyed input (not the full corpus). Diagnoses the **retriever**.
- **`utilization`** = |U| / T where U is the doc sentences the judge says were USED in the answer. Diagnoses the **generator+prompt**.
- **`completeness`** = |R ∩ U| / |R|. "Of the useful material, how much did the answer cover?"
- **`adherence`** = `len(unsupported_response_sentence_keys) == 0`. Boolean per example; aggregated as a True-rate.

Math half (`src/evaluator/trace.py`) is already validated against RAGBench's shipped reference (RMSE ≈ 0 on all 12 configs).

---

## 2. Budget math — both Groq legs accounted for (Option 2: measured-cost re-baseline)

Groq `llama-3.3-70b-versatile` published rates (re-verified 2026-06-21 — see § 9):
- **$0.59 / 1M input tokens**, **$0.79 / 1M output tokens**.

### Pre-flight smoke (2026-06-21, `legal_rerank_only` × N=3 × Legal)

| Leg | In tokens (avg) | Out tokens (avg) | $ / call (measured) |
|---|---|---|---|
| Generator | 3,197 | 23 | **$0.00190** |
| Judge | 4,388 | 269 | **$0.00280** |
| **Per example (combined, rerank-on)** | 7,585 | 292 | **$0.00470** |

The smoke's per-example cost is **2.25× the original paper-estimate** ($0.00470 vs $0.00209). The culprit is CUAD's huge per-question documents — 5 reranked chunks at chunker.size=512 land ~3.2 K input tokens to the generator, not the ~840 a generic RAG corpus would. The original $2.16 plan assumed shorter chunks and a smaller judge prompt; the 2× buffer covered that but only barely, and on the no-rerank legs (A1/A2) cost EXPLODES to ~3× (top-20 chunks not top-5).

### Measured per-config-family cost

| Config family | Gen in_tok | Judge in_tok | $/example |
|---|---|---|---|
| **Rerank top_n=5, chunks=512** (A3, A4, A7, A8) | ~3,200 | ~4,400 | **$0.0047** |
| **No rerank, top-20 chunks** (A1, A2) | ~12,800 | ~11,700 | **$0.0147** |
| **Rerank top_n=5, chunks=1024** (A5) | ~6,400 | ~7,400 | **$0.0080** |
| **Rerank top_n=5, chunks=256** (A6) | ~1,600 | ~2,200 | **$0.0024** |

### Revised schedule — slim Phase A (R1-R4) at N=50

R1/R2/R3 have NO reranker → top-20 chunks per generator call → $0.0147/example. R4 has reranker top-5 → $0.0047/example. SAC adds 150 chars of summary per chunk so R1/R3/R4 are ~5% higher than the bare-fixed baseline; absorbed in the rounding.

Note: R1/R2/R3 use **char-based** 500-char chunks (SAC) or **80-word ≈ 500-char** chunks (R2's fixed). Either way the chunks are ~30% smaller than the smoke's 512-word chunks, so the per-call gen cost is lower than the smoke's $0.0147 measurement — but we budget at the measured smoke rate (conservative).

| Phase | Configs | N | Total examples | $/example | $ |
|---|---|---|---|---|---|
| A — R1 sac_dense (no rerank) | 1 | **50** | 50 | $0.0147 | **0.74** |
| A — R2 fixed500_dense (no rerank) | 1 | **50** | 50 | $0.0147 | **0.74** |
| A — R3 sac_hybrid (no rerank) | 1 | **50** | 50 | $0.0147 | **0.74** |
| A — R4 sac_rerank (top-5) | 1 | **50** | 50 | $0.0047 | **0.24** |
| **Phase A subtotal** | **4** | 50 | **200** | mixed | **$2.46** |
| B (refine; top-2 winners + 6 perturbations) | 8 | **40** | 320 | ≤$0.0147 | **≤$0.95** (4 rerank @ 0.0047 + 4 no-rerank @ 0.0147) |
| C (lock-in winners) | 3 | **70** | 210 | ≤$0.0147 | **≤$0.62** (worst case all no-rerank) |
| Cross-judge (Groq vs local 8B on R1 × 30) | 1 | 30 | 30 | mixed | **$0.20** |
| **TOTAL** | — | — | **760** | — | **~$4.23** |

Margin under $4.50 hard cap: **$0.27**. Inside $5 ceiling with $0.77 headroom. This is **more** Phase A coverage per config (N=50 vs N=15-25 in the scrapped plan) at the cost of fewer configs (4 vs 8) — research narrowed the search space, so concentrate spend on signal.

Confidence interval at the chosen N (Wilson, 95%):
- **N=50** (Phase A R1-R4) → ±0.07 on a fraction metric. Distinguishes Δ≈0.10 cleanly at α=0.05. Sufficient for ranking and decent for reporting.
- **N=40** (Phase B) → ±0.08. Distinguishes Δ≈0.08 between configs.
- **N=70** (Phase C) → ±0.06. Holds the Δ≥0.06 sanity-floor at ~1σ headroom.

**Hard stop tripwires:**
1. `src/cost_tracker.py` updates `results/telemetry/legal/_cost_running.json` after every Groq call (both legs). **Run hard-aborts at $4.50** (margin under the $5 cap).
2. After Phase A, if combined spend > $0.65 we re-estimate before Phase B.
3. Generator `max_new_tokens` capped at 256 (framework default). Never raise above 384.
4. Judge response is bounded by Groq's response_format=json_object (no runaway tokens).
5. Every Groq response's `usage` block (Groq returns it on the OpenAI-compatible endpoint) is tallied in real time — we know cost-to-date to within one call.

### What if pricing has changed?
The cost tracker reads `usage` from each live response, NOT a hardcoded rate. The $/token constants live in **one** place (`src/cost_tracker.py::PRICES`). If Groq has raised the 70B rate ≥ 2× since the framework was written, Phase B's worst-case buffer would push past $5 — the cost tracker would abort the run, and we'd revert to **local 8B judge** for B+C while keeping Groq judge for A + the cross-judge check (cost falls to ~$0.95).

---

## 3. What to optimize — the four levers + their hypotheses

CUAD's particulars predict which levers matter:

| Lever | Choices | Research-informed prior (2026-06-21) |
|---|---|---|
| **chunker** | sac(500/sum=150), fixed(500), pgc, semantic | **Published optimum: SAC 500-char no-overlap + 150-char doc-summary prepended** (Reuter 2025). On CUAD, 500-char beat RCTS at P@1 (Pipitone 2024). 256/1024 contraindicated. |
| **embedder** | gte_large, minilm, mxbai | gte-large best open-source on LegalBench-RAG (Reuter Appendix C). Legal-BERT WORST (no contrastive training). Phase A pins gte-large. |
| **retriever** | dense, hybrid, bm25 | **Dense BEATS hybrid** on text-level P/R (Reuter Table 2: hybrid degrades precision 11.0→9.8 as BM25 weight rises). Direction prior: dense-only wins; R3 tests this. |
| **reranker** | none / bge_large / cross_encoder (ms-marco) | Cohere DEGRADED CUAD P@1 9.27→1.97 (Pipitone). BGE untested on legal — R4 tests it. **Strong prior: NOT default-on.** |
| **prompt** | grounded / grounded_complete / minimal | No published CUAD-prompt evidence. Smoke (2026-06-21, N=5 on R1) showed `grounded` produces all-refusals on 70B even when retrieval contains the answer (rel=0.37 but util=0). `grounded_complete` unblocks utilization (rel=0.25, util=0.09, compl=0.64, adh=0.8). **Phase A R1-R4 pin `grounded_complete`**; Phase B re-perturbs prompt as a deliberate ablation. |
| **top_n (chunks → generator)** | 5 (default) / 10 | Pipeline default 5 is too few on long CUAD contracts. Smoke showed top_n=10 lifts util off the floor. **Phase A R1-R4 pin top_n=10** (set via `reranker.top_n: 10` on R1/R2/R3 where reranker=none; on R4 the bge_large reranker keeps top_n=10). |
| **summarizer (Stage 5)** | none / lexical | Pooled Exp 14b on other domains FALSIFIED the compression line. Phase B nullability check. |
| **query_classifier (Stage 1)** | none (off) | CUAD has no chit-chat. Keep OFF. |

---

## 4. The experiment matrix

**Mode: `per_example` for all of Phase A & B** (single doc per question — matches RAGBench's reference universe, no leakage). Pooled mode is meaningless for CUAD (would pool 510 contracts; legal language is so domain-shared that pooled retrieval would explode in noise — separate later experiment).

### Phase A — slim 4, research-anchored, at N=50 (~$2.46 combined)

Each of R2/R3/R4 changes **exactly one lever** vs R1 (the published champion), so each delta is interpretable. Embedder is pinned to `gte_large` (research: best open-source on LegalBench-RAG). Chunker pinned to 500-char-equivalent.

| # | name | chunker | embedder | retriever | reranker | repacker | prompt | tests |
|---|---|---|---|---|---|---|---|---|
| **R1** | `legal_sac_dense` | sac(500/0/sum=150) | gte_large | dense k=20 | none, top_n=10 | forward | grounded_complete | published-best baseline |
| **R2** | `legal_fixed500_dense` | fixed(80 words ≈ 500 chars, no overlap) | gte_large | dense k=20 | none, top_n=10 | forward | grounded_complete | **SAC delta** (R1 − R2) |
| **R3** | `legal_sac_hybrid` | sac(500/0/sum=150) | gte_large | hybrid k=20, w_dense=0.7, w_sparse=0.3, k_rrf=60 | none, top_n=10 | forward | grounded_complete | **hybrid delta** (R3 − R1) |
| **R4** | `legal_sac_rerank` | sac(500/0/sum=150) | gte_large | dense k=20 | bge_large top_n=10 | reverse | grounded_complete | **reranker delta** (R4 − R1), non-Cohere |

Pinned (not ablated in A): chunk size (500 chars), embedder (gte_large), prompt (`grounded_complete` after smoke-validation), top_n=10. All can be re-ablated in Phase B.

Research sources for each pinned choice:
- chunk=500 chars: Pipitone & Alami 2024 §5.1 (CUAD P@1 9.27 with 500-char beat 1.97 with RCTS); Reuter et al. NLLP 2025 Table 1
- SAC (summary prepended): Reuter et al. NLLP 2025 §3.2
- gte-large: Reuter et al. NLLP 2025 Appendix C, Figure 4
- dense-only default: Reuter et al. NLLP 2025 Table 2
- reranker on/off ablation: Pipitone & Alami 2024 §5.1-5.2 — Cohere DEGRADED CUAD, BGE untested

### Phase B — 8 configs at N=40 (~$0.95 combined)
Take the **top 2 from Phase A** (likely R1 and the rerank-winner; whichever is winning at N=50) and add 6 perturbations on the levers Phase A pinned:
- chunk size variants only **if** the SAC delta in Phase A was negligible: sac(400) vs sac(500) vs sac(600)
- summary length: `summary_chars` ∈ {100, 150, 200}
- prompt template: grounded vs grounded_complete vs minimal (on Phase A winner)
- top_n (R4 family if it won) ∈ {3, 5, 8}
- summarizer (Stage 5) nullability check — none vs lexical(0.5) — expected null per other-domain history

### Phase C — 3 configs at N=70 (~$0.62 combined)
The 3 best from Phase B re-run at N=70 to tighten the noise band to ±0.06. The Phase C winner is the deliverable.

### Cross-judge check — 1 config × N=30 (~$0.06)
Run `legal_rerank_only` (or whichever Phase-A config wins) once with `--backend groq --judge-model llama-3.3-70b-versatile` and once with `--backend ollama --judge-model llama3.1:8b` on the SAME 30 examples (deterministic seed). Report Pearson r and mean signed delta per metric. This is the "did the judge upgrade matter?" datapoint — important because the framework's existing TRACe validation was done against the local 8B judge.

---

## 5. Telemetry — per-stage in/out, retrievable for retrospection

Today the framework writes one **aggregate row per (config, domain)** to the matrix CSV. That's enough for the final table but blind to per-example failures. Add a parallel per-example trace WITHOUT changing the runner's CSV contract:

### Add: `src/telemetry.py`
- A `TraceLogger` that opens one `.jsonl` per (config_name, domain) under `results/telemetry/legal/`.
- Hook into `runner.run_experiment` (≤ 20 LOC): after `pipe.answer()` and after `judge.label()`, append one JSON record per example:

```json
{
  "config_name": "legal_rerank_only",
  "domain": "Legal",
  "example_id": "cuad_0",
  "question": "Does one party have the right to terminate ...",
  "stages": {
    "retrieve":   {"k": 20, "n_returned": 20, "top1_score": 0.84, "top1_chunk_id": "doc0_chunk_3", "latency_ms": 24},
    "rerank":     {"top_n": 5, "kept_chunk_ids": [...], "scores": [...], "latency_ms": 480},
    "repack":     {"order": [4,3,2,1,0], "latency_ms": 0},
    "summarize":  null,
    "generate":   {"model": "llama-3.3-70b-versatile",
                   "prompt_chars": 3120, "answer_chars": 612,
                   "prompt_tokens": 845, "completion_tokens": 153,
                   "latency_ms": 1240, "groq_cost_usd": 0.000618},
    "judge":      {"model": "llama3.1:8b",
                   "n_response_sentences": 5, "n_doc_sentences": 31,
                   "n_relevant": 14, "n_utilized": 9, "overall_supported": true,
                   "latency_ms": 8200}
  },
  "scores": {"relevance": 0.451, "utilization": 0.290, "completeness": 0.642, "adherence": true}
}
```

This is the **single source of truth for retrospection**. Once Phase A finishes, we can:
- Compute per-example variance to set Phase B sample size correctly.
- Identify "always-failing" CUAD questions (one giant 250K-char outlier exists).
- Slice by chunk size to see if relevance falls because R shrinks or because T grows.
- Cumulative live Groq cost from the `groq_cost_usd` field — primary tripwire.

### Add: `src/cost_tracker.py`
Tiny module that reads each Groq response's `usage` block (Groq returns it on the OpenAI-compatible endpoint) and writes a running tally to `results/telemetry/legal/_cost_running.json`:

```json
{"input_tokens": 712450, "output_tokens": 154002, "cost_usd": 0.542, "calls": 1180, "since": "2026-06-21T..."}
```

Print a one-liner after every config: `[cost] ~$0.54 / $5.00 spent (10.8%)`. Hard-fail the run if `cost_usd > 4.5` (a soft safety margin under the $5 hard cap).

### Modify: `groq_generator.py` (~6 LOC)
After `resp.json()` succeeds, return not just the content string but also an optional usage callback. The minimum-change path: have the generator stash `last_usage` as an attribute, and have the telemetry hook read it.

---

## 6. What is already there vs. what I need to build

### Already there (verified by reading source 2026-06-21):
- `src/data_loader.py::load_domain("Legal", split="test", n=N)` — works, returns CUAD examples with `documents`, `documents_sentences`, gold-label fields.
- `src/runner.py::run_named_matrix` — resumable, file-swap-safe, writes one aggregate row per (config, domain).
- `src/generation/groq_generator.py` — registered as type `groq`, reads `GROQ_API_KEY` from env, handles 429s.
- `src/evaluator/trace.py` — the four TRACe formulas, already validated.
- `src/judge/groq_judge.py` AND `src/judge/ollama_judge.py` — registered as `judge:groq` and `judge:ollama`. After today's patch, `scripts/run_matrix.py --backend groq` works and routes both the generator and the judge through Groq's API. Local Ollama judge is preserved as a fallback (`--backend ollama`).
- `configs/legal_*.yaml` — 4 Phase-A configs already written (`legal_sac_dense`, `legal_fixed500_dense`, `legal_sac_hybrid`, `legal_sac_rerank`). All build a Pipeline cleanly (verified 2026-06-21).
- `src/chunking/sac_chunker.py` — **NEW** (Summary-Augmented Chunking, registered as `chunker.sac`).
- `src/embeddings/sentence_transformer_embedder.py` — **PATCHED** to add `GteLargeEmbedder` subclass (registered as `embedder.gte_large`, pins `thenlper/gte-large`).
- `src/reranking/cross_encoder_reranker.py` — **PATCHED** to add `BgeLargeReranker` subclass (registered as `reranker.bge_large`, pins `BAAI/bge-reranker-large`).

### To build (estimated total ≤ 150 LOC remaining):
1. `src/telemetry.py` — `TraceLogger` writing per-example JSONL. (Partial — smoke run already producing this; verify still works under R1-R4 names.)
2. `src/cost_tracker.py` — Groq usage accounting + tripwire. (Already implemented; verified during smoke.)
3. `src/generation/groq_generator.py` — `last_usage` already exposed.
4. `src/runner.py` — telemetry hook already wired.
5. `scripts/run_legal_search.py` — thin wrapper around `run_matrix.py` (optional convenience; not blocking).
6. `scripts/load_groq_key.sh` — already in place.
7. **NEW deferred (Phase B)**: a `legalB_*.yaml` family — written after Phase A R1-R4 results inform which perturbations to run.

### Pre-flight checklist (before any Groq call):
- [ ] `cat project-paper/groq/apikey.secret | wc -c` returns ~57 (a valid key length).
- [ ] `ollama list | grep llama3.1:8b` — judge model present.
- [ ] `python -c "import src; from src.data_loader import load_domain; print(len(load_domain('Legal','test',n=3)))"` returns 3.
- [ ] `python tests/check_pipeline.py` 6/6 pass (already true after the morning's pull).
- [ ] A `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"` returns 200 (confirms key + that `llama-3.3-70b-versatile` is still served — Groq has retired model IDs before).

---

## 7. Execution order (deterministic, resumable)

```bash
# 0. Set env, verify judge
source scripts/load_groq_key.sh
ollama list | grep llama3.1:8b || ollama pull llama3.1:8b

# 1. Phase A — slim 4 (R1-R4), N=50 each, ~$2.46 combined (gen+judge on Groq 70B).
#
# Collision check: R1 (sac+dense+none), R2 (fixed+dense+none), R3 (sac+hybrid+none),
# R4 (sac+dense+bge_large) each differ from R1 in at least ONE stage TYPE, so all 4
# have distinct config_ids and can run in a SINGLE invocation. (No collision split
# needed unlike the scrapped 8-config plan.)

python scripts/run_matrix.py \
       --backend groq \
       --judge-model llama-3.3-70b-versatile \
       --gen-model   llama-3.3-70b-versatile \
       --domains Legal --n 50 \
       --configs legal_sac_dense legal_fixed500_dense legal_sac_hybrid legal_sac_rerank \
       --out results/per_example/legal_phase_A_R1R4_n50.csv

# 2. Inspect Phase A R1-R4 deltas (the three planned ablations):
#    SAC delta     = R1 − R2  (expect positive — published champion)
#    hybrid delta  = R3 − R1  (expect ≤0 — research direction; check confound)
#    reranker delta = R4 − R1 (expect unknown — Cohere degraded, BGE untested)
#    Pick the top 2 winners for Phase B.

# 3. Phase B — N=40, 8 configs (top-2 winners + 6 perturbations on locked levers)
python scripts/run_matrix.py \
       --backend groq \
       --judge-model llama-3.3-70b-versatile \
       --gen-model   llama-3.3-70b-versatile \
       --domains Legal --n 40 \
       --configs legalB_ \
       --out results/per_example/legal_phase_B_n40.csv

# 4. Pick 3 survivors; final tightening (legalC_*.yaml, N=70, ~$0.62)
python scripts/run_matrix.py \
       --backend groq \
       --judge-model llama-3.3-70b-versatile \
       --gen-model   llama-3.3-70b-versatile \
       --domains Legal --n 70 \
       --configs legalC_ \
       --out results/per_example/legal_phase_C_n70.csv

# 5. Cross-judge agreement check (Groq-70B vs local-8B on same 30 examples)
python scripts/run_matrix.py --backend ollama --judge-model llama3.1:8b \
       --gen-model llama-3.3-70b-versatile --domains Legal --n 30 \
       --configs legal_rerank_only \
       --out results/per_example/legal_crossjudge_local8b_n30.csv
python scripts/run_matrix.py --backend groq --judge-model llama-3.3-70b-versatile \
       --gen-model llama-3.3-70b-versatile --domains Legal --n 30 \
       --configs legal_rerank_only \
       --out results/per_example/legal_crossjudge_groq70b_n30.csv
```

Every phase is **resumable**: if the M4 Max sleeps or Groq 429s, re-run the same command — `run_named_matrix` skips `(config_name, domain)` rows already in the output CSV. The telemetry `.jsonl` is append-only and is also keyed by `example_id` so per-example retries don't double-count cost.

---

## 8. Decision rule for "best config"

Composite score per config (after Phase C, N=70):
```
score = 0.30 * relevance
      + 0.20 * utilization
      + 0.30 * completeness
      + 0.20 * adherence_rate
```
Weights are roughly RAGBench's own emphasis. Tie-break by adherence first (legal — hallucination is unacceptable), then by completeness.

Also report each metric on its own — the composite is for *picking*, the four individual numbers are for the writeup.

Sanity floor: the Phase-C winner must beat `legal_baseline_grounded` (A2) by **≥ 0.06** on the composite, else we report "no statistically meaningful gain over baseline on N=70" rather than overclaim. (95%-CI at N=70 ≈ ±0.06; the floor is ~1σ headroom — looser than the original ±0.04 because Option-2's Phase-C N was trimmed from 80 → 70 to fit the budget.)

---

## 9. Risks + how each is contained

| Risk | Containment |
|---|---|
| Groq retires `llama-3.3-70b-versatile` | Pre-flight `curl /models` check. Fallback model in config: `llama-3.1-70b-versatile` (older, still listed in the framework's groq_judge.py default chain). |
| Groq pricing has changed since the framework was written | Cost-tracker reads `usage` from the live response, not a hardcoded rate. The $-per-token constants live in **one** place (`src/cost_tracker.py::PRICES`); verify them against console.groq.com on launch. |
| Groq judge call rate-limit (free tier may rate-limit at ~30 req/min) | `groq_judge.py` honours 429 + Retry-After via the shared httpx path. Judge calls are sequential per example; sustained 30 req/min over Phase B = ~16 min ceiling. Run overnight if free tier. |
| CUAD's 250K-char outlier blows up index time | Already in the per_example flow each ex is re-indexed; we don't pool. The outlier still indexes in <30 s. Not a budget hit. |
| Judge silently truncates a long CUAD context window | Appendix-7.4 prompt feeds judge only the **retrieved** chunks (5 × ~512 chars ≈ 2.5 K chars), not the full doc. Groq 70B's 128K ctx easily accommodates this; the framework's GroqJudge fails LOUD (4xx) rather than silently truncating. |
| Free Groq tier daily token cap | 429 back-off is already implemented in `groq_generator.py` and `groq_judge.py` (shared path). Total Phase A+B+C is ~2.2 M input + ~0.65 M output tokens (gen + judge); well within paid-tier capacity. If on free tier, expect to run across two days. |
| I unintentionally enable summarizer/query_classifier by default | Defended already (this morning's UI fix + the rule "OFF unless config declares"). All Phase-A configs explicitly omit both. |
| Cost overshoots between two cost-tracker writes | Cost is checked every Groq call (synchronous). Worst-case overshoot: one extra call after threshold = ~$0.001. |

---

## 10. Deliverables

1. `docs/LEGAL_EXPERIMENT_PLAN.md` — this file.
2. `configs/legal_*.yaml` — 8 + 12 + 4 (some reuse).
3. `results/per_example/legal_phase_{A,B,C}_n{30,50,100}.csv` — aggregate matrices.
4. `results/telemetry/legal/*.jsonl` — per-example, per-stage telemetry.
5. `docs/legal_phase_A_analysis.md`, `docs/LEGAL_FINAL_REPORT.md` — narrative writeups.
6. Total Groq spend (recorded in `_cost_running.json`) — should land in **$0.5–$1.5**.

---

## 11. Open items for human confirmation BEFORE we start spending

1. **Groq pricing — RESOLVED** at $0.59/M in + $0.79/M out (smoke confirms; usage block reads live).
2. **Judge choice — RESOLVED** — Groq 70B scoring judge active; local 8B fallback wired.
3. **Test split — RESOLVED** — running all phases on test split (math validated upstream).
4. **Model origins (Phase A R1-R4): gte-large and BGE-reranker-large are Chinese-affiliated open-weights** (Alibaba DAMO, Beijing Academy of AI respectively). User confirmed proceed because the research findings specifically measured these on legal text and swapping introduces "untested-on-legal" risk. Weights run locally — no data leaves machine.

Phase A is **ready to launch**. Next step is a 1-config smoke at N=3 against R1 (`legal_sac_dense`) to confirm the SAC chunker + gte-large embedder produce sensible TRACe scores end-to-end before committing the full ~$2.46 Phase A run.
