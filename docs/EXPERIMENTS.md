# Experiments & Findings

**What this is:** a living log of every evaluation run in the project, the numbers, and the
*reasoning* for why each design lever moved a TRACe metric up or down. It is the analysis spine
of the Task-1 deliverable ("strategy × domain matrix + analysis of what works where, and why").

> **Reading note — TRACe metrics.** Relevance = fraction of retrieved context relevant to the
> question (diagnoses the **retriever**). Utilization = fraction of context the answer used
> (**generator/prompt**). Completeness = of the *relevant* info, how much the answer covered.
> Adherence = is the answer fully grounded / non-hallucinated (**faithfulness**). All are in [0,1];
> higher is better.

> ⚠️ **Sample-size caveat applies throughout.** Early runs use small N (10–50). At N=10 an
> adherence of "0.80" is literally "8 of 10 answers" — **directions that repeat across cells are
> trustworthy; exact values are noisy.** Final reported numbers will use larger N.

---

## Experiment 1 — Judge validation (which judge to trust)

**Question:** before scoring our own pipeline, is our LLM *judge* trustworthy? We measure how well
its TRACe scores agree with RAGBench's shipped GPT-4 reference scores.

**Setup:** judge = `llama3.1:8b` (local Ollama), N=50/domain, the example's own gold keyed sentences
fed to the judge, scores compared to reference. Metric: RMSE (lower = closer), signed error
(direction of bias), adherence accuracy. Source: `results/judge_validation__llama3-1-8b__baseline__n50.csv`.

| Domain | rel RMSE | rel signed | adh acc | JSON failures |
|---|---|---|---|---|
| covidqa (Biomedical) | 0.176 | −0.001 | 0.74 | 0/50 |
| hotpotqa (GenKnowledge) | 0.224 | +0.063 | 0.72 | 0/50 |
| emanual (CustomerSupport) | 0.283 | +0.142 | 0.82 | 0/50 |
| tatqa (Finance) | 0.233 | +0.136 | 0.78 | 0/50 |
| **mean (4 short)** | **0.229** | **+0.085** | **0.77** | **0/200** |
| cuad (Legal) | 0.542 | +0.174 | 0.62 | **13/50** |

**Findings + reasoning:**
- **Short domains: trustworthy.** Mean relevance RMSE 0.23, adherence 0.77, **zero JSON failures
  across 200 examples.** Good enough to score our pipeline on these.
- **Consistent mild over-marking** (signed +0.06…+0.14): the judge marks slightly *more* sentences
  relevant than GPT-4 did. *Reasoning:* an 8B model applies a looser "is this relevant?" threshold
  than GPT-4. Because the bias is consistent, it shifts all configs equally → it does **not** distort
  *config-vs-config* comparisons (the thing we actually care about).
- **CUAD is the outlier and a documented limitation.** Relevance RMSE 0.54, and **13/50 produced no
  parseable JSON.** *Reasoning:* CUAD contracts are 9k–56k tokens; they overflow an 8B local judge's
  context (no room left to generate) and are slow to prefill. This is a *structural* limit of local
  OSS judging on long documents, not a bug. CUAD's real pipeline scores still come from the matrix
  (Experiment 2), where the judge sees only **retrieved chunks**, not full contracts.

### Judge comparison — we tested two alternatives before locking in

To justify the judge choice (not just default to it), we validated two other candidates the same
way and compared agreement on the short domains:

| judge | rel RMSE | util RMSE | compl RMSE | adh acc | speed | notes |
|---|---|---|---|---|---|---|
| **`llama3.1:8b`** | **0.229** | **0.185** | **0.529** | 0.765 | fast | wins 3/4 metrics; most consistent across domains |
| `gemma2:9b` | 0.238 | 0.208 | 0.685 | **0.805** | fast | better adherence (but uneven: hotpotqa 0.94 / emanual 0.62); erratic on fractions |
| `llama3.1:70b` | 0.166* | 0.119* | 0.382* | 0.780* | **~6–10× slower** | *covidqa only (1 domain); too slow to finish — see below |

(\* 70B finished only covidqa: ~2.5 h for one domain, ~1–3 min/example — projected 6–10 h for all four,
and far worse for the full matrix. We stopped it after covidqa; that one domain showed only a ~0.01
relevance gain over 8B.)

**Finding:** *no judge dominates* — all three sit in the same agreement band (~0.17–0.24 relevance RMSE)
with different strengths (8B best on the fraction metrics + most consistent; gemma best on adherence but
uneven; 70B marginally best where measured but impractically slow). This matches the RAGBench paper's
point that LLM judges are imperfect, and shows **model size/choice is not the lever that meaningfully
improves judge agreement here** (9× bigger → +0.01 relevance).

**DECISION (locked 11 Jun 2026): `llama3.1:8b` is THE judge for all of Task 1.** Rationale: wins the most
metrics (incl. completeness clearly), most consistent across domains, fast, and the incumbent. Changing
the judge after producing results would invalidate all comparisons, so this is fixed and not revisited.
A different-family or larger judge (gemma/70B) was tested and is **not** a clear upgrade — recorded here
so we don't re-litigate it. (Judge and generator are independent roles; the generator can be any model.)
**Parked (only if ever needed):** schema-constrained JSON output (proven on Ollama) for fewer parse retries.

---

## Experiment 2 — Strategy × domain matrix (first pass)

**Question:** how do our RAG *design choices* affect answer quality, per domain? This is the headline
deliverable. Each cell = run a strategy on N questions through the full pipeline, judge each answer,
average the four TRACe scores.

**Setup:** generator = `llama3.2:3b`, judge = `llama3.1:8b` (both local Ollama), **N=10/cell**,
2 domains (GenKnowledge, CustomerSupport), the 2×2 strategy grid. Source: `results/ragbench_matrix.csv`.

The 2×2 grid varies **one lever at a time**:
- **prompt:** `grounded` (answer ONLY from context, else refuse) vs `minimal` (bare context+question)
- **reranker:** `cross_encoder` (re-score query–chunk pairs) vs `none` (keep retriever's top-k)

### The numbers (mean over N=10)

**Adherence** (faithfulness — the clearest signal):

| strategy | GenKnowledge | CustomerSupport |
|---|---|---|
| grounded + no-rerank | 0.600 | 0.300 |
| **grounded + rerank** | **0.800** | **0.400** |
| minimal + no-rerank | 0.500 | 0.300 |
| **minimal + rerank** | **0.778** | **0.400** |

**Relevance:**

| strategy | GenKnowledge | CustomerSupport |
|---|---|---|
| grounded + no-rerank | 0.285 | 0.327 |
| grounded + rerank | 0.340 | 0.350 |
| minimal + no-rerank | 0.437 | 0.280 |
| minimal + rerank | 0.467 | 0.235 |

**Utilization** / **Completeness** (full numbers in the CSV): utilization 0.11–0.29;
completeness 0.10–0.63 (high variance at N=10).

### Findings + reasoning

**F1 — The reranker improves adherence, consistently (the strongest result).**
In **all four** prompt×domain pairs, turning the cross-encoder reranker ON raised adherence:
GenKnowledge grounded 0.60→0.80, GenKnowledge minimal 0.50→0.78, CustomerSupport both 0.30→0.40.
*Reasoning:* the cross-encoder re-scores each (question, chunk) pair *together* (not just vector
similarity), so it surfaces chunks that genuinely answer the question. With better material in the
prompt, the generator can ground its answer instead of inventing — directly lifting faithfulness.
Because it repeats across every pair, this is the most trustworthy finding in the matrix.

**F2 — Domain matters more than strategy for adherence.** Every CustomerSupport cell (0.30–0.40) is
well below its GenKnowledge counterpart (0.50–0.80). *Reasoning (hypothesis):* CustomerSupport
(emanual) answers are procedural/specific ("use the HYPERLINK function, click cell…"); a small 3B
generator more easily drifts from the exact source wording, and the judge marks those as unsupported.
GenKnowledge answers are more factual/extractive, easier to ground. Worth confirming at higher N.

**F3 — `minimal` prompt shows *higher* relevance than `grounded` on GenKnowledge** (0.44–0.47 vs
0.29–0.34). *Important nuance:* **relevance measures the retrieved context, which the prompt does not
change** — retrieval is identical for grounded vs minimal at the same reranker setting. So this
difference is **not** a real prompt effect; it's **N=10 noise** (and/or judge variance), and a good
illustration of why we must not over-read small-N cells. (If it persisted at large N with identical
retrieval, it would signal a judge artifact, not a pipeline effect.)

**F4 — Utilization and completeness are noisy and low.** Utilization 0.11–0.29 means the answers use
a small fraction of the retrieved context — partly real (short answers don't need 5 chunks), partly
that a 3B model writes terse answers. Completeness swings wildly (0.10–0.63) because it is a *derived*
ratio (relevant ∩ used / relevant) that compounds R and U noise; at N=10 it's the least reliable cell.

### Caveats for this experiment
- **N=10 → directional only.** F1 (reranker→adherence) is trustworthy because it repeats 4×; F3/F4
  are within noise. Re-run at N=30–50 before citing exact values.
- **Only 2 of 5 domains.** Biomedical, Legal/CUAD, Finance not yet run.
- **Adherence judged by llama3.1:8b**, which mildly over-marks (Experiment 1) — read absolute
  adherence with that grain of salt; *comparisons* between cells remain valid (same judge throughout).
- **Generator is the small `llama3.2:3b`** (fast-dev). A larger generator would likely lift utilization
  and adherence; that's a future lever.

---

## Open follow-ups (carried forward)

1. **Scale the matrix:** all 5 domains, N=30–50, so numbers are reportable not just directional.
2. **Compare to reference scores:** the problem statement asks us to compare our pipeline's TRACe
   scores to RAGBench's shipped reference scores, per domain. Not yet built.
3. **More levers:** chunking strategy (esp. on CUAD), embedder (MiniLM vs BGE), repacker
   (forward/reverse/sides), bigger generator.
4. **Judge quality pass:** Llama-3-70B / Gemma-2-27B as judge; schema-constrained JSON.
5. **CUAD:** judge-validation is impractical locally (Exp 1); decide whether to report CUAD pipeline
   scores with a caveat or document the limitation.

---

*Data sources: `results/judge_validation__llama3-1-8b__baseline__n50.csv` (Exp 1),
`results/ragbench_matrix.csv` (Exp 2). See `PIPELINE_WALKTHROUGH.md` for how a single number is
produced end-to-end.*
