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

## Experiment 3 — Our pipeline vs RAGBench reference scores (where do we lag?)

**Question:** the problem statement asks us to compare our computed TRACe scores against the
dataset's shipped *reference* scores, explain *why* they differ, and optimise. Concretely: **where
is our pipeline lagging, so we know which component to experiment on next?**

**This is a different comparison from Experiment 1 — don't conflate them.** Exp 1 fed the judge the
*same* gold inputs the reference was built on, to ask "does our *judge* agree with GPT-4?" (it
*calibrates the ruler*). Exp 3 takes the scores our *pipeline* actually earned — our answer, over
our retrieved context — and lays them next to the reference, to ask "is our *system* as good as the
reference system?" (it *measures the cake with the calibrated ruler*). You need Exp 1 to trust Exp 3.

Both numbers end in the *same* `trace.py` math; only the inputs differ:

```
OUR number:        our answer  + our retrieved context  -> judge labels -> TRACe
REFERENCE number:  gold answer + full gold documents    -> gold labels  -> TRACe
```

**Setup:** the reference baseline is the shipped `*_score` fields (proven equal to our own math at
RMSE 0 in Exp 0 / notebook 02), averaged over the **same N=10 examples** the matrix sampled
(`load_domain(domain, "test", n=10, seed=42)` — a fair head-to-head, not a different slice). `gap =
ours − reference`. Producer: `scripts/compare_reference.py` → `results/reference_comparison.csv` +
`results/figures/compare_*.png`. No model needed (the reference fields are free), so this is instant.

> **Each metric's gap means something different — they are NOT symmetric:**
> - **relevance** → cleanest *retriever* signal (it ignores the answer entirely). Reference =
>   relevant-density of the raw document pile; ours = relevant-density of what our retriever
>   surfaced. A good retriever *filters junk*, so ours should sit **above** reference; ours *below*
>   = the retriever is dropping relevant material.
> - **adherence** → *prompt + generator faithfulness*. Reference is a real system's (gpt-3.5) rate,
>   so **match-or-beat is the goal**.
> - **utilization** → mixes retrieval *and* generation; triangulate.
> - **completeness** → a derived ratio; noisiest; read last.

### The gaps (gap = ours − reference; matched-N, N=10)

| metric | GenKnowledge (our range) | CustomerSupport (our range) | gap direction |
|---|---|---|---|
| **relevance** | −0.13 … +0.05 (mixed) | **+0.12 … +0.23 (all positive)** | at/above reference |
| **utilization** | −0.13 … +0.01 | +0.02 … +0.11 | near/slightly above |
| **adherence** | **−0.10 … −0.40 (all negative)** | **−0.20 … −0.30 (all negative)** | **below reference** |
| **completeness** | −0.02 … −0.35 | −0.21 … −0.48 | below reference |

(Reference levels: GenKnowledge adherence 0.90, relevance 0.42; CustomerSupport adherence 0.60,
relevance 0.12. Full per-cell numbers in `results/reference_comparison.csv`.)

### Findings + reasoning

**F1 — We are NOT lagging on retrieval (the surprise).** Relevance gaps are mostly *positive* —
strongly so on CustomerSupport (+0.12…+0.23 across all four strategies). Our retriever concentrates
relevant material *above* the raw-document baseline, exactly what a good retriever should do. *So the
intuitive "improve retrieval" reflex (hybrid search, better embedder) is aimed at the wrong place* —
the reference comparison says retrieval is already at or above par. (Caveat: our relevance is judged
by `llama3.1:8b`, which mildly over-marks ~+0.09 from Exp 1 — a *small* positive gap may be partly
that. The bias is ~constant across configs, so config-vs-config comparisons stay valid.)

**F2 — We lag on adherence, everywhere (the real gap).** Every single cell is below reference
(−0.10 to −0.40). *Reasoning:* adherence is the generator's faithfulness, and ours is the tiny
`llama3.2:3b`, vs the reference's gpt-3.5. A small model drifts from source wording / adds
unsupported detail, and the judge marks those sentences unsupported. **This is the component to
experiment on next: a bigger generator and/or a stronger grounding prompt — not the retriever.**

**F3 — The reranker narrows the adherence gap (second confirmation of Exp 2's F1).** On
GenKnowledge the gap shrinks from −0.30 (grounded, no-rerank) to −0.10 (grounded, rerank), and
−0.40 → −0.12 (minimal). Seeing the reranker help *both* in absolute terms (Exp 2) *and* relative to
the reference (here) makes it the most robust positive result in the project so far. The
`compare_adherence.png` figure shows this at a glance (blue rerank bars reach toward the red
reference bars).

**F4 — Completeness is below reference everywhere.** Our 3B answers are terse and cover less of the
relevant material than gpt-3.5's. Consistent with F2 (same small-generator root cause) and with the
metric being a noisy derived ratio at N=10 — read it as corroborating, not independent.

### Caveats for this experiment
- **N=10 → directional.** The *signs* (relevance ≥ 0, adherence < 0) repeat across all cells and are
  trustworthy; exact gap magnitudes are noisy. Re-running at **N=50** (in progress) will firm these up.
- **Matched-N baseline is itself noisy at N=10** — e.g. GenKnowledge reference relevance is 0.42 over
  these 10 examples but 0.31 over the full split (`--full-split`). The *direction* of the adherence gap
  is robust to this; borderline relevance cells are not. The full-split baseline is the stable yardstick.
- **Judge over-marking** (Exp 1, +0.09 relevance) inflates our relevance slightly — accounted for in F1.

**Bottom line / direction for experiments:** spend the next effort budget on the **generator and the
grounding prompt** (close the adherence + completeness gaps), *not* on retrieval (already at/above
reference). This is the reference comparison doing its job: aiming the ablations.

---

## Open follow-ups (carried forward)

1. **Scale the matrix:** all 5 domains, N=30–50, so numbers are reportable not just directional.
   (N=50 on the 2 starter domains is in progress → will refresh Experiment 3's numbers.)
2. ✅ **Compare to reference scores (Experiment 3, done):** built `src/evaluator/compare.py` +
   `scripts/compare_reference.py`. Headline: retrieval is at/above reference; **adherence is the gap**
   → next effort goes to the generator + prompt, not the retriever.
3. **More levers (now aimed by Exp 3):** **bigger generator + grounding prompt FIRST** (closes the
   adherence/completeness gap); then chunking (esp. CUAD), embedder (MiniLM vs BGE), repacker.
4. **Judge quality pass:** Llama-3-70B / Gemma-2-27B as judge; schema-constrained JSON.
5. **CUAD:** judge-validation is impractical locally (Exp 1); decide whether to report CUAD pipeline
   scores with a caveat or document the limitation.

---

*Data sources: `results/judge_validation__llama3-1-8b__baseline__n50.csv` (Exp 1),
`results/ragbench_matrix.csv` (Exp 2), `results/reference_comparison.csv` + `results/figures/compare_*.png`
(Exp 3). See `PIPELINE_WALKTHROUGH.md` for how a single number is produced end-to-end.*
