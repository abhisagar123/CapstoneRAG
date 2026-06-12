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
(direction of bias), adherence accuracy. Source: `results/validation/judge_validation__llama3-1-8b__baseline__n50.csv`.

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
2 domains (GenKnowledge, CustomerSupport), the 2×2 strategy grid. Source: `results/per_example/ragbench_matrix.csv`.

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

**F1 — The reranker improves adherence, consistently (the strongest result *at N=10*).**
In **all four** prompt×domain pairs, turning the cross-encoder reranker ON raised adherence:
GenKnowledge grounded 0.60→0.80, GenKnowledge minimal 0.50→0.78, CustomerSupport both 0.30→0.40.
*Reasoning (proposed at the time):* the cross-encoder re-scores each (question, chunk) pair *together*
(not just vector similarity), so it surfaces chunks that genuinely answer the question; with better
material in the prompt, the generator can ground its answer instead of inventing.
> **⚠️ SUPERSEDED at N=50 — see Experiment 3, F3.** This "consistent across 4 pairs" result did **not**
> replicate at N=50: the adherence gain collapsed to ~+0.03 (flat) and went *negative* on one cell. At
> N=10 each cell was 10 answers, so "0.60→0.80" was a 2-answer swing — sampling luck, not a real effect.
> Kept here as written to document the correction. **A finding repeating across small cells can still be
> noise; only scaling N exposed it.**

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
RMSE 0 in Exp 0 / notebook 02), averaged over the **same examples** the matrix sampled
(`load_domain(domain, "test", n=N, seed=42)` — a fair head-to-head, not a different slice). `gap =
ours − reference`. Producer: `scripts/compare_reference.py` → `results/per_example/reference_comparison*.csv` +
`results/per_example/figures*/compare_*.png`. No model needed (the reference fields are free), so this is instant.

**We ran it at two sizes — N=10 first, then N=50** (2 domains, the 2×2 grid). The N=50 numbers below
are the reportable ones; the N=10→N=50 shift is itself a finding (F3), so both are kept. Sources:
`results/per_example/ragbench_matrix.csv` / `reference_comparison.csv` (N=10) and the `*_n50` files (N=50).

> **Each metric's gap means something different — they are NOT symmetric:**
> - **relevance** → cleanest *retriever* signal (it ignores the answer entirely). Reference =
>   relevant-density of the raw document pile; ours = relevant-density of what our retriever
>   surfaced. A good retriever *filters junk*, so ours should sit **above** reference; ours *below*
>   = the retriever is dropping relevant material.
> - **adherence** → *prompt + generator faithfulness*. Reference is a real system's (gpt-3.5) rate,
>   so **match-or-beat is the goal**.
> - **utilization** → mixes retrieval *and* generation; triangulate.
> - **completeness** → a derived ratio; noisiest; read last.

### The gaps (gap = ours − reference; matched-N, **N=50**)

| metric | GenKnowledge (our range) | CustomerSupport (our range) | gap direction |
|---|---|---|---|
| **relevance** | **+0.085 … +0.104 (all positive)** | **+0.071 … +0.152 (all positive)** | above reference |
| **utilization** | −0.124 … +0.118 (prompt-dependent) | −0.031 … +0.064 | mixed |
| **adherence** | **−0.133 … −0.280 (all negative)** | **−0.240 … −0.320 (all negative)** | **below reference** |
| **completeness** | −0.200 … −0.346 | **−0.359 … −0.539** | **well below reference** |

(N=50 reference levels: GenKnowledge adherence 0.80, relevance 0.327, completeness 0.701;
CustomerSupport adherence 0.70, relevance 0.139, completeness 0.696. A few cells scored 48–49/50 —
the judge skipped a handful of unparseable/timed-out answers, as designed. Full per-cell numbers in
`results/per_example/reference_comparison_n50.csv`; charts in `results/per_example/figures_n50/`.)

### Findings + reasoning

**F1 — We are NOT lagging on retrieval (now unanimous at N=50).** Relevance gaps are positive in
**all 8 cells** (+0.07 … +0.15). At N=10 GenKnowledge was mixed (one cell −0.13); scaling to N=50
resolved it cleanly positive — our retriever concentrates relevant material *above* the raw-document
baseline, exactly what a good retriever should do. *So the intuitive "improve retrieval" reflex
(hybrid search, better embedder) is aimed at the wrong place* — retrieval is already at/above par.
(Caveat: our relevance is judged by `llama3.1:8b`, which mildly over-marks ~+0.09 from Exp 1 — so the
GenKnowledge gap of ~+0.09 may be *mostly judge bias*, while CustomerSupport's larger +0.13…+0.15 is
real headroom above it. The bias is ~constant across configs, so config-vs-config stays valid.)

**F2 — We lag on adherence, everywhere (the real gap; stable across N).** All 8 cells below reference
(−0.13 … −0.32 at N=50; the same all-negative picture as N=10). *Reasoning:* adherence is the
generator's faithfulness, and ours is the tiny `llama3.2:3b`, vs the reference's gpt-3.5. A small
model drifts from source wording / adds unsupported detail, and the judge marks those sentences
unsupported. **This is the component to experiment on next: a bigger generator and/or a stronger
grounding prompt — not the retriever.**

**F3 — ⚠️ The reranker's adherence "win" did NOT survive scaling — a small-sample mirage (the most
instructive finding).** At **N=10** the headline result (Exp 2 F1) was "the cross-encoder reranker
raised adherence in all 4 pairs" (e.g. GenKnowledge grounded 0.60→0.80). At **N=50 that effect
collapses to near-zero, and on one cell it reverses:**

| pair | N=10 (no-rerank → rerank) | N=50 (no-rerank → rerank) |
|---|---|---|
| GenKnowledge, grounded | 0.60 → 0.80 (+0.20) | 0.64 → 0.67 (+0.03, flat) |
| GenKnowledge, minimal | 0.50 → 0.78 (+0.28) | 0.52 → 0.55 (+0.03, flat) |
| CustomerSupport, grounded | 0.30 → 0.40 (+0.10) | 0.46 → **0.38 (−0.08, hurt)** |
| CustomerSupport, minimal | 0.30 → 0.40 (+0.10) | 0.44 → 0.43 (−0.01, flat) |

*Reasoning:* at N=10, "0.60→0.80" was literally "6 vs 8 of 10 answers" — a 2-answer swing that looked
like a strong effect but was mostly **sampling luck**. At N=50 the reranker gives **no reliable
adherence gain**. It *does* still help relevance/utilization (it surfaces better chunks), but better
chunks in front of a weak 3B generator don't translate into faithfulness. **Lesson: a finding that
repeats across N=10 cells can still be noise if every cell is small — only scaling N revealed it.**
This is a strong report anecdote for *why* rigorous evaluation (not just a directional matrix) matters,
and it reinforces F2: the bottleneck is the generator, not retrieval.

**F4 — Completeness is well below reference (the widest gap; clearer at N=50).** −0.20 … −0.54, worst
on CustomerSupport (reference ~0.70, ours ~0.16–0.34). Our terse 3B answers cover far less of the
relevant material than gpt-3.5's. The gap *grew* vs N=10 because the N=50 reference completeness (~0.70)
is higher/more-stable than the noisy N=10 estimate. Same small-generator root cause as F2.

### Caveats for this experiment
- **N=50 is reportable but still modest; N=10 was only directional.** F3 is the cautionary tale — the
  *signs* of F1/F2 held from N=10 to N=50, but a *magnitude*-based finding (reranker→adherence) did not.
  Trust signs that survive scaling; be wary of magnitudes until N is larger and ideally with CIs.
- **Matched-N baseline is itself noisy at small N** — e.g. GenKnowledge reference relevance was 0.42
  over the N=10 sample but 0.33 over the N=50 sample and 0.31 over the full split (`--full-split`). The
  full-split baseline is the stable yardstick; matched-N is the fair head-to-head at a given N.
- **Judge over-marking** (Exp 1, +0.09 relevance) inflates our relevance — so the GenKnowledge
  relevance gap is plausibly mostly bias; CustomerSupport's is largely real (accounted for in F1).

**Bottom line / direction for experiments:** spend the next effort budget on the **generator and the
grounding prompt** (close the adherence + completeness gaps), *not* on retrieval (already at/above
reference) and *not* on the reranker (no adherence payoff at N=50). This is the reference comparison
doing its job: aiming the ablations — and F3 shows it also *guards* us from over-reading small-N wins.

---

## Experiment 4 — A completeness-aware grounding prompt (a new component, aimed by Exp 3)

**Question:** Exp 3 said our two gaps vs reference are **adherence** and **completeness**, both
prompt/generator-side. The baseline `grounded` prompt targets adherence ("use ONLY the context") but
says nothing about *coverage* — yet completeness is "of the relevant material, how much did the answer
cover?". **Can a prompt that adds a thoroughness instruction lift completeness — without sinking
adherence?** This is a directed ablation: the reference comparison pointed here, so we built a
component to test it.

**The component (new):** `GroundedCompletePromptBuilder` (prompt type `grounded_complete`,
`src/prompting/grounded_complete_prompt_builder.py`). It keeps the baseline's strict grounding +
refusal **verbatim** and adds one scoped sentence: *"Be thorough: include every relevant detail that
the context provides … but do not add any information that is not in the context."* The "from the
context" scope is deliberate — to push recall (completeness) while trying to hold the grounding
(adherence). **Built-in tension (predicted before running):** "be thorough" can tempt a small 3B model
to pad with unsupported claims, which would *hurt* adherence. Measuring that trade-off IS the
experiment.

**Setup:** clean one-variable ablation — `grounded_complete_norerank.yaml` is **identical** to
`grounded_norerank.yaml` except the prompt type, run at the **same seed=42, N=50**, so it appended just
2 cells to the existing matrix and compares head-to-head against rows we already had. Generator =
`llama3.2:3b`, judge = `llama3.1:8b`. Source: `results/per_example/ragbench_matrix_n50.csv` (the two
`grounded_complete_norerank` rows) + `reference_comparison_n50.csv`.

### grounded_complete vs grounded (no-rerank, N=50; prompt is the only difference)

| metric | GenKnowledge Δ | CustomerSupport Δ |
|---|---|---|
| relevance | +0.009 (~flat — expected; prompt doesn't change retrieval) | −0.024 (~flat) |
| **utilization** | **+0.105** (uses more context) | +0.025 |
| **completeness** | **+0.119** (gap to reference −0.31 → −0.19) | +0.030 |
| **adherence** | **−0.040** (gap −0.16 → −0.20) | +0.009 (~flat) |

### Findings + reasoning

**F1 — The completeness push WORKED (the intended effect).** On GenKnowledge completeness rose
+0.119 — `grounded_complete` is now the **best completeness of any no-rerank config**, and its
reference gap (−0.19) is the smallest of the grounded family. Utilization rose in lockstep (+0.105):
told to be thorough, the 3B model genuinely *read and used more* of the retrieved context. The
relevance non-change (+0.009) is a good sanity check — the prompt doesn't touch retrieval, so
relevance *shouldn't* move, and it didn't.

**F2 — It cost a little adherence on GenKnowledge — the predicted precision/recall trade.** Adherence
dipped −0.040 (gap widened −0.16 → −0.20). *Reasoning:* pushing coverage nudged the model to include
material it couldn't fully ground, so the judge marked a few more sentences unsupported. This is the
tension we flagged *before* the run, now observed: **a completeness gain trades against faithfulness.**
It's a *trade*, not a free win.

**F3 — The effect is domain-dependent.** CustomerSupport barely moved (completeness +0.030, adherence
+0.009) — the thoroughness instruction helped much less on short procedural answers than on the more
expository GenKnowledge answers. So "be thorough" is a GenKnowledge-shaped lever, not a universal one.

### Caveats
- **The adherence cost (−0.040 on GenKnowledge) is ±2 answers out of 50 — within small-sample noise**
  (the Exp 3 F3 lesson). Read it as: *completeness gain is solid and sizable; the adherence cost is
  small and not yet confirmed real.* Needs larger N (ideally with confidence intervals) to be sure.
- **One generator (3B).** A larger generator might get the completeness gain with less adherence cost
  (more capacity to be both thorough AND grounded) — a natural next ablation (prompt × generator size).
- Judge over-marking (Exp 1) applies equally to both prompts, so the *delta* is unaffected.

**Takeaway:** a directed, single-component experiment confirmed the reference comparison's diagnosis is
actionable — the completeness gap *is* movable by the prompt. But it surfaced a real precision/recall
tension, so the better fix is likely **prompt + a bigger generator together**, not the prompt alone.
Good report material: "we let the gap analysis pick the experiment, predicted a trade-off, and measured
it" is exactly the rigorous-evaluation story the project is graded on.

---

## Experiment 5 — Stronger embedder (MiniLM → BGE): a control for "is retrieval the bottleneck?"

**Question:** the "Best Practices in RAG" paper ranks embedding models on MS MARCO (Table 2) and puts
our `all-MiniLM` family at the **bottom** (MRR@10 ≈ 11) with **BAAI/bge-base ≈ 3× higher** (≈ 37). So a
MiniLM→BGE swap is a *paper-validated retrieval upgrade*. But Exp 3 said **retrieval is NOT our gap**
(relevance already at/above reference). So this is deliberately a **control**: does a real retrieval
improvement move our TRACe? If it barely moves, that's independent confirmation the generator — not
retrieval — is the bottleneck (and justifies the bigger-generator move with evidence, not assumption).

**Setup:** clean one-variable ablation — `grounded_bge_norerank.yaml` is identical to
`grounded_norerank.yaml` except `embedder: {type: sentence_transformer, model: BAAI/bge-base-en-v1.5}`
(768-dim) vs `minilm` (all-MiniLM-L6-v2, 384-dim). Same seed=42, N=50; appended 2 cells. Generator
`llama3.2:3b`, judge `llama3.1:8b`. Source: the two `grounded_bge_norerank` rows in
`results/per_example/ragbench_matrix_n50.csv`.

### bge-base vs minilm (grounded, no-rerank, N=50; embedder is the only difference)

| metric | GenKnowledge Δ | CustomerSupport Δ |
|---|---|---|
| **relevance** | +0.002 (~flat) | **−0.050** (down) |
| utilization | −0.021 | −0.030 |
| completeness | −0.006 (~flat) | +0.017 (~flat) |
| adherence | −0.020 | −0.052 |

### Findings + reasoning

**F1 — The stronger embedder did NOT improve TRACe (the informative null result).** Relevance was flat
on GenKnowledge (+0.002) and *fell* on CustomerSupport (−0.050); nothing improved materially. A
paper-validated retrieval upgrade produced ~no retrieval gain in our setup. **This is not "BGE is
worse than MiniLM"** — it's that a better embedder only helps when retrieval is *hard*, and ours isn't:

**F2 — Why it didn't help: small candidate pools + a ranking-vs-TRACe mismatch.** Two reasons. (a) We
run **per-example** corpus mode — the retriever picks top-k from *one example's own documents*, a tiny
already-on-topic pool. With little to discriminate, MiniLM and BGE retrieve nearly the same chunks. (b)
The paper's BGE advantage is on **MRR@10** (does the #1 result rank well), whereas our **relevance** is
*fraction of retrieved sentences that are relevant* — different target. So the paper's embedder ranking
doesn't transfer to our metric in this configuration. (Connects to §13: these two domains ship
pre-segmented short docs → small pools → weak retrieval lever, exactly as EDA predicted.)

**F3 — The small drops trace back to the generator, not the embedder.** The −0.05 relevance on
CustomerSupport is ≈±2–3 sentences at N=50 (near noise), but adherence/utilization dipped too. Reasoning:
BGE retrieved *slightly different* chunks, and the weak 3B generator produced slightly less-grounded
answers off them. That's the generator being **sensitive to retrieval reshuffling**, not the embedder
being bad — another finger pointing at the generator as the fragile link.

**F4 — The control fired as designed → retrieval is confirmed not-the-bottleneck.** Three independent
interventions now agree: reranker (Exp 3 F3, no adherence help at N=50), prompt (Exp 4, only a traded
gain), embedder (here, no help). **All the retrieval/prompt-side levers are largely tapped out on these
two domains; the generator is the lever.**

### Caveats
- **Per-example mode is the LEAST favorable setting for an embedder swap** (tiny pools). On a **pooled**
  per-domain corpus (the real Phase-2 setting, large pool) or on **CUAD** (1 long doc, real chunking +
  retrieval challenge), BGE could still matter. So this is "no help *in this configuration*," **not**
  "BGE is useless." Re-test embedder on pooled/CUAD before concluding globally.
- N=50; the CustomerSupport −0.05 is near noise. The *direction of the conclusion* (no gain) is what's
  robust, not the exact deltas.
- Judge over-marking (Exp 1) applies equally to both embedders, so the *delta* is unaffected.

**Takeaway:** a paper-recommended upgrade that *didn't* move our metric is a real finding, not a failed
experiment — it's the gap analysis (Exp 3) being **confirmed by an independent, paper-backed control**.
The evidence that the generator is the bottleneck is now triangulated across three levers. Next
experiment: the generator itself.

---

## Experiment 6 — Squeezing the small (3B) generator: four cheap levers (A/B/C/D)

**Question:** before paying for a bigger/slower generator, how far can we push the fast `llama3.2:3b`?
Each sub-experiment is a one-variable change off the `grounded_norerank` baseline (or its rerank
sibling), N=50, same seed. Generator stays `llama3.2:3b`, judge `llama3.1:8b`. Whatever helps here also
transfers to the bigger model later, so this is not throwaway work.

### 6A — Longer answers (max_new_tokens 256 → 512): a VERIFIED NULL

**Hypothesis:** completeness (our worst gap) might be low because answers are *truncated* at 256 tokens,
not because the model is terse. **Result: raising the cap changed nothing** — GenKnowledge identical to
the last decimal (rel 0.411, util 0.175, compl 0.395, adh 0.640); CustomerSupport moved only by
judge-parse jitter (rel 0.291→0.286). **Verified the cause directly:** probed real grounded answers —
they were **3, 17, 20, 62 words** against a ~190-word (256-token) ceiling. Nothing was being truncated.
*So the terse, low-coverage answers are a model behaviour, not a length-cap artifact* — this rules out
the cheapest explanation for the completeness gap and (again) points at generator capability + prompting.
(Source: `results/per_example/ragbench_matrix_n50_maxtok512.csv` vs the grounded_norerank rows.) A `--max-new-tokens`
flag was added to `scripts/run_matrix.py` (the cap was previously hardcoded at 256).

### 6B — grounded_complete + reranker: the WIN (the levers compound)

The missing cell of Exp 4 — the completeness prompt **with** the cross-encoder reranker.

**Result: `grounded_complete_rerank` reaches GenKnowledge adherence 0.714 — the best of ANY config, and
the smallest adherence gap to reference in the project (−0.086, vs −0.16 at the plain grounded baseline).**
Isolating the *prompt* effect when reranking is on (vs `grounded_rerank`), GenKnowledge lifts on **all
metrics at once**: utilization +0.114, completeness +0.142, adherence +0.048.

*Reasoning — an interaction effect a one-at-a-time matrix misses.* Exp 3 F3 found the reranker gave **no**
adherence help with the *plain* grounded prompt. But the reranker (better chunks) and grounded_complete
("use them fully") **compound**: with genuinely relevant chunks in front of it AND an instruction to cover
them, the 3B grounds more of its answer instead of drifting. Two levers each judged marginal alone are
**synergistic together** — the headline finding of this batch. ⚠️ **GenKnowledge only:** on CustomerSupport
6B was flat-to-down (adh 0.469→0.408). So it's "best lever we've found, on the domain where levers work,"
not universal.

### 6C — sides repacker (vs reverse): the clear LOSER

Swapping `reverse` → `sides` **hurt almost everything**: GenKnowledge completeness −0.091, CustomerSupport
completeness −0.150, adherence −0.060 in both. **Verdict: `reverse` is clearly better than `sides` for our
setup; drop `sides`.** This *contradicts* the Best Practices paper's "reverse ≈ sides" — a legitimate
finding: the paper used a much stronger generator; a 3B apparently handles "most-relevant-chunk-last"
(reverse) better than splitting the strong chunks to both ends (sides). Repacker ordering is now a
closed question for us — `reverse` stays.

### 6D — extractive prompt ("quote the source"): a narrow, mostly-negative split

A new prompt (`extractive`) attacking adherence from a different angle than grounded_complete: "quote or
closely paraphrase the context; do not rephrase." **Result: mostly negative** — it made the already-terse
3B *terser* (GenKnowledge completeness −0.057, utilization −0.008; CustomerSupport −0.056/−0.041). The one
bright spot: **CustomerSupport adherence 0.500 — the best no-rerank adherence on that domain** (+0.040).
*Reasoning:* forcing source-wording cuts the invented phrasings that break adherence, which helps on
procedural CustomerSupport answers, but the "don't elaborate" framing suppresses coverage everywhere else.
Niche, not a general lever; documented and parked.

### Scoreboard + takeaway

| sub-exp | lever | result | verdict |
|---|---|---|---|
| 6A | max_new_tokens 256→512 | null (answers never hit the cap — verified 3–62 words) | rules out truncation as the cause |
| **6B** | grounded_complete **+ reranker** | **adh 0.714 GenKnowledge (best config); prompt+reranker compound** | **the win — carry into the bigger-gen run** |
| 6C | sides repacker | worse across the board | reverse > sides; drop sides |
| 6D | extractive prompt | terser; only a CustomerSupport-adherence niche | parked |

**Net:** we *did* squeeze a real win out of the 3B — **`grounded_complete` + cross-encoder reranker** is the
best config to date (on GenKnowledge), and the prompt×reranker **synergy** is the key new insight. It also
gives the bigger-generator experiment a sharp hypothesis: run **grounded_complete + rerank** there, since
that's where the compounding lives. Caveats: all N=50 (directional on exact magnitudes); GenKnowledge is
the domain where levers move and CustomerSupport stays stubborn (a real domain-difficulty finding).

---

## Open follow-ups (carried forward)

1. **Scale the matrix:** all 5 domains, N=30–50, so numbers are reportable not just directional.
   (N=50 on the 2 starter domains is in progress → will refresh Experiment 3's numbers.)
2. ✅ **Compare to reference scores (Experiment 3, done):** built `src/evaluator/compare.py` +
   `scripts/compare_reference.py`. Headline: retrieval is at/above reference; **adherence is the gap**
   → next effort goes to the generator + prompt, not the retriever.
3. **Bigger generator — the clear priority, with a SHARP hypothesis from Exp 6B.** The 3B is the
   bottleneck (Exp 3/4/5), AND Exp 6B found `grounded_complete` + reranker is the best 3B config (the
   prompt×reranker synergy). So run the bigger generator (`llama3.1:8b` / `gemma2:9b`, both pulled) on
   **`grounded_complete_rerank`**, not the plain baseline — test whether the synergy amplifies with capacity.
4. **Confirm Exp 6B at larger N + on more domains:** the GenKnowledge win is sizable but CustomerSupport
   was flat — is the synergy GenKnowledge-specific or general? Needs more domains / higher N.
5. **Re-test embedder where it CAN matter (Exp 5 caveat):** bge vs minilm on a **pooled** corpus or on
   **CUAD** (big candidate pool / real retrieval challenge) — per-example mode was too easy to show a gap.
6. ✅ **Repacker decided (Exp 6C):** `reverse` > `sides` for our setup — drop `sides`. (forward untested but low-priority.)
7. **Other levers:** chunking (esp. CUAD), hybrid retrieval (BM25+RRF) — best on a pooled corpus.
8. **Judge quality pass:** Llama-3-70B / Gemma-2-27B as judge; schema-constrained JSON.
9. **CUAD:** judge-validation is impractical locally (Exp 1); decide whether to report CUAD pipeline
   scores with a caveat or document the limitation.

---

*Data sources: `results/validation/judge_validation__llama3-1-8b__baseline__n50.csv` (Exp 1),
`results/per_example/ragbench_matrix.csv` (Exp 2, N=10), `results/per_example/ragbench_matrix_n50.csv` +
`results/per_example/reference_comparison_n50.csv` + `results/per_example/figures_n50/compare_*.png` (Exp 3, 4, 5 & 6B/C/D);
`results/per_example/ragbench_matrix_n50_maxtok512.csv` (Exp 6A).
See `PIPELINE_WALKTHROUGH.md` for how a single number is produced end-to-end.*
