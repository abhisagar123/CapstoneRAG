# Pooled-Corpus Experiments & Findings

**What this is:** the experiment log for the **pooled corpus** track — the *realistic* RAG
setting where the retriever searches the **whole domain corpus**, not just each question's own
documents. Kept separate from `EXPERIMENTS.md` (the per-example track) on purpose, because pooled
asks a different question and carries a different baseline + caveat.

> **Reading note — TRACe metrics.** Relevance = fraction of retrieved context relevant to the
> question (the **retriever**). Utilization = fraction of context the answer used (**generator/
> prompt**). Completeness = of the *relevant* info, how much the answer covered. Adherence = is the
> answer grounded / non-hallucinated (**faithfulness**). All in [0,1]; higher is better.

> ⚠️ **Per-example vs pooled — do NOT mix them.**
> - *Per-example* (`EXPERIMENTS.md`): each question is answered against ONLY its own documents —
>   matches RAGBench's reference-score universe, so those scores ARE comparable to the reference.
> - *Pooled* (this doc): the index is the WHOLE domain corpus; a question may retrieve other
>   questions' docs. This is realistic RAG, but our retrieval universe ≠ the reference's, so
>   **relevance/utilization/completeness are valid as OUR-OWN scores but NOT comparable to the
>   RAGBench reference** (adherence still is). We therefore compare **pooled-vs-pooled** here, and
>   defer any reference comparison. Pooled results live in their own CSV
>   (`results/ragbench_matrix_n50_pooled.csv`), never merged with the per-example matrix.

> **Sample-size caveat:** N=50/cell, 2 domains (GenKnowledge, CustomerSupport). Directions that
> repeat are trustworthy; exact values are noisy. One component changes at a time, so any score
> change is attributable to that single change.

---

## Pooled Exp 1 — the baseline (`grounded_pooled_norerank`)

**Question:** how does our pipeline do in *real* RAG — searching the whole domain corpus instead of
being handed each question's documents? And where is the headroom?

**Setup:** the per-example `grounded_norerank` config with ONLY `index.corpus_mode: pooled`. Full
domain corpus indexed once (GenKnowledge ≈ 7,813 unique docs → ≈ 8,164 chunks; CustomerSupport
smaller); each of N=50 questions retrieves dense top-k=20 → top-5 → grounded prompt → `llama3.2:3b`,
judged by `llama3.1:8b`. Source: `results/ragbench_matrix_n50_pooled.csv`.

### Baseline numbers

| metric | GenKnowledge | CustomerSupport |
|---|---|---|
| relevance | 0.304 | 0.205 |
| utilization | 0.128 | 0.069 |
| completeness | 0.363 | 0.121 |
| adherence | 0.540 | 0.500 |

### Diagnosis — WHERE the gap is (measured, not assumed)

We probed the retriever directly before drawing conclusions (the same discipline as Exp 6A). On
GenKnowledge:

- **Recall is fine.** ≥1 gold chunk appears in the top-5 for **12/12** probed questions, and in the
  top-20 for 12/12. *We are finding the right documents.* So the gap is NOT a recall failure.
- **Granularity is fine.** The chunker is per-document and GenKnowledge docs are short (~86-word
  median ≈ one chunk each); shrinking chunk size barely changes the index (512w→8,164 chunks,
  128w→10,663). *So chunk size is NOT the lever here.*
- **It's a top-5 PRECISION problem.** Gold ranks high but the tail of the top-5 is noise:
  - top-1 is a gold chunk **90%** of the time,
  - top-3 precision **72%** (2.15 gold / 3),
  - top-5 precision **58%** (2.90 gold / 5).
  We hand the generator 5 chunks when only ~2–3 deserve to be there. The 2 padding chunks drag
  relevance down (more irrelevant sentences in the denominator) AND give the small generator
  off-topic material — which also explains the low utilization/completeness.

**CustomerSupport is worse on every metric** (relevance 0.20, completeness 0.12) — the harder
domain, and its longer, more-similar support docs are easier to confuse in a big shared pool.

### What this implies for improvement (the levers worth testing)

Because the gap is *precision of the top-5*, not recall or chunking, the candidate levers are:
1. **Smaller `top_n` (5 → 3):** feed only the higher-precision top-3 (72% vs 58%). Cheapest;
   directly targets the measured gap. **← tested next (Pooled Exp 2).**
2. **Reranker ON:** push the gold chunks up so the top-5 is cleaner (and test if Exp 6B's
   prompt×reranker synergy transfers to pooled). *Parked — one change at a time.*
3. **BGE embedder:** the honest re-test of the Exp 5 null (a stronger embedder *should* matter more
   in a big haystack — fewer look-alike neighbours). *Parked.*

Levers explicitly **de-prioritised** by the diagnosis: chunk-size (docs already ~atomic),
recall-boosters like bigger k (recall is already fine).

---

## Pooled Exp 2 — tighter context (`top_n` 5 → 3)

**Question:** Exp 1 showed the top-5's tail is noise (top-3 precision 72% vs top-5 58%). Does feeding
the generator only the 3 best chunks raise quality? **One-variable change:** only `reranker.top_n`
(5 → 3) differs from `grounded_pooled_norerank`; pooled, N=50, same seed.

### top3 vs top5 (pooled baseline)

| metric | GenKnowledge (5→3) | CustomerSupport (5→3) |
|---|---|---|
| **relevance** | 0.304 → **0.385** (+0.080) | 0.205 → **0.269** (+0.064) |
| utilization | 0.128 → 0.151 (+0.024) | 0.069 → 0.094 (+0.025) |
| completeness | 0.362 → 0.337 (−0.026) | 0.121 → 0.146 (+0.024) |
| **adherence** | 0.540 → **0.408** (−0.132) | 0.500 → **0.420** (−0.080) |

(GenKnowledge top3 scored 49/50 — one judge parse failure — vs 50/50 baseline; the −0.13 adherence
swing is ~6–7 answers, real but with slight single-example wobble. CustomerSupport is 50/50 and clean.)

### Findings + reasoning

**F1 — Relevance/utilization rose exactly as the diagnostic predicted.** Cutting the noisy 4th–5th
chunks raised the relevant-fraction (+0.08 / +0.06 relevance, both domains) and the generator used
more of the tighter context (+0.02 util). The Exp 1 precision diagnosis was correct on its own terms.

**F2 — But adherence DROPPED hard (the unexpected, important result).** GenKnowledge −0.132,
CustomerSupport −0.080. *Reasoning:* fewer chunks = less total context. When the answer needs a fact
that lived in the dropped chunk #4/#5, the small 3B generator fills the gap from parametric knowledge
instead of refusing → that sentence is unsupported → adherence falls. So top_n trades **precision of
what's shown** (relevance ↑) against **coverage of what's needed** (adherence ↓) — a recall/precision
tension at the *context-size* level, and at top_n=3 we cut too much for a gap-filling 3B.

**F3 — Verdict: a TRADE, not a win — and likely a bad one for us.** Adherence (faithfulness) is the
metric the project cares most about; trading −0.13 adherence for +0.08 relevance is a poor deal.
**Keep top_n=5.**

### What Exp 2 implies for the next lever

The diagnosis ("top-5 tail is noise") was right, but the fix can't be "show fewer chunks" — that
starves coverage. The right fix **keeps 5 chunks but makes them higher-precision**, i.e. **rerank the
top-20 and keep the best 5**. That should capture Exp 2's relevance gain *without* the coverage/
adherence loss. → **Pooled Exp 3 = reranker ON** (now data-motivated, not a guess).

---

## Pooled Exp 3 — reranker ON (completing the `{top_n} × {reranker}` 2×2)

**Question:** Exp 2 showed cutting chunks (top_n 5→3) trades adherence for relevance. Does a
cross-encoder reranker instead — *keep* the chunks but make them better — capture the relevance gain
*without* the adherence loss? We ran both reranker cells (top_n 5 and 3) to complete the pooled 2×2.
One-variable changes; pooled, N=50.

### The full pooled 2×2 (mean over N=50)

**GenKnowledge:**

| config | relevance | utilization | completeness | adherence |
|---|---|---|---|---|
| norerank, top5 (baseline) | 0.304 | 0.128 | 0.362 | 0.540 |
| norerank, top3 | 0.385 | 0.151 | 0.337 | 0.408 |
| **rerank, top5** | 0.337 | 0.086 | 0.328 | **0.604** |
| rerank, top3 | **0.435** | 0.159 | 0.324 | 0.520 |

**CustomerSupport:**

| config | relevance | utilization | completeness | adherence |
|---|---|---|---|---|
| norerank, top5 (baseline) | 0.205 | 0.069 | 0.121 | 0.500 |
| norerank, top3 | 0.269 | 0.094 | 0.146 | 0.420 |
| rerank, top5 | 0.165 | 0.051 | 0.102 | 0.440 |
| rerank, top3 | **0.324** | 0.076 | 0.079 | 0.480 |

(Parse failures: norerank-top3 GenK 49/50, **rerank-top5 GenK 48/50** — the 0.604 adherence is over 48,
slight wobble; all other cells 50/50.)

### Findings + reasoning

**F1 — Reranking @top5 is the GenKnowledge win — and the answer to Exp 2's question is YES (there).**
On GenKnowledge, rerank@top5 beat the baseline on **both** relevance (+0.032) **and** adherence
(+0.064 → **0.604, the highest adherence of any pooled config**). This is exactly the "improve the
chunks, don't reduce them" outcome Exp 2 pointed to: the reranker put genuinely better chunks in the
same 5 slots, so the generator grounded better. Unlike top_n=3, it did NOT sacrifice adherence.

**F2 — But reranking @top5 BACKFIRES on CustomerSupport** (relevance −0.039, adherence −0.060). On the
harder domain the cross-encoder's "best 5" were *worse* than the dense top-5 — it promoted
plausible-but-wrong support docs. Same domain split we keep seeing: a lever that helps GenKnowledge
stalls or hurts CustomerSupport. **Reranking is not universally good in pooled mode — it's
domain-dependent.**

**F3 — At top_n=3, reranking RESCUES the adherence that plain top3 lost.** rerank-top3 vs norerank-top3:
adherence +0.112 (GenK) / +0.060 (CustS), relevance also up. Reasoning: the reranker's best-3 are
better-chosen than the dense top-3, so even at 3 chunks they cover more of what the answer needs —
confirming the loss in Exp 2 was about chunk *quality*, not just count.

**F4 — Picking a pooled winner: adherence-first → rerank@top5 on GenKnowledge.** rerank@top3 has the
highest *relevance* in both domains (0.435 / 0.324), but lower adherence. Since adherence (faithfulness)
is the metric the project prioritises, **`grounded_pooled_rerank` (rerank, top5) is the most defensible
pooled config** — the only one that beat the baseline on both relevance and adherence (GenKnowledge).
CustomerSupport stays stubborn: no pooled config clearly wins it (its best adherence is still the plain
baseline's 0.500).

### Caveats
- N=50; rerank-top5 GenK is 48/50 (parse fails) so its standout 0.604 is "promising, not locked" —
  confirm at larger N. CustomerSupport cells are clean (50/50); its negative reranker effect is solid.
- Plain `grounded` prompt throughout — this does NOT yet test the per-example Exp 6B prompt×reranker
  synergy (that needs `grounded_complete` + rerank in pooled). Candidate next experiment.

**Takeaway:** in a real (pooled) corpus the lever that helps is **reranking — keep the chunk count,
improve the chunks** — and it works *where retrieval is tractable* (GenKnowledge: best-of-both,
adherence 0.604) but backfires on the hard domain (CustomerSupport). A defensible, domain-specific
finding about *when* reranking pays off, not a blanket verdict.

### Pooled track — candidate next levers
1. **`grounded_complete` + rerank in pooled** — does the per-example Exp 6B synergy transfer? (best per-example config)
2. **BGE embedder in pooled** — the honest Exp 5 re-test; might fix CustomerSupport's look-alike retrieval.
3. **Bigger generator on the pooled winner** — does rerank@top5's GenKnowledge win amplify with capacity?

---

*Data source: `results/ragbench_matrix_n50_pooled.csv`. Per-example track + reference comparison
live in `EXPERIMENTS.md`.*
