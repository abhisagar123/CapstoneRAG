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
>   (`results/pooled/ragbench_matrix_n50_pooled.csv`), never merged with the per-example matrix.

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
judged by `llama3.1:8b`. Source: `results/pooled/ragbench_matrix_n50_pooled.csv`.

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

---

## Pooled Exp 4 — embedder (BGE) vs prompt (grounded_complete), run in parallel

**Question:** two independent one-change levers off the pooled baseline — does a stronger **embedder**
(retrieval-side) or the **completeness prompt** (generation-side) help in pooled mode? Run as two
parallel processes (separate `--out` files), each a clean one-variable swap vs `grounded_pooled_norerank`.
N=50. Sources: `results/pooled/ragbench_matrix_n50_pooled_bge.csv` + `..._complete.csv`.

### 4A — BGE embedder (minilm → bge-base): another NULL

| metric | GenKnowledge Δ | CustomerSupport Δ |
|---|---|---|
| relevance | −0.009 (~flat) | −0.008 (~flat) |
| utilization | −0.026 | −0.012 |
| completeness | +0.011 | +0.025 |
| adherence | 0.000 | −0.020 |

**Finding: no improvement — BGE is now a confirmed null in BOTH per-example (Exp 5) and pooled.**
*Reasoning:* the pooled bottleneck was never "the embedder retrieves the wrong docs" — Exp 1 showed
recall was already fine (gold in top-5 for 12/12). A different dense embedder reshuffles near-duplicates
but doesn't fix top-5 *precision*, so TRACe barely moves. **Swapping dense embedders is not our lever**
(repeated across two corpus modes — a solid report point).

### 4B — grounded_complete prompt (grounded → grounded_complete): the POOLED WINNER

| metric | GenKnowledge Δ | CustomerSupport Δ |
|---|---|---|
| relevance | +0.017 (~flat) | +0.026 |
| **utilization** | **+0.059** | **+0.047** |
| **completeness** | **+0.055** | **+0.127** |
| adherence | −0.009 (~flat) | **+0.040** |

**Finding: the strongest pooled lever found — and a CLEAN win, no trade.** The completeness prompt lifted
utilization and completeness in **both** domains, with **no adherence cost** (flat GenK, +0.040 CustS).
Two things make it stand out:
- **It moved the worst pooled metric:** CustomerSupport completeness 0.121 → 0.249 (more than doubled).
- **It's the FIRST lever to help CustomerSupport** — the domain that stonewalled the reranker (Exp 3
  backfired there), top_n, and BGE. Every retrieval-side lever failed on CustomerSupport; the
  generation-side prompt succeeded.

*Reasoning:* reranker/BGE are *retrieval* levers, but pooled recall was already fine — the real
bottleneck was the **generator under-using** the retrieved context (low util/completeness, Exp 1).
grounded_complete is a *generation-side* instruction ("use every relevant detail"), so it hits the
actual bottleneck. Note the contrast with per-example **Exp 4**, where grounded_complete *traded*
−0.04 adherence for its completeness gain; **in pooled mode there's no such trade** (more retrieved
context to legitimately draw on → being thorough doesn't push it into ungrounded territory).

**Caveat:** GenKnowledge complete = 49/50 (one parse fail), minor wobble; CustomerSupport clean 50/50. N=50.

### Charts
Pooled configs compared among each other (one chart per domain, 4 TRACe bars per config):
`results/pooled/figures/matrix_{genknowledge,customersupport}.png`, regenerated by
`scripts/plot_pooled.py` (merges all `results/pooled/*.csv`). No reference bars — pooled is a
configs-vs-each-other track.

---

## Pooled Exp 5 — building on the winner: complete + reranker (5A) vs complete + top3 (5B)

**Question:** `grounded_complete` (Exp 4B) is the pooled prompt winner. Two one-change extensions of it,
run in parallel: **5A** adds the reranker (combine with GenKnowledge's lever, Exp 3); **5B** tightens
context to top_n=3. Each is one change off `grounded_pooled_complete`; N=50. Sources:
`results/pooled/ragbench_matrix_n50_pooled_complete_rerank.csv` + `..._complete_top3.csv`.

### 5A — complete + reranker: did NOT compound — it AVERAGED (synergy did not transfer)

complete+rerank lands *between* its two parents on nearly every metric — worse than complete where
complete was strong, worse than rerank where rerank was strong:

| metric | combo | vs complete-alone | vs rerank-alone |
|---|---|---|---|
| GenK completeness | 0.348 | −0.070 | +0.020 |
| GenK adherence | 0.551 | +0.020 | −0.053 |
| **CustS completeness** | 0.123 | **−0.126** | +0.021 |
| CustS adherence | 0.500 | −0.040 | +0.060 |

**Finding: the two levers FIGHT, they don't stack.** `grounded_complete` says "use *more* of the
context"; the reranker→top5 *narrows and reorders* what's available. Being thorough over a
reranked-narrowed set gives a smaller pool to cover → completeness drops (it erased complete's
signature CustomerSupport completeness win, 0.249→0.123). **The per-example Exp 6B prompt×reranker
synergy did NOT transfer to pooled** — a synergy real in one corpus regime broke in another (a
report-worthy negative result). complete+rerank is a wash; not the pooled winner.

### 5B — complete + top3: the GenKnowledge STANDOUT (best pooled completeness by far)

| metric | GenKnowledge (vs complete) | CustomerSupport (vs complete) |
|---|---|---|
| **completeness** | 0.418 → **0.655** (+0.237) | 0.249 → **0.318** (+0.069) |
| utilization | 0.187 → 0.313 (+0.126) | 0.115 → 0.142 |
| relevance | 0.322 → 0.402 (+0.080) | 0.231 → 0.269 |
| adherence | 0.531 → 0.540 (held) | 0.540 → **0.320 (−0.220)** |

**Finding: tight context + "use it fully" COMPLEMENT on GenKnowledge.** GenKnowledge completeness
**0.655 is by far the highest of any pooled config** (next best 0.42) with adherence *held flat* —
fewer chunks (top3) makes "be thorough" achievable, and grounded_complete cancels top3's usual
coverage-starvation (the Exp 2 adherence loss). **But it's domain-split:** on CustomerSupport the
top3 precision/coverage tension bit hard (adherence 0.540 → 0.320), same failure mode as plain top3
(Exp 2). So 5B is a strong GenKnowledge win, a bad CustomerSupport trade.

⚠️ **Caveat:** GenK completeness 0.655 is a large jump on the noisiest (derived) metric at N=50 —
direction is trustworthy (top3+complete lifts coverage on both domains), exact value wants a sanity
re-run before being cited as final. GenK 5A = 49/50 (one parse fail).

### Best pooled config on the small generator (3B) — it's DOMAIN-SPLIT (the finding)

No single config wins both domains; the per-domain bests are:

| domain | best config | headline |
|---|---|---|
| **GenKnowledge** | **`complete + top3`** (5B) | completeness 0.655 (dominant), adherence held |
| **CustomerSupport** | **`complete` (top5)** (4B) | only config with decent completeness (0.249) AND adherence (0.540); every top3/rerank variant wrecks its adherence |

CustomerSupport stays stubborn across the whole track: reranker (Exp 3), top3 (Exp 2), BGE (4A), and
now complete+rerank (5A) and complete+top3 (5B, adherence) all failed it. Only the plain
generation-side `grounded_complete` (4B) helps it. This is a real, repeated domain-difficulty finding.

---

## Pooled Exp 6 — Hybrid retrieval (dense + BM25 via RRF): a decisive NEGATIVE result

**Question:** every config lever failed on CustomerSupport; does *hybrid retrieval* (dense + BM25, the
"Best Practices" paper's top retrieval pick) finally crack it? We built BM25 + a weighted-RRF hybrid
(see `EXPERIMENTS`/code) and ran two weightings — one change each off the dense baseline / each other.
N=50. Sources: `..._hybrid.csv` (equal RRF) + `..._hybrid_dense_heavy.csv` (0.7/0.3).

### Three-way: dense → hybrid (1/1) → hybrid (0.7/0.3 dense-leaning)

**GenKnowledge:**

| metric | dense | hybrid 1/1 | hybrid .7/.3 |
|---|---|---|---|
| relevance | **0.304** | 0.228 | 0.280 |
| utilization | **0.128** | 0.082 | 0.089 |
| completeness | **0.362** | 0.306 | 0.300 |
| adherence | **0.540** | 0.531 | 0.449 |

**CustomerSupport:**

| metric | dense | hybrid 1/1 | hybrid .7/.3 |
|---|---|---|---|
| relevance | **0.205** | 0.177 | 0.178 |
| utilization | **0.069** | 0.042 | 0.045 |
| completeness | **0.121** | 0.090 | 0.112 |
| adherence | 0.500 | 0.480 | **0.531** |

### Findings + reasoning

**F1 — Hybrid does NOT beat dense, under EITHER weighting.** Plain RRF (1/1) hurt everything
(relevance −0.076 GenK / −0.028 CustS). Dense-leaning (0.7/0.3) clawed *back toward* baseline
(GenK relevance 0.228→0.280) — confirming BM25 was diluting — but never *past* it; the best case is
"≈ dense." The lone positive cell is CustomerSupport adherence 0.531 (vs dense 0.500), but it's
±a couple of answers at N=50 (and that cell scored 49/50), so it's within noise.

**F2 — The instructive part: BM25 RAISED recall, yet TRACe DROPPED (recall ≠ TRACe-relevance).** A
diagnostic on CustomerSupport found BM25-alone recalled *more* gold chunks in the top-5 than
dense-alone (2.83 vs 2.25), and hybrid the most (2.92) — retrieval recall went UP. But our **relevance
= relevant-sentences / total-retrieved-sentences** is a precision-of-text metric, not recall. BM25
favors longer, keyword-dense chunks (median retrieved length 390→401 words), which inflates the
denominator: more gold present, spread over more total text → the relevant *fraction* falls. (The
length effect is small ~3% so it only partially explains the −0.076; the rest is BM25 surfacing
lexically-matched-but-not-judged-relevant chunks + N=50 noise — not fully attributable, stated honestly.)

**F3 — Why this is a valuable negative result.** Hybrid delivered exactly what the paper promises
(higher recall) and still hurt *our* metric. The paper recommends hybrid because it optimizes
recall/nDCG; TRACe relevance measures something different. This is the cleanest demonstration in the
project that **a paper-endorsed retrieval lever does not transfer to a precision-of-retrieved-text
metric** — closes the retrieval-lever line with evidence, not a hunch.

---

## Pooled Exp 7 — Chunking sweep (size + PGC): the dilution hypothesis, confirmed asymmetrically

**Question:** chunking was the ONE component never varied (all prior runs used fixed 512/50). The Exp 1
diagnostic argued big chunks dilute relevance (= relevant-sents / total-retrieved-sents). Does
smaller / structure-aware chunking raise it? One-change off the dense pooled baseline: fixed-256,
fixed-128, and **PGC** (paragraph-group, the "Document Chunking Strategies" paper's best chunker).
N=50. Sources: `..._chunk256.csv`, `..._chunk128.csv`, `..._pgc.csv`.

### The sweep (relevance / completeness; full 4-metric in CSVs)

**GenKnowledge** — chunking barely moves anything:

| chunker | relevance | completeness | adherence |
|---|---|---|---|
| fixed-512 (base) | 0.304 | 0.362 | 0.540 |
| fixed-256 | 0.289 | 0.360 | 0.580 |
| fixed-128 | 0.322 | 0.282 | 0.460 |
| pgc | 0.314 | 0.375 | 0.620 |

**CustomerSupport** — chunking is a REAL lever, clean monotonic trend:

| chunker | relevance | completeness | adherence |
|---|---|---|---|
| fixed-512 (base) | 0.205 | 0.121 | 0.500 |
| fixed-256 | 0.272 | 0.121 | 0.480 |
| fixed-128 | 0.315 | 0.179 | 0.440 |
| **pgc** | **0.372** | **0.256** | 0.460 |

### Findings + reasoning

**F1 — GenKnowledge doesn't care about chunking.** Relevance wobbles 0.29–0.32 with no trend;
fixed-128 actively *hurt* (over-fragmented: completeness 0.362→0.282, adherence −0.08). PGC is
marginally best but within N=50 noise of baseline. Matches EDA: GenKnowledge docs are short /
pre-segmented, so re-chunking has little to fix. Chunking is **not** GenKnowledge's lever.

**F2 — CustomerSupport: chunking is the biggest lever found for it, and the dilution hypothesis is
CONFIRMED.** Finer chunks → monotonically higher relevance (0.205→0.272→0.315→**0.372**) and
completeness (→**0.256**, doubled). **PGC wins decisively: relevance +0.167 (+81%), completeness
×2.1** vs the 512 baseline. CustomerSupport's long support docs were burying relevant sentences in
off-topic text inside big 512-word chunks; PGC's paragraph-coherent chunks isolate them, so the
relevant-FRACTION jumps. This is the first *retrieval-side* win on CustomerSupport (every other
retrieval lever — embedder, reranker, hybrid — failed it).

**F3 — The honest cost: a small adherence trade.** PGC's CustomerSupport adherence is 0.460 (vs 0.500
baseline). Finer chunks trade a little faithfulness for big relevance/completeness gains — a *small*
trade for a *large* gain, and within N=50 noise (±0.04). Not a free win, but a strong one.

**F4 — recall-vs-TRACe footnote:** unlike hybrid (Exp 6), which raised recall but *lowered* the
relevant-fraction (longer chunks), PGC raises the relevant-fraction directly by making chunks
*shorter & coherent* — the opposite, complementary mechanism. Consistent story: on our metric, chunk
PRECISION (relevant-fraction) is what moves CustomerSupport, and chunk size/structure is the knob.

---

## Pooled Exp 8 — PGC + grounded_complete: the levers STACK (orthogonal-stage combine)

**Question:** PGC (Exp 7, CustomerSupport's retrieval-side winner) and grounded_complete (Exp 4B, its
prompt winner) act on DIFFERENT stages — does combining them stack (beat both parents) or merely
average (like 5A's complete+rerank)? One-change off each parent. N=50.
Source: `..._pgc_complete.csv`.

### Stack test vs both parents

**CustomerSupport:**

| metric | pgc | complete | pgc+complete | verdict |
|---|---|---|---|---|
| relevance | 0.372 | 0.231 | 0.373 | keeps PGC's win |
| utilization | 0.122 | 0.115 | **0.213** | **STACKS (> both)** |
| completeness | 0.256 | 0.249 | **0.351** | **STACKS (> both)** |
| adherence | 0.460 | 0.540 | 0.440 | **WORSE (< both)** |

**GenKnowledge:** util 0.143/0.187→**0.218** and completeness 0.375/0.418→**0.509** both STACK;
relevance/adherence ≈ average.

### Findings + reasoning

**F1 — The levers genuinely STACK on utilization + completeness (both domains).** CustomerSupport
completeness hits **0.351 — the highest of ANY pooled config** (prev best complete_top3 0.318), while
keeping PGC's relevance win (0.373, also the highest). *Reasoning (the hypothesis, confirmed):* the two
act on orthogonal stages — PGC gives the generator better *material* (coherent on-topic chunks),
grounded_complete makes it *use that material fully* — so they compound. Contrast 5A (complete+rerank),
which *averaged* because both touched retrieval and reshuffled the same pool. **Orthogonal levers stack;
overlapping levers average** — a clean, report-worthy mechanism.

**F2 — The honest cost: adherence dropped BELOW both parents** (CustomerSupport 0.440 vs pgc 0.460,
complete 0.540). Same precision/coverage tension: "be thorough" over PGC's many fine-grained chunks
pulls in more material than the 3B can reliably ground. 0.440 vs 0.460 is ~noise; 0.440 vs complete's
0.540 is a real −0.10. So it's a *trade*, not a free win.

**F3 — So which is CustomerSupport's winner? It depends on the metric you weight (document, don't hide):**
- maximize **completeness / coverage** → `pgc_complete` (0.351, best ever) — decisively.
- maximize **adherence / faithfulness** → plain `complete` (0.540) still wins.
This per-domain, per-metric trade-off IS the finding — there is no single dominating CustomerSupport config.

> ⚠️ **INFRA NOTE (14 Jun):** the PGC run first SEGFAULTED on CustomerSupport (14,968 chunks). Root
> cause: faiss-cpu + PyTorch each bundle their own OpenMP runtime; loaded together they trip
> `OMP: Error #15` → hard crash during faiss search under load. Fixed in `src/__init__.py`
> (`KMP_DUPLICATE_LIB_OK=TRUE` + `OMP_NUM_THREADS=1`, set before faiss/torch load). Any large pooled
> index would hit this — teammates included.

---

## Pooled Exp 9 — stronger embedder (mxbai-large) on plain + PGC: the embedder lever is EXHAUSTED

**Question:** every prior embedder swap (BGE, per-example Exp 5 + pooled Exp 4A) was a null vs MiniLM —
but all used fixed-512 chunks: big, noisy chunks where a better embedder has little to discriminate. PGC
(Exp 7) makes topically-coherent chunks. **Hypothesis: on coherent chunks, embedding QUALITY may finally
matter** — mxbai could beat MiniLM on top of PGC even though it was null on fixed chunks. Tested
`mxbai-embed-large-v1` (1024-dim) as a one-change swap on BOTH the plain pooled baseline AND on top of
PGC, N=50. Sources: `..._mxbai.csv`, `..._pgc_mxbai.csv`.

> ⚠️ **Config-only caveat:** mxbai documents a query *prompt* ("Represent this sentence for searching…")
> for best retrieval. We embed it **un-prompted** (symmetric `encode()`) to keep this config-only — adding
> the prompt is the same `embed_query()` plumbing e5 needs (deferred). So this measures mxbai *without* its
> prompt. Since BGE needs no prompt and was *also* null, the prompt is unlikely to flip the conclusion.

### Deltas vs the matching MiniLM baseline (same chunker — a clean one-variable swap)

**GenKnowledge:**

| config | rel | util | compl | adh | Δrel vs MiniLM |
|---|---|---|---|---|---|
| MiniLM plain (baseline) | 0.304 | 0.128 | 0.362 | 0.540 | — |
| BGE plain (Exp 4A) | 0.296 | 0.101 | 0.373 | 0.540 | −0.009 |
| **mxbai plain** | 0.290 | 0.108 | 0.293 | 0.500 | **−0.014** |
| MiniLM PGC (baseline) | 0.314 | 0.143 | 0.375 | 0.620 | — |
| **mxbai PGC** | 0.309 | 0.129 | 0.345 | 0.531 | **−0.006** |

**CustomerSupport:**

| config | rel | util | compl | adh | Δrel vs MiniLM |
|---|---|---|---|---|---|
| MiniLM plain (baseline) | 0.205 | 0.069 | 0.121 | 0.50 | — |
| BGE plain (Exp 4A) | 0.197 | 0.057 | 0.147 | 0.48 | −0.008 |
| **mxbai plain** | 0.190 | 0.076 | 0.166 | 0.46 | **−0.015** |
| MiniLM PGC (baseline) | 0.372 | 0.122 | 0.256 | 0.46 | — |
| **mxbai PGC** | 0.367 | 0.149 | 0.225 | 0.56 | **−0.005** |

### Findings + reasoning

**F1 — INFORMATIVE NULL: relevance did not improve in a single cell** (−0.005 to −0.015 everywhere),
*including the case we cared about* (mxbai-on-PGC: −0.006 GenK / −0.005 CustSupport). The "coherent chunks
let embedding quality matter" hypothesis is **rejected** — mxbai on PGC is no better than MiniLM on PGC.

**F2 — The embedder lever is now EXHAUSTED by triangulation.** Two stronger dense embedders (BGE 768-dim,
mxbai 1024-dim) × two chunkers (fixed-512, PGC) × two domains — **none beats the little 384-dim MiniLM on
TRACe relevance.** The dense embedder is not our lever, full stop. It joins reranker and hybrid (Exp 6) on
the pile of paper-endorsed *retrieval* levers that don't transfer to our *whole-pipeline-text* metric.

**F3 — Why (consistent with Exp 1's recall diagnostic).** Recall was already fine (Exp 1: gold in top-5
for 12/12 Qs). A better embedder improves *ranking* (MRR/nDCG — what the embedding papers optimize), but
our metric is relevance = relevant-FRACTION of retrieved *text*. With a small, on-topic pool, all dense
models grab roughly the same near-duplicate chunks; reshuffling their order doesn't change the fraction.
Same recall-≠-TRACe lesson as hybrid (Exp 6).

**F4 — The one tempting number, and why it's NOT a finding.** CustomerSupport mxbai/PGC adherence **+0.100**
is the lone positive. Resist it: (a) at N=50 each flipped answer = 0.020, so +0.100 = **5 answers** —
squarely in the noise band that produced the false Exp 3 F3 reranker "win"; (b) it's *contradicted* by the
same swap on GenK (**−0.089**) — a real effect wouldn't swing opposite by domain; (c) adherence is
generation-side, which an embedder only touches second-hand → the noisiest possible embedder signal. A
watch-item for a confirmation re-run, not a result.

### Pooled track — status + next
1. ✅ 4A BGE null · 4B complete · 5A complete+rerank (averaged) · 5B complete+top3 (GenK win) ·
   6 hybrid (negative) · 7 chunking (PGC = CustomerSupport retrieval win) · 8 pgc+complete (STACKS on
   coverage; trades adherence) · **9 mxbai embedder (null; embedder lever exhausted).**
2. **Per-domain bests (unchanged by Exp 9):** GenKnowledge → `complete+top3` (compl 0.655). CustomerSupport →
   **`pgc_complete` if optimizing coverage (compl 0.351, rel 0.373)**, or plain `complete` if optimizing
   adherence (0.540). GenKnowledge wants generation-side levers; CustomerSupport wants PGC chunking +
   the completeness prompt — a clean, documented domain split.
3. **3B levers now FULLY explored** (chunking, embedder ×2, retriever, reranker, hybrid, prompt, top_n,
   orthogonal combine). Every *retrieval-side* lever that helps does so via chunk PRECISION (PGC);
   stronger embedders / rerankers / hybrid all fail. The persistent gaps (adherence, completeness) are
   **generation-side**. **Next = bigger generator** (`--gen-model llama3.1:8b`) on the per-domain winners —
   does generation-side capacity lift the adherence the coverage combos traded away? (User's order:
   small-gen levers first — now done — then scale up.) Low-odds leftover: extractive prompt; prompted-mxbai
   (only if a reason to revisit embedders appears).

---

## Pooled Exp 10 — Semantic chunking (cut on MEANING): the PGC mechanism GENERALIZES, but the gains are a chunk-size artifact

**Question:** PGC (Exp 7) was our only retrieval-side win, and only on CustomerSupport. Was that a
one-off quirk of *paragraph* structure, or is there a general lever "smaller, coherent chunks raise
the relevant-FRACTION"? Semantic chunking is the clean test: it cuts on a **completely different
signal** — embed every sentence, cut where neighbour cosine similarity drops (topic shift) — so if it
reproduces PGC's pattern, the lever is "coherent chunks," not "PGC specifically." Ran three granularity
settings as one-change swaps off the pooled baseline (chunker fixed-512 → semantic), N=50:
`percentile-90` (coarse), `percentile-80` (finer), `absolute-0.5` (finest). Sources:
`..._semantic_p90.csv`, `..._semantic_p80.csv`, `..._semantic_abs.csv`. All 50/50 scored, 0 parse fails.

### The sweep vs the chunking baselines (plain `grounded` prompt — apples-to-apples)

**CustomerSupport** (relevance climbs monotonically as cuts get finer):

| config | median words/chunk | rel | util | compl | adh |
|---|---|---|---|---|---|
| fixed-512 (baseline) | 289 | 0.205 | 0.069 | 0.121 | 0.500 |
| fixed-128 | ~70 | 0.315 | 0.084 | 0.179 | 0.440 |
| PGC (prior best) | 30 | 0.372 | 0.122 | 0.256 | 0.460 |
| semantic p90 | 91 | 0.270 | 0.117 | 0.154 | 0.500 |
| semantic p80 | 54 | 0.316 | 0.134 | 0.257 | 0.480 |
| **semantic abs-0.5** | **19** | **0.587** | **0.222** | 0.230 | 0.400 |

**GenKnowledge** (same shape; chunking is otherwise flat here — short docs, median 84 w/doc):

| config | median words/chunk | rel | util | compl | adh |
|---|---|---|---|---|---|
| fixed-512 (baseline) | 86 | 0.304 | 0.128 | 0.362 | 0.540 |
| PGC | — | 0.314 | 0.143 | 0.375 | 0.620 |
| semantic p90 | — | 0.406 | 0.137 | 0.310 | 0.480 |
| semantic p80 | — | 0.393 | 0.157 | 0.369 | 0.480 |
| **semantic abs-0.5** | 20 | 0.583 | 0.224 | 0.420 | 0.520 |

### Findings + reasoning

**F1 — the PGC mechanism GENERALIZES (the conceptual win).** A cut rule with nothing in common with
PGC (sentence embeddings vs blank lines) reproduces the same pattern on *both* domains: finer, coherent
chunks → higher relevance + utilization. So the lever is **"small coherent chunks raise the
relevant-fraction," not "PGC specifically."** PGC was not a one-off. Report-worthy: two independent
chunkers, one mechanism.

**F2 — but most of the relevance jump is a chunk-SIZE ARTIFACT, not new retrieval quality** (the Exp 6
hybrid lesson, in reverse). TRACe relevance = relevant-sentences / **total-retrieved-sentences**. Tiny
chunks shrink the denominator mechanically. Measured on the real corpus: PGC median 30 w/chunk →
semantic-abs **19** (1.58× finer); CustomerSupport relevance 0.372 → 0.587 (**1.58× higher**) — the
ratios match almost exactly, so the gain is denominator, not quality. Where hybrid (Exp 6) made chunks
*longer* and relevance *fell*, semantic-abs makes them *shorter* and relevance *rises* — same metric
mechanic, opposite sign. (`abs-0.5` also emits ~7% near-empty ≤2-word fragments — it over-cuts.)

**F3 — adherence, the gap we actually care about, does NOT improve.** Flat on GenK (0.52 vs baseline
0.54), **down** on CustomerSupport (0.40 vs 0.50). Adherence is a boolean grounding rate — *not*
confounded by chunk size — so it is the trustworthy signal here, and it says: finer chunks → less
coverage per chunk → the 3B fills gaps from parametric knowledge → unsupported sentences. Same trade as
top3 (Exp 2) and PGC (Exp 7). Semantic chunking **joins the pile** of retrieval-side levers (reranker,
BGE, mxbai, hybrid) that move relevance/util but leave the adherence gap untouched.

**F4 — the one signal that ISN'T just smallness:** semantic-**p80** matches PGC's completeness (0.257 vs
0.256) at *coarser* granularity (54 vs 30 w/chunk) — i.e. coherence buys a little real coverage beyond
being small. And completeness (denominator = relevant set, not all-retrieved → far less confounded) rose
on GenK abs (0.362 → 0.420). Soft, but real. It's why semantic chunking is a legitimate *capability*,
not a measurement trick — just not a fix for our bottleneck.

**Bottom line:** semantic chunking confirms the chunk-precision lever generalizes, but it neither beats
PGC where it counts nor closes the adherence/completeness gap. **Per-domain bests UNCHANGED:** GenK →
`complete+top3` (compl 0.655); CustomerSupport → `pgc_complete` (coverage 0.351) / plain `complete`
(adherence 0.540). The persistent gap is generation-side — the generator is the one lever held fixed at
3B across all 10 experiments.

---

## Pooled Exp 11 — Reranker-rescue on the per-domain coverage winners: REJECTED (the 3B is exhausted)

**Question:** our best-coverage config in each domain trades away adherence — GenKnowledge
`complete_top3` (completeness 0.655, project-best) sits at adherence 0.540; CustomerSupport
`pgc_complete` (completeness 0.351, best) dropped to adherence 0.440. The reranker-rescue is PROVEN
once: plain `top3` adherence 0.408 → `top3_rerank` **0.520 (+0.11)** — re-scoring the kept chunks made
them better-grounded. **Hypothesis: stack that rescue onto the coverage winners** — add the
cross-encoder so the model grounds a thorough answer on *better* chunks, recovering adherence without
losing coverage. Two one-change swaps (reranker none → cross_encoder), one per domain, N=50. Sources:
`..._complete_top3_rerank.csv` (GenK), `..._pgc_complete_rerank.csv` (CustSupport). Both 50/50, 0 fails.

### vs each parent (the no-reranker config it tries to rescue)

**GenKnowledge** — `complete_top3` + cross_encoder@top3:

| metric | parent `complete_top3` | + reranker | Δ |
|---|---|---|---|
| relevance | 0.402 | 0.453 | +0.051 |
| utilization | 0.313 | 0.266 | −0.047 |
| **completeness** | **0.655** | **0.450** | **−0.205** |
| adherence | 0.540 | 0.560 | +0.020 |

**CustomerSupport** — `pgc_complete` + cross_encoder@top5:

| metric | parent `pgc_complete` | + reranker | Δ |
|---|---|---|---|
| relevance | 0.373 | 0.368 | −0.005 |
| utilization | 0.213 | 0.238 | +0.025 |
| completeness | 0.351 | 0.383 | +0.032 |
| adherence | 0.440 | 0.440 | 0.000 |

### Findings + reasoning

**F1 — the rescue is REJECTED in both domains.** Adherence was the bar to clear; at N=50 each answer =
0.020, so the noise band is ±0.02–0.04. GenK +0.020 = **one answer = noise**; CustSupport **0.000 =
flat**. Neither recovered the traded-away adherence. **Per-domain bests UNCHANGED:** GenK stays plain
`complete_top3` (compl 0.655 ≫ the reranked 0.450); CustSupport stays `pgc_complete`/`complete`.

**F2 — on GenK the reranker actively HURT the thing that config is FOR.** Completeness 0.655 → 0.450 (a
clear, far-beyond-noise drop), dragging the project's best-coverage config down to mid-pack. The
cross-encoder re-scores top-20 by query-chunk *ranking* relevance and keeps a different 3 — and those 3
covered less of the judge's relevant-sentence set. Same recall/ranking-≠-TRACe-coverage lesson as hybrid
(Exp 6) and the embedders (Exp 9): rerankers optimize ranking, not coverage of relevant text.

**F3 — WHY the rescue worked on plain grounded but not on grounded_complete (the mechanistic point).**
Under **plain grounded** the model answers narrowly from context, so better chunks (rerank) = more
groundable material = adherence up (the proven +0.11 on top3). Under **grounded_complete** ("be thorough,
use every relevant detail") the model is *already* pushed to over-reach — it pulls in more than the 3B
can ground **regardless of which chunks** it sees, so improving chunk quality can't relieve the pressure.
**The bottleneck under the coverage prompt is the generator's capacity to ground a thorough answer, not
the chunks.** This is the Exp 5A / Exp 8 "overlapping levers average, they don't stack" lesson, third
confirmation: reranker and grounded_complete both act on the context-the-generator-uses, so they
interfere rather than compound.

**Bottom line — the 3B is EXHAUSTED, single levers AND stacks.** Across 11 experiments every within-3B
knob has been swept and every promising stack tested; the persistent adherence/completeness gaps survive
all of them. The one component held FIXED in all 11 is the generator (llama3.2:3b). F3 is now a positive
*mechanistic* reason — not just elimination — that capacity is the bottleneck. **Next = bigger generator
(`llama3.1:8b` / `gemma2:9b`, both pulled) on the per-domain winners** (`complete_top3` GenK,
`pgc_complete` CustSupport): does generation-side capacity ground the thorough answers the 3B couldn't?

---

## Pooled Exp 12 — BIGGER GENERATOR (llama3.1:8b) on the per-domain winners: the lever for CustomerSupport, NOT GenKnowledge

**Question:** the generator (llama3.2:3b) is the ONE component held fixed across Exp 1–11, and Exp 11 F3
gave a positive mechanistic reason it's the bottleneck — under `grounded_complete` the 3B over-reaches
and can't ground a thorough answer regardless of chunks. **Does generation-side capacity fix it?** Ran
`llama3.1:8b` (pulled) on each domain's winner — `complete_top3` (GenK), `pgc_complete` (CustSupport) —
ONLY the generator changed (`--gen-model llama3.1:8b`, same retrieval/seed), so any delta is pure
capacity. Per-example Exp 6A proved answers aren't length-capped → expect BETTER grounding at normal
length, not longer answers. (Bonus: the substring `--configs` filter also ran each winner's `_rerank`
variant on 8B — a free robustness check on Exp 11.) Sources: `..._complete_top3_llama8b.csv`,
`..._pgc_complete_llama8b.csv`. N=50.

### 8B vs 3B (identical retrieval — only the generator changed)

**CustomerSupport — `pgc_complete`:**

| metric | 3B | 8B | Δ |
|---|---|---|---|
| relevance | 0.373 | 0.351 | −0.022 |
| utilization | 0.213 | 0.248 | +0.034 |
| completeness | 0.351 | **0.473** | **+0.122** |
| adherence | 0.440 | **0.600** | **+0.160** |

**GenKnowledge — `complete_top3`** (8B n_scored=49, one skipped):

| metric | 3B | 8B | Δ |
|---|---|---|---|
| relevance | 0.402 | 0.410 | +0.008 |
| utilization | 0.313 | 0.246 | −0.067 |
| completeness | **0.655** | 0.445 | **−0.209** |
| adherence | 0.540 | 0.551 | +0.011 |

### Findings + reasoning

**F1 — CustomerSupport: the bigger generator IS the lever (Exp 11 F3 confirmed).** Adherence +0.160 =
8 answers, far beyond the ±0.04 N=50 noise band, AND completeness +0.122, moving together with relevance
flat — the **first lever in the whole pooled track to lift BOTH persistent gaps at once on
CustomerSupport.** Lands `pgc_complete`@8B at adherence **0.600** and completeness **0.473** — *both the
best of any CustomerSupport pooled config.* The 3B couldn't ground a thorough answer; the 8B can. **New
CustomerSupport winner: `pgc_complete` on llama3.1:8b.**

**F2 — GenKnowledge: capacity did NOT help, and completeness fell.** Adherence flat (+0.011 = noise);
completeness dropped 0.655 → 0.445. The mechanism is legible: completeness = |R∩U|/|R|, and R (relevant
sentences) depends only on the retrieved context, which is **identical** across the two runs (same
chunker/retriever/seed) → the drop is entirely in U: **the 8B utilized FEWER relevant sentences.**
Utilization corroborates (0.313 → 0.246, also down). On GenK the 8B wrote *tighter* answers; on
CustSupport it used *more* context — opposite behaviours, same model.

**F3 — the GenK drop re-frames the 0.655 "project-best completeness."** Exp 5B flagged 0.655 as "a big
jump on a noisy derived metric, wants a sanity re-run." This *is* that independent re-run (only the
generator changed) and **0.655 did not survive.** Likely story: under "be thorough" the 3B padded answers
with extra relevant-sentence text (mechanically inflating completeness), while the stronger 8B answers
more precisely. PLAUSIBLE that the 8B answer is higher quality despite lower completeness — but that
claim needs an answer-text probe (aggregates can't settle it). For now: report both, and treat the 3B
0.655 as partly a thoroughness-padding artifact. **GenK best stays `complete_top3` (3B) with that caveat.**

**F4 — Exp 11's reranker-rejection HOLDS at 8B (free robustness check).** The `_rerank` variants came out
worse on 8B too: CustSupport `pgc_complete_rerank` completeness 0.473→0.263, adherence 0.600→0.420; GenK
`complete_top3_rerank` adherence 0.560→0.480. The rescue fails regardless of generator size — reranker +
grounded_complete interfere at any capacity (Exp 11 F3 confirmed across model sizes).

**Bottom line — DOMAIN-SPLIT, consistent with the whole track.** CustomerSupport → **`pgc_complete` @
llama3.1:8b** is the clean new best (adh 0.600 + compl 0.473, both track-best): PGC chunking + the
completeness prompt + a generator big enough to ground it. GenKnowledge → the generator is NOT the lever;
best stays `complete_top3` (3B), 0.655 completeness caveated as partly artifact. The generator scale-up
closes the CustomerSupport gap and, by *not* closing GenKnowledge, sharpens that GenK's ceiling is set
elsewhere (short docs, already-near-reference relevance — see per-example track). Optional follow-ups:
gemma2:9b as a 2nd big model (confirm CustSupport, retest GenK); an answer-text probe to settle F3.

## Pooled Exp 13 — Extractive prompt (quote, don't rephrase): REJECTED — no adherence gain, and it costs coverage

**Question:** every prior lever leaves the **adherence** gap (faithfulness) untouched, and Exp 12 showed a
bigger generator fixes it on CustomerSupport but NOT GenKnowledge — so on GenKnowledge the unexplored axis is
a *grounding-side prompt* that needs no extra capacity. The `extractive` prompt attacks adherence from a
different angle than `grounded_complete`: instead of "be thorough" (which traded adherence for completeness,
Exp 4), it says *"quote or closely paraphrase the context's own sentences — do NOT rephrase in your own
words."* **Hypothesis:** killing free-rephrasing removes the unsupported-wording failure mode → adherence up,
especially on GenKnowledge. One-change swap (prompt `grounded` → `extractive`) off the Exp 1 pooled baseline,
3B generator, same retrieval/seed, N=50. Source: `..._extractive.csv`. GenK 49/50 scored (one skip), CustS 50/50.

### vs the Exp 1 pooled baseline (grounded @ 3B — same retrieval, ONLY the prompt changed)

**GenKnowledge:**

| metric | baseline `grounded` | `extractive` | Δ |
|---|---|---|---|
| relevance | 0.304 | 0.304 | 0.000 |
| utilization | 0.128 | 0.105 | −0.023 |
| completeness | 0.363 | 0.302 | −0.061 |
| **adherence** | **0.540** | **0.490** | **−0.050** |

**CustomerSupport:**

| metric | baseline `grounded` | `extractive` | Δ |
|---|---|---|---|
| relevance | 0.205 | 0.245 | +0.040 |
| utilization | 0.069 | 0.040 | −0.029 |
| completeness | 0.121 | 0.104 | −0.017 |
| **adherence** | **0.500** | **0.540** | **+0.040** |

### Findings + reasoning

**F1 — the hypothesis is REJECTED: adherence did not rise.** The metric the prompt was built to lift went the
WRONG way on its target domain (GenK −0.050) and rose only at the noise edge on the other (CustS +0.040). At
N=50 each scored answer ≈ 0.020 and the noise band is ±0.02–0.04, so −0.050 ≈ 2–3 answers and +0.040 = 2
answers — and the two domains move in OPPOSITE directions by ~2 answers each, the classic signature of noise,
not a real effect (the same logic that killed the Exp 9 F4 mxbai "adherence +0.10"). Net read: **adherence is
flat; the extractive steer does not reduce the unsupported-sentence rate.**

**F2 — the one CONSISTENT effect is negative: extractive makes answers terser → less coverage.** Utilization
fell in BOTH domains (−0.023 / −0.029) and completeness fell in both (−0.061 GenK / −0.017 CustS) — a
direction-consistent-across-domains move (unlike the see-sawing adherence), so it is the trustworthy signal.
*Reasoning:* told to "quote, don't rephrase," the 3B answers with a narrow snippet or two instead of
synthesizing across the relevant set, so it uses less of the context (util ↓) and covers less of the relevant
set (completeness ↓). The extractive steer changed answer STYLE (terser), not GROUNDING rate (adherence flat).

**F3 — why it failed: the grounding clause was already there.** Plain `grounded` already carries the strict
grounding + refusal clause; the ONLY delta is "quote / don't rephrase." That delta shortens answers but does
not change whether the judge marks sentences supported. So the adherence gap is NOT caused by "free-rephrasing
drift" as the hypothesis assumed — it's the small generator's grounding CAPACITY itself (consistent with
Exp 11 F3 / Exp 12: capacity is the CustomerSupport lever).

**Bottom line — lever REJECTED, per-domain bests UNCHANGED** (GenK `complete_top3` 3B compl 0.655 caveated;
CustS `pgc_complete` @ 8B adh 0.600 / compl 0.473). Importantly this CLOSES the grounding-side-prompt axis on
GenKnowledge: capacity didn't help it (Exp 12) and now a grounding-focused prompt didn't either → GenKnowledge's
ceiling is set on the retrieval/data side (short, already-near-reference docs — per-example track), not by any
generation-side knob. Per the pre-registered gate ("proceed to extractive @ 8B ONLY if adherence moves at 3B"),
we do NOT pursue extractive @ 8B — there is no 3B signal to amplify.

## Pooled Exp 14 — Context compression (Recomp extractive): REJECTED as a lever, but it surfaces the project's FIRST embedding-component win

**Question:** the "Best Practices" paper's recommended recipe (§5.1) includes a **Summarization/Recomp**
stage we had never built — compress retrieved context before generation. Exp 1 showed our top-5 is only
~58% relevant text; the small/over-reaching generator wades through that padding and grounds poorly
(adherence gap, Exp 11/12). **Hypothesis: trim each chunk to its query-relevant sentences (keeping ≥1 per
chunk, so unlike top_n=3 no chunk's facts are lost) → less noise to over-reach on → adherence up.** Built a
`summarization` brick with three arms (none / lexical = word-overlap, no model / extractive_embedding =
cosine(query, sentence)) and added it as the `[summarize]` stage (retrieve→rerank→repack→**summarize**→prompt).
Tested per-domain (per the per-domain-tuning decision) on each domain's WINNER, one-change swaps. N=50.
Sources: `..._ctop3_compress_emb.csv` (GenK, 49/50), `..._ctop3_compress_lex.csv` (GenK control, 50/50),
`..._pgccomplete_compress_emb.csv` (CustSupport, 50/50).

### vs each domain's winner (only the summarizer stage added)

**GenKnowledge** — `complete_top3` @ 3B + compression:

| metric | winner | + emb (cosine) | + lex (word-overlap) |
|---|---|---|---|
| relevance | 0.402 | 0.561 | 0.596 |
| utilization | 0.313 | 0.379 | 0.375 |
| completeness | 0.655 | 0.579 | 0.504 |
| **adherence** | **0.540** | **0.531** (−0.009, flat) | **0.380** (−0.160) |

**CustomerSupport** — `pgc_complete` @ 8B + emb compression:

| metric | winner | + emb (cosine) | Δ |
|---|---|---|---|
| relevance | 0.351 | 0.516 | +0.165 |
| utilization | 0.248 | 0.409 | +0.161 |
| completeness | 0.473 | 0.555 | +0.082 |
| **adherence** | **0.600** | **0.440** | **−0.160** |

### Findings + reasoning

**F1 — REJECTED as a lever: compression does not lift adherence anywhere.** GenK emb = flat (−0.009 = noise);
CustS emb = **crashed −0.160** (8 answers, far beyond the ±0.04 band) for only +0.082 completeness — a bad
trade on our most-valued metric. Per-domain bests **UNCHANGED** (GenK `complete_top3` 0.655 compl; CustS
`pgc_complete`@8B adh 0.600 / compl 0.473). The relevance jumps (+0.159 / +0.165) are largely the chunk-size
DENOMINATOR ARTIFACT (Exp 10): keeping ~half the sentences ~halves total-retrieved-sentences → relevance ≈×1.4.
Not new quality. Adherence (a grounding rate, not size-confounded) is the honest signal, and it says: no gain.

**F2 — the project's FIRST embedding-component win: embedding compression ≫ lexical (adherence 0.531 vs 0.380).**
The model-free control (lexical = keep sentences sharing query WORDS) **actively wrecked adherence** (0.540 →
0.380), while the embedding arm (keep sentences by MEANING) stayed neutral (0.531). *Why:* a supporting
sentence often PARAPHRASES the question rather than repeating its words — lexical drops it and keeps
keyword-matching distractors, so the model can't ground; cosine keeps the real support. Every prior embedding
lever was NULL (BGE, mxbai) — but those were *retrieval*, where our recall was already saturated (Exp 1). In
*sentence selection* meaning genuinely beats keywords. **Report reframing: embeddings matter for SELECTING
text, not for RETRIEVING documents** — sharpens the whole embedder-null story rather than contradicting it.

**F3 — CustS is the THIRD confirmation of the grounded_complete over-reach mechanism.** Under
`grounded_complete` ("use every relevant detail") the model writes a thorough answer; SHRINK the context it can
draw on (compression) and some of those sentences become unsupported → adherence falls. The reranker did the
identical thing (Exp 11 F3); compression now repeats it. Consistent rule: **under the coverage prompt the
bottleneck is the generator's capacity to ground a thorough answer — anything that reduces/reshuffles available
context makes the over-reach worse, it cannot help.** (The symmetric prediction — compression should HELP under
PLAIN `grounded`, where the model answers narrowly — is the one motivated follow-up, untested here.)

**Bottom line — paper's last unbuilt stage is built + swept; it is not our lever.** Compression trades adherence
for coverage/relevance like top3/PGC/semantic before it, and joins that pile. Its lasting value is F2 (embedding
selection beats lexical — our first embedding win) and F3 (over-reach mechanism, triple-confirmed). With every
stage of the paper's recommended pipeline now tested, the only lever that closes the faithfulness gap remains
**generator capacity, and only where the domain supports it** (CustS, Exp 12). The summarization brick (none /
lexical / extractive_embedding) is reusable for the demo + report.

---

*Data sources: `results/pooled/ragbench_matrix_n50_pooled.csv` (Exp 1–3),
`..._bge.csv` + `..._complete.csv` (Exp 4), `..._complete_rerank.csv` + `..._complete_top3.csv` (Exp 5),
`..._hybrid.csv` + `..._hybrid_dense_heavy.csv` (Exp 6),
`..._chunk256.csv` + `..._chunk128.csv` + `..._pgc.csv` (Exp 7), `..._pgc_complete.csv` (Exp 8),
`..._mxbai.csv` + `..._pgc_mxbai.csv` (Exp 9),
`..._semantic_p90.csv` + `..._semantic_p80.csv` + `..._semantic_abs.csv` (Exp 10),
`..._complete_top3_rerank.csv` + `..._pgc_complete_rerank.csv` (Exp 11),
`..._complete_top3_llama8b.csv` + `..._pgc_complete_llama8b.csv` (Exp 12),
`..._extractive.csv` (Exp 13),
`..._ctop3_compress_emb.csv` + `..._ctop3_compress_lex.csv` + `..._pgccomplete_compress_emb.csv` (Exp 14),
`results/pooled/figures/` (charts). Per-example track + reference comparison live in `EXPERIMENTS.md`.*
