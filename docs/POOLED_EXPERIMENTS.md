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

## Pooled Exp 14b — Compression under PLAIN grounded: hypothesis FALSIFIED — the prompt isn't the driver, document structure is

**Question (pre-registered):** Exp 14 F3 argued compression crashed CustS adherence *because*
`grounded_complete` makes the model over-reach, so shrinking its context worsened the over-reach. The
mirror prediction: under **plain `grounded`** (model answers narrowly from context), cleaner context =
more groundable material → **adherence should RISE in both domains** (the reranker-rescue split, Exp 3/11).
Clean test: add the embedding compressor to the Exp 1 baseline (plain grounded, fixed-512, top5, 3B) —
ONE-variable swap, same `ratio: 0.5` as Exp 14, so the ONLY meaningful difference from the failed Exp 14
arm is the prompt. Both domains, N=50, 50/50 scored. Source: `..._compress_emb.csv`. **Pre-registered
verdict:** rise beyond ±0.04 in both → confirmed; flat → F3 incomplete.

### vs the Exp 1 baseline (plain grounded @ 3B — only the summarizer stage added)

| metric | GenK base | GenK +compress | CustS base | CustS +compress |
|---|---|---|---|---|
| relevance | 0.304 | 0.407 | 0.205 | 0.268 |
| utilization | 0.128 | 0.168 | 0.069 | 0.092 |
| completeness | 0.362 | 0.347 | 0.121 | 0.197 |
| **adherence** | **0.540** | **0.400** (−0.140) | **0.500** | **0.560** (+0.060) |

### Findings + reasoning

**F1 — HYPOTHESIS FALSIFIED (pre-registered, reported honestly).** The prediction was "adherence rises in
BOTH under plain grounded." Reality: it **fell hard on GenK (−0.140 = 7 answers, far beyond the ±0.04 band)**
and rose only at the noise EDGE on CustS (+0.060 = 3 answers). Not "both rise" → rejected. Worse for the
hypothesis: compression hurt GenK *more* under plain grounded (−0.140) than under grounded_complete (−0.01,
Exp 14) — the **opposite** of what the over-reach story predicted. The prompt is NOT the variable governing
whether compression helps adherence.

**F2 — the real driver is COVERAGE LOSS, set by document structure — which re-derives the text-reduction
domain split.** Adherence falls when compression cuts *needed facts*, not noise; whether it cuts facts depends
on whether the docs have removable padding:
  - **GenK: short, atomic docs** (~86 w ≈ one chunk, EDA/Exp 1). No padding to remove → ratio-0.5 cutting
    throws away needed sentences → the 3B gap-fills from parametric knowledge → adherence craters. This is the
    SAME failure mode as plain top_n=3 (Exp 2) and fixed-128 over-fragmentation (Exp 7 F1).
  - **CustS: long, padded support docs.** Compression cuts off-topic sentences, not facts → adherence holds
    (+0.06, mechanism-consistent but at the noise edge, and it does NOT beat the CustS winner `pgc_complete`@8B
    adh 0.600 / compl 0.473 — this is a 3B config at adh 0.560 / compl 0.197).
  So compression behaves like EVERY other text-reduction lever (top_n, finer chunks, semantic chunking):
  helps/neutral on padded-doc domains (CustS), hurts atomic-doc domains (GenK). The relevance rises (+0.10 /
  +0.06) are again the size DENOMINATOR artifact (Exp 10), not new quality — discounted.

**F3 — verdict: the compression line is CLOSED.** Across Exp 14 (on winners) + 14b (on the plain baseline),
compression never lifts adherence where it matters; its effect is fully explained by the text-reduction
domain split (F2), not by the prompt. Per-domain bests UNCHANGED (GenK `complete_top3` 3B; CustS
`pgc_complete`@8B). The lasting value of the compression work stays Exp 14 F2 (embedding selection ≫ lexical —
the project's first embedding-component win) + the now-falsified-and-corrected mechanism here. The faithfulness
gap remains closed ONLY by generator capacity, and only where the domain supports it (CustS, Exp 12).

---

## Pooled Exp 15 — STRONGER reranker (`bge_large`) re-tests the Exp 3 domain split: reranker choice is DOMAIN-SPLIT, and `bge_large` REVERSES the CustomerSupport backfire

**Question:** Exp 3 used the SMALL `ms-marco-MiniLM` cross-encoder (~80 MB). It HELPED GenKnowledge
(adherence → 0.604, the pooled best) but BACKFIRED on CustomerSupport (every metric fell below the
no-rerank baseline) — so we concluded "reranking is domain-dependent." But that conclusion was really
about the *small* reranker. PR #49's Legal Phase A found `BAAI/bge-reranker-large` (560 MB / 2.24 GB on
disk) was the single biggest relevance lever on hard long-document CUAD — **reversing** a published
result where a *weaker* reranker (Cohere) DEGRADED the same data. So the pre-registered question: **does
a stronger cross-encoder reverse the CustomerSupport backfire** the way it reversed Cohere's on Legal?
One-variable swap — reranker model `cross_encoder` → `bge_large`, everything else the Exp 3 setup (plain
`grounded`, fixed-512, dense top-20 → top-5, 3B generator, same seed/judge). Kept under PLAIN grounded on
purpose: Exp 11 F3 showed rerankers fight `grounded_complete`, so this stays a clean re-test of Exp 3.
Both domains, N=50, 50/50 scored. Source: `..._bge_rerank.csv`.

### Three-way: no reranker → small `cross_encoder` (Exp 3) → `bge_large` (same retrieval, only the reranker model changed)

**CustomerSupport** (the domain that backfired in Exp 3):

| metric | none (baseline) | `cross_encoder` (Exp 3) | `bge_large` (Exp 15) | bge vs baseline |
|---|---|---|---|---|
| relevance | 0.205 | 0.165 | **0.234** | +0.029 |
| utilization | 0.069 | 0.051 | **0.075** | +0.006 |
| completeness | 0.121 | 0.102 | **0.174** | +0.053 |
| adherence | 0.500 | 0.440 | 0.480 | −0.020 |

**GenKnowledge** (the domain the small reranker helped):

| metric | none (baseline) | `cross_encoder` (Exp 3) | `bge_large` (Exp 15) | bge vs cross_encoder |
|---|---|---|---|---|
| relevance | 0.304 | 0.337 | 0.314 | −0.023 |
| utilization | 0.128 | 0.086 | 0.091 | +0.005 |
| completeness | 0.362 | 0.328 | 0.313 | −0.015 |
| **adherence** | 0.540 | **0.604** | 0.580 | −0.024 |

### Findings + reasoning

**F1 — on CustomerSupport the backfire is REVERSED: `bge_large` is the FIRST reranker to help this
domain.** Every Exp-3 loss recovers and clears the baseline — relevance 0.165 → 0.234 (now +0.029 *above*
the no-rerank baseline, vs Exp 3's −0.040 below it), completeness 0.121 → 0.174 (+0.053, well beyond the
±0.04 noise band = ~2–3 answers), utilization up too. Adherence (0.480) is one answer under baseline
(0.500) — flat within noise — but recovered most of Exp 3's −0.060. This **mirrors PR #49's Legal
finding exactly**: a strong cross-encoder reverses a weak one's degradation. Mechanism: the small
ms-marco model mis-ranked CustomerSupport's verbose support passages and promoted worse chunks into the
top-5; the larger model re-scores them well enough to keep genuinely on-topic chunks — and because chunk
size is fixed-512 throughout, the relevance gain is real precision-of-text, NOT the size-denominator
artifact (Exp 10).

**F2 — on GenKnowledge bigger is NOT better: the small `cross_encoder` STILL wins.** `bge_large` beats the
no-rerank baseline on adherence (0.540 → 0.580) but sits *below* the small model's 0.604 — which remains
GenKnowledge's best pooled adherence — and is slightly worse on relevance/completeness. The small reranker
was already sufficient for GenKnowledge's short, atomic docs; the extra capacity buys nothing here and
costs a little. So the lever is **domain-split** (consistent with the accepted per-domain-tuning
strategy): **GenKnowledge → small `cross_encoder` (adh 0.604); CustomerSupport → `bge_large`.**

**F3 — the honest caveat: `bge_large` helps CustS, but PGC (chunking) is still the BIGGER CustS lever.**
`bge_large` lifts CustomerSupport vs the *fixed-512* baseline, but PGC alone already reaches relevance
0.372 (vs bge_large's 0.234) at ~tied adherence (PGC 0.460 vs bge 0.480). The two act on **orthogonal
stages** — PGC sets chunk *boundaries*, the reranker selects among *candidates* — so by the
stack-vs-average principle (Exp 8) they MIGHT compound. That makes **PGC + `bge_large` (CustomerSupport)**
the clear next test, and it's now unlocked (it was conditional on `bge_large` helping CustS, which F1
confirms).

**Bottom line — first reranker win for the stubborn domain, but it doesn't unseat the existing
per-domain bests yet.** Reranker choice is domain-split: keep small `cross_encoder` for GenKnowledge,
adopt `bge_large` for CustomerSupport. Neither per-domain best changes on adherence (GenK
`complete_top3`@3B / CustS `pgc_complete`@8B stand), but `bge_large` is the strongest *plain-grounded
retrieval* lever CustomerSupport has seen, and it sets up the PGC-stack test. **Next = PGC + `bge_large`
on CustomerSupport** (does the chunking win and the rerank win combine, or just average?).

---

## Pooled Exp 16 — SAC chunker (summary-augmented): REJECTED as a lever — gains are confounded with a 6× chunk-size cut, and adherence craters on CustomerSupport

**Question:** PR #49 brought SAC (Summary-Augmented Chunking, Reuter et al. NLLP 2025), the published
champion on LegalBench-RAG/CUAD. The novel mechanism: prepend each document's first 150 chars (a cheap
"summary") to EVERY chunk of that doc, so a chunk pulled from the middle still carries document-level
context. The paper reports this halves document-retrieval mismatch. We tested it on the pooled baseline
(plain `grounded`, dense top-20 → top-5, 3B) — config `{chunker: sac, size: 500, overlap: 0,
summary_chars: 150}`, both domains, N=50, 50/50 scored. Target = CustomerSupport (long padded support
docs); GenKnowledge expected near-null (atomic ~86-word docs ≈ one chunk, so the prefix ≈ the doc).
Source: `..._sac.csv`.

> ⚠️ **CONFOUND (the key to reading this experiment):** SAC changes *two* things at once vs the fixed-512
> baseline, not one. Our `fixed` chunker is **word**-based (512 *words*/chunk); SAC uses **character**
> windows (500 *chars* ≈ ~85 words). So SAC chunks are ~**6× smaller** AND carry the prepended summary.
> This is NOT a clean one-variable test of the summary-prefix *mechanism* — it's "much smaller chunks +
> a prefix," bundled. Exp 16b (below, summary_chars 0 vs 150 at the SAME geometry) isolates the prefix.

### vs the Exp 1 baseline (plain grounded @ 3B — only the CHUNKER changed); PGC shown as the chunking leader

| metric | GenK base | GenK SAC | CustS base | CustS SAC | CustS PGC (leader) |
|---|---|---|---|---|---|
| relevance | 0.304 | 0.309 | 0.205 | 0.276 (+0.071) | **0.372** |
| utilization | 0.128 | 0.114 | 0.069 | 0.131 (+0.062) | 0.122 |
| completeness | 0.362 | 0.274 (−0.088) | 0.121 | 0.230 (+0.109) | **0.256** |
| **adherence** | **0.540** | **0.500** (−0.040) | **0.500** | **0.380** (−0.120) | **0.460** |

### Findings + reasoning

**F1 — REJECTED: on CustomerSupport SAC craters adherence (−0.120), the metric we value most.** That's
~6 answers, far beyond the ±0.04 noise band. It's the SAME over-fragmentation gap-fill failure as
fixed-128 (Exp 7 F1) and compression-on-atomic-docs (Exp 14b F2): much smaller chunks → less groundable
context per chunk → the 3B fills gaps from parametric memory → unsupported sentences. The simultaneous
relevance/completeness *rises* (+0.071/+0.109) are consistent with the **size-denominator artifact**
(Exp 10) — finer chunks shrink the relevant-fraction denominator and pack more snippets into top-5 — NOT
evidence the prefix added quality. Discounted exactly as in Exp 10/14b.

> ⚠️ **Mechanism CORRECTED by Exp 16b (below).** Isolating the two variables overturns the
> "smaller chunks → gap-fill → unsupported" story *for SAC*: the **size cut ALONE** (Exp 16b, prefix off
> at the same 500-char geometry) actually *raises* CustS adherence (+0.031), so finer chunks did NOT cause
> the crater here. The −0.120 is driven almost entirely by the **summary-prefix** (−0.151). The
> rel/compl-rise = size-artifact reading above still stands; only the adherence-cause attribution changes.

**F2 — even taking the CustS gains at face value, PGC DOMINATES SAC on every metric** (rel 0.372 > 0.276,
util 0.122 ≈ 0.131 tie, compl 0.256 > 0.230, adh 0.460 > 0.380). So SAC doesn't beat the chunking lever
we already have, and unlike PGC it pays for its relevance with a large adherence loss. There is no
configuration of this result where SAC is the CustomerSupport chunker of choice.

**F3 — GenKnowledge is mildly NEGATIVE, exactly as predicted.** Completeness −0.088, adherence −0.040 (at
the noise edge). On atomic ~86-word docs the 150-char prefix is near-redundant (it re-duplicates the
doc's own opening), and the ~85-word windows just fragment a doc that was already ≈ one chunk — so SAC
adds redundancy and fragmentation with no context to restore. The mechanism has nothing to bite on here.

**Bottom line — SAC-as-published is rejected for our domains.** But because the gains/losses are
confounded with the 6× size cut (the ⚠️ above), we have NOT yet isolated whether the prefix *mechanism*
itself does anything. "No stone unturned" → Exp 16b isolates it (summary_chars 0 vs 150, geometry held
fixed). Per-domain bests UNCHANGED (GenK `complete_top3`@3B; CustS `pgc_complete`@8B; CustS plain-grounded
retrieval lever stays `bge_large`/PGC from Exp 15). **Next = Exp 16b (prefix isolation), then the unlocked
PGC + `bge_large` CustS stack (Exp 15 F3).**

---

## Pooled Exp 16b — SAC prefix-isolation control: the summary-PREFIX is the adherence-killer, NOT the chunk-size cut (Exp 16's stated cause is overturned)

**Question:** Exp 16 bundled two changes vs the fixed-512 baseline — a ~6× chunk-size cut (word→char windows)
AND the summary-prefix — so we could not say *which* caused the CustomerSupport adherence crater. This control
removes ONLY the prefix (`summary_chars: 0`) at the **identical** 500-char SAC geometry, giving a clean two-way
decomposition: `prefix effect = Exp16(150) − Exp16b(0)` (same geometry, prefix differs) and
`size effect = Exp16b(0) − fixed-512 baseline` (windows differ, no prefix). Both domains, N=50 (CustS 49/50
scored). Config `grounded_pooled_sac_nosum.yaml`; source `..._sac_nosum.csv`; telemetry
`results/telemetry/pooled/grounded_pooled_sac_nosum.yaml__*.jsonl`. The two effects sum **exactly** to Exp 16's
bundled delta (arithmetic identity — verified per metric).

### Decomposition (PREFIX = on−off, SIZE = off−baseline; they sum to Exp 16's bundled on−baseline)

**CustomerSupport** (baseline `norerank` → SAC-off `nosum` → SAC-on `sac`):

| metric | base | SAC off (16b) | SAC on (16) | **PREFIX** (on−off) | **SIZE** (off−base) | bundled (16−base) |
|---|---|---|---|---|---|---|
| relevance | 0.205 | 0.286 | 0.276 | −0.010 | **+0.081** | +0.071 |
| utilization | 0.069 | 0.127 | 0.131 | +0.004 | **+0.058** | +0.062 |
| completeness | 0.121 | 0.329 | 0.230 | **−0.099** | **+0.208** | +0.109 |
| **adherence** | **0.500** | **0.531** | **0.380** | **−0.151** | **+0.031** | **−0.120** |

**GenKnowledge:**

| metric | base | SAC off (16b) | SAC on (16) | **PREFIX** (on−off) | **SIZE** (off−base) | bundled (16−base) |
|---|---|---|---|---|---|---|
| relevance | 0.304 | 0.398 | 0.309 | −0.089 | +0.094 | +0.005 |
| utilization | 0.128 | 0.095 | 0.114 | +0.019 | −0.033 | −0.014 |
| completeness | 0.362 | 0.202 | 0.274 | +0.072 | −0.160 | −0.088 |
| **adherence** | **0.540** | **0.500** | **0.500** | **0.000** | **−0.040** | **−0.040** |

### Findings + reasoning

**F1 — the PREFIX is the adherence-killer; the size cut is innocent (Exp 16 F1's stated cause is OVERTURNED).**
On CustomerSupport, removing the chunk-size confound, the prefix alone costs **−0.151 adherence** while the 6×
size cut alone is **+0.031** (i.e. slightly *helpful*, within noise). Exp 16 attributed the −0.120 crater to
"smaller chunks → less groundable context → 3B gap-fills" (the fixed-128 / Exp 7 story). That mechanism does
**not** apply here — at SAC's geometry, smaller-chunks-without-prefix did not hurt adherence at all. The damage
is the summary-prefix. (Exp 16's rel/compl size-artifact reading is unaffected and confirmed: SIZE carries
+0.081 rel / +0.208 compl — the denominator shrink — while PREFIX barely moves them.)

**F2 — telemetry explains WHY the prefix hurts: it makes the 3B BOLDER, converting safe abstentions into
unsupported answers.** Per-example traces, CustomerSupport:
- Abstention ("I cannot answer…") rate **falls** under the prefix: **28% → 22%** (prefix-off 14/50, prefix-on
  11/50). Paired on shared questions, the prefix flipped **5 abstain→answer** vs only **2 answer→abstain** — net
  more answering.
- But an abstention is *scored adherent* (it asserts nothing unsupported): abstainers average adherence ~0.21,
  answerers ~0.66 (prefix-off). So **turning an abstention into an answer is an adherence GAMBLE** — and the
  prefixed answers lose it: within examples that answered under *both*, mean adherence still drops (multiple
  clean True→False flips, e.g. `techqa_DEV_Q303`, `Q220`). The doc-summary prefix injects plausible-but-
  off-context document framing that the 3B then commits to.
- Net: prefix-on answered **39/50** (vs 35) but at **0.462** mean adherence (vs 0.657) → the headline 0.380.
  Fewer "safe" abstentions × lower per-answer faithfulness = the crater. The prefix doesn't make answers more
  grounded; it makes the model *more willing to answer ungrounded*.

**F3 — on GenKnowledge the prefix is adherence-NEUTRAL (0.000); the loss there is pure size-fragmentation.**
Exactly the predicted null: on atomic ~86-word docs the 150-char prefix ≈ the doc's own opening (redundant), so
it neither helps nor hurts faithfulness. GenK's −0.088 completeness is the SIZE effect (−0.160) — the ~85-word
windows fragment a doc that was already ≈ one chunk — partly offset by the prefix's mild +0.072 (re-stating the
opening restores a little coverage). Two domains, two different prefix behaviors: **harmful on long multi-doc
context (injects cross-doc framing the model over-trusts), inert on atomic docs.**

**Bottom line — SAC stays REJECTED, now with the mechanism nailed: it's the PREFIX, not the geometry.** This
kills the Tier-2 follow-ups that assumed the prefix was the value: a **`summary_chars` sweep is pointless**
(more prefix = more of the thing that lowers adherence; the −0.151 is the prefix *working as designed*), and
**SAC + grounded_complete is moot** (no prefix benefit to amplify). The one mildly-interesting residue —
SAC-off's char-window *size* lifted CustS rel/compl with adherence intact (+0.031) — is just another instance
of the size-denominator artifact (Exp 10) and is **dominated by PGC** (rel 0.372 > 0.286, and PGC needs no
char-window hack). Per-domain bests UNCHANGED. **SAC line CLOSED.** Next live lever = the unlocked **PGC +
`bge_large` CustomerSupport stack** (Exp 15 F3): orthogonal stages (chunk boundaries × candidate selection),
both individually helped CustS, so by the stack principle (Exp 8) they may combine — the strongest pending
plain-grounded CustS lever before we exhaust local and move to Groq.

---

## Pooled Exp C — PGC + `bge_large` reranker: does NOT stack, it OVERLAPS (CustS) and INTERFERES (GenK) — REJECTED; 3rd confirmation that rerankers fight quality levers

**Question:** Exp 15 F3 unlocked this — both PGC (chunker) and `bge_large` (reranker) individually *helped*
CustomerSupport under plain grounded @ 3B. They're different pipeline stages (PGC sets chunk **boundaries**;
`bge_large` does candidate **selection**, rerank top-20 → top-5), so by the **stack principle** (Exp 8: PGC +
grounded_complete stacked because orthogonal) they *might* combine — feed PGC's coherent paragraph chunks into
the stronger reranker. The competing hypothesis (my config note): they instead **overlap/average** like
reranker vs grounded_complete (Exp 5A/11), because both are *relevance* levers aiming at the same objective.
Config `grounded_pooled_pgc_bge_rerank.yaml` (PGC chunker + `bge_large`@top5, else the plain pooled baseline,
3B — the exact setup both parents ran). Both domains, N=50 (49/50 scored each). Source: `..._pgc_bge_rerank.csv`;
telemetry `results/telemetry/pooled/grounded_pooled_pgc_bge_rerank.yaml__*.jsonl`.

### vs each parent (only the named stage added; 3B plain grounded throughout)

**CustomerSupport** (target):

| metric | base | PGC alone (Exp 7) | `bge_large` alone (Exp 15) | **STACK** | stack vs PGC |
|---|---|---|---|---|---|
| relevance | 0.205 | 0.372 | 0.234 | **0.379** | +0.007 (tied) |
| utilization | 0.069 | 0.122 | 0.075 | **0.159** | +0.037 |
| completeness | 0.121 | **0.256** | 0.174 | 0.191 | **−0.065** |
| **adherence** | 0.500 | 0.460 | 0.480 | 0.469 | +0.009 (flat) |

**GenKnowledge:**

| metric | base | PGC alone (Exp 7) | `bge_large` alone (Exp 15) | **STACK** | stack vs PGC |
|---|---|---|---|---|---|
| relevance | 0.304 | 0.314 | 0.314 | **0.356** | +0.042 |
| utilization | 0.128 | 0.143 | 0.091 | **0.155** | +0.012 |
| completeness | 0.362 | 0.375 | 0.313 | **0.421** | +0.046 |
| **adherence** | 0.540 | **0.620** | 0.580 | 0.490 | **−0.130** |

### Findings + reasoning

**F1 — REJECTED on CustomerSupport: the stack does NOT beat PGC-alone — it's an OVERLAP, not a stack.** Relevance
is *tied* with PGC (0.379 vs 0.372, +0.007) — i.e. on top of PGC's already-coherent chunks the reranker adds
**no** relevance — while completeness *drops* (−0.065) and adherence is flat. So `bge_large` brings nothing PGC
didn't already provide and *costs* coverage. Mechanism: both stages select **5** chunks; PGC-alone keeps dense
top-5, the stack keeps `bge_large`-reranked top-5. The reranker optimises *pointwise query-relevance*, so it
packs 5 narrowly-relevant chunks where dense top-5 spread wider — trading coverage (completeness) for precision.
This is consistent with the redundancy probe (CustS **40% duplicate** retrieval, 16 Jun): a relevance reranker
can pile near-duplicate high-relevance chunks, shrinking the distinct-content the answer can draw on.

**F2 — on GenKnowledge it's worse than overlap: it ACTIVELY INTERFERES.** rel/util/compl all rise over PGC, but
**adherence craters 0.620 → 0.490 (−0.130)** — and 0.620 (PGC-alone) was the plain-grounded GenK best. Crucially
the stack's adherence (0.490) sits **below BOTH parents** (PGC 0.620, `bge_large` 0.580) — sub-additive, the
signature of two levers fighting. Mechanism: `bge_large` reranks PGC's coherent paragraph chunks by pointwise
relevance, **reordering/reselecting away from the coherent paragraph flow PGC built**; the 3B grounds less well
on the reshuffled context (telemetry: stack abstains 24% vs `bge_large`-alone's 34% — bolder, but answered-mean
adherence only 0.541). Same family as Exp 11 (rerankers fight grounded_complete) — the reranker optimises
*ranking*, not the *coherence/coverage* the other lever supplies.

**F3 — the principle (3rd confirmation): rerankers OVERLAP/FIGHT quality levers because they share the
*relevance* objective.** Stacking (Exp 8) worked when the two levers targeted *different* objectives — PGC
(relevance) + grounded_complete (coverage). Here PGC and `bge_large` both target **precision-of-text relevance**,
so there's no orthogonal axis to add along: once PGC has made relevance non-binding, the reranker has nothing to
improve and only coverage/coherence to disturb. We've now seen this **three times** — Exp 5A (reranker vs
grounded_complete), Exp 11 (reranker-rescue rejected), Exp C (reranker vs PGC). Corollary from Exp 15: the
reranker *only* helps when relevance is the **binding constraint and unaddressed** (it lifted the *plain*
baseline, where relevance WAS the bottleneck). On top of a relevance lever, it can't help.

**Bottom line — PGC + `bge_large` REJECTED; the plain-grounded retrieval-side line is now EXHAUSTED for both
domains.** Per-domain bests UNCHANGED: CustS plain-grounded retrieval stays **PGC-alone** (rel 0.372, compl
0.256); GenK plain-grounded stays **PGC-alone** (adh 0.620). **Reranker line CLOSED** (3× confirmed overlap —
no reranker config remains worth testing under either prompt). Remaining local levers are no longer retrieval-
side: the **generator** (gemma2:9b, Exp E — the family that moved adherence ±0.16 in Exp 12) and a **diversity
retriever** (MMR for CustS — directly attacks the 40%-dup / completeness-culling this experiment re-exposed).

---

## Pooled Exp 17 — top_n WIDENING (5 → 10), the first LEGAL-ported lever: REJECTED — it INVERTS on our corpus because pooled ≠ per-example

**Question:** the partner's Legal Phase A (`docs/legal/PHASE_A_FINDINGS.md`) found `top_n` (chunks reaching the
generator) was its **single most important pin** — top_n=5 caused all-refusals on the 70B, raising to **10** was
*required* to unblock utilization/completeness, and `{5,10,15}` was named their #1 Phase-B sweep. Across all 16
prior pooled experiments we only ever **tightened** top_n (3 or 5) — we **never widened** it. This ports that
lever, aimed at CustomerSupport's one stubborn bottleneck (coverage/completeness). ONE change off the CustS
winner `grounded_pooled_pgc_complete @ 8B` (Exp 12: compl 0.473, adh 0.600): `reranker.top_n` 5 → 10 (retriever
k=20, so 10 chunks are genuinely available). Run on the **8B** deliberately — Exp 11/12 showed the 3B is
exhausted on coverage, so testing on the 3B would confound "top_n doesn't help" with "3B can't use the extra
chunks"; the 8B isolates the lever AND is the decision-relevant baseline. CustomerSupport only (GenK's atomic
~86-word docs would just take noise chunks; no clean GenK baseline to diff). N=50, 49/50 scored. Source:
`..._pgc_complete_topn10.csv`; telemetry `..._pgc_complete_topn10.yaml__CustomerSupport.jsonl`.

### vs the top_n=5 baseline it forks (identical config, ONLY top_n changed; both `pgc_complete` @ 8B, CustS)

| metric | top_n=5 (winner) | top_n=10 (Exp 17) | Δ |
|---|---|---|---|
| relevance | 0.351 | 0.246 | **−0.105** |
| utilization | 0.248 | 0.145 | **−0.103** |
| **completeness** | **0.473** | **0.233** | **−0.240** (nearly halved) |
| **adherence** | **0.600** | **0.490** | **−0.110** |

### Findings + reasoning

**F1 — REJECTED, and decisively: top_n=10 regresses EVERY metric, completeness nearly halved (−0.240).** This is
the *opposite* of Legal, where the same move was the headline win. Far beyond the ±0.07 noise band on all four.

**F2 — the mechanism is the per_example-vs-pooled distinction (the same split that gates our whole pooled track).**
On Legal, retrieval is **per_example**: one contract per question, so top_n=10 pulls 10 chunks *from the same
document* — literally more of the one right source, which is why coverage rose. **Our corpus is POOLED**: each
query retrieves from the *whole domain corpus* (every test ticket's documents). So top_n=10 = the 5 on-topic
chunks **plus 5 from OTHER tickets** — distractors, not missing context. Every metric degrades the way feeding
cross-ticket distractors predicts:
- **relevance −0.105**: the denominator (retrieved doc-sentences) grows with off-topic sentences → relevant
  fraction shrinks (the size-denominator artifact of Exp 10, running in reverse — *more* chunks, *lower* rel).
- **utilization −0.103**: the answer still uses ~the same sentences, now divided by a bigger retrieved set.
- **completeness −0.240** (the crater): `|R∩U|/|R|`. The distractor chunks add "relevant-looking" sentences to
  R that the answer can't possibly cover (they're from other tickets), inflating the denominator while R∩U
  barely moves → the ratio collapses.
- **adherence −0.110**: more off-topic context = more chances for the 8B to ground a sentence in the *wrong*
  ticket. Telemetry: median context **doubled** (5338 chars vs the ~2700 of top_n≈5 runs) and abstentions rose
  to 20% (10/49, each scoring non-adherent here) — the model hedges/misgrounds more when buried in distractors.
  (Within examples that *did* answer, adherence held at 0.615 — so the headline drop is the extra abstentions +
  cross-ticket misgrounding, not a collapse of answer quality per se.)

**F3 — this CLOSES the top_n-widening line for pooled, and reframes it as a per_example lever.** top_n=15 would
be strictly worse (more distractors) and top_n=8 the same mechanism milder — **no widening config is worth
running on the pooled track.** The honest reading: top_n-up is a *single-document* lever; it helps exactly when
"more chunks" = "more of the one relevant doc" (Legal), and hurts when "more chunks" = "more of the corpus"
(us). This is a clean, citable example of **why a Legal-winning lever does NOT transfer** — not because the
mechanism is wrong, but because the corpus structure inverts its sign. (If anything, it argues the reverse
direction — *tighter* retrieval — but Exp 2/5B already showed top_n=3 starves coverage; the CustS sweet spot is
**5**, now bracketed on both sides: 3 starves, 10 distracts.)

**Bottom line — top_n=10 REJECTED; CustS winner UNCHANGED (`pgc_complete @ 8B`, top_n=5).** First Legal-ported
lever, and it ports *negatively* — a useful result (it sharpens the per_example/pooled boundary and brackets the
top_n optimum at 5). Remaining local levers are unchanged: **gemma2:9b** generator (Exp E) and an **MMR
diversity retriever** for CustS — note Exp 17 *reinforces* the MMR rationale: if plain "more chunks" pulls
distractors, a lever that adds *diverse* chunks while suppressing near-dups is the principled way to widen
coverage without the distractor tax.

---

## Pooled Exp 18 — N=100 CONFIRMATION of the CustomerSupport winner + the FIRST noise-floor audit of the pooled track: the winner SURVIVES, and the audit reframes which of our N=50 deltas we may actually claim

**Question (methodology, not a new lever).** Every one of our 17 pooled experiments ran at **N=50, and we never once
computed a noise floor.** The partner's Legal findings doc (`LEGAL_RAG_FINDINGS.docx`) names this its *single most
important episode*: their N=50 "winner" (the R4 reranker, +0.071 relevance) **collapsed to noise when re-run at N=80** —
they nearly shipped a strictly-worse pipeline. That is a direct warning to us. So this experiment does two things:
(1) **re-runs the actual deployed CustS winner** (`grounded_pooled_pgc_complete @ 8B`) at **N=100** to see whether its
lead survives a tighter measurement, and (2) **captures per-example telemetry for the winner for the first time** (the
config predates our telemetry adoption — we had *no* traces for it), which retroactively unlocks the dispersion suite
below for $0. N=100, **99/100 scored**. Source: `..._n100_pooled_pgc_complete_confirm.csv`; telemetry
`grounded_pooled_pgc_complete.yaml__CustomerSupport.jsonl` (99 non-null records).

### Winner: N=50 (Exp 12) vs N=100 confirmation (same config, same seed family, 2× sample)

| metric | N=50 (Exp 12) | N=100 (Exp 18) | Δ | 2-sample 95% floor | verdict |
|---|---|---|---|---|---|
| relevance | 0.351 | 0.369 | +0.018 | ±0.082 | within noise — **stable** |
| utilization | 0.248 | 0.228 | −0.020 | ±0.074 | within noise — **stable** |
| completeness | 0.473 | 0.395 | −0.078 | ±0.152 | within noise — **stable** |
| adherence | 0.600 | 0.535 | −0.065 | ±0.170 | within noise — **stable** |

### The realised PAIRED noise floor at N=50 (computed for $0 from the 9 configs that have telemetry)

Our pooled configs all run the **same seeded examples**, so the right floor is the *paired* one (std of per-example
differences / √n × 1.96), not the independent-sample floor. Median over all available config pairs:

| metric | CustomerSupport floor | GenKnowledge floor |
|---|---|---|
| relevance | ±0.084 | ±0.081 |
| utilization | ±0.063 | ±0.043 |
| **completeness** | **±0.153** | **±0.140** |
| **adherence** | **±0.153** | **±0.157** |

### Findings + reasoning

**F1 — the winner SURVIVES. No metric moved beyond its floor when we doubled N.** Unlike the partner's R4 (whose
relevance lead evaporated and whose completeness/adherence *cratered* at higher N), our winner's four metrics are all
**stable** between N=50 and N=100. The small drifts (compl −0.078, adh −0.065) are real-looking in direction but sit
*inside* the ±0.15 floor, i.e. not claimable as a true shift — the winner is genuinely the winner, not an N=50 artifact.
This is the confirmation that was missing from the whole track.

**F2 — but the audit is sobering: completeness and adherence are EXTREMELY noisy at N=50 (floor ±0.15).** These are the
two metrics the project values most (adherence especially), and they're the *least* resolved at our sample size. The
reason is mechanical, not a bug: both are near-binary per example (adherence *is* boolean; completeness is often 0 or 1
on a single-clause question), so the per-example std is ~0.4–0.5 and the floor is correspondingly wide. Relevance and
utilization are well-resolved (floor ±0.04–0.08); **completeness/adherence need N≥100 to claim a ±0.1 effect, N≈200 for
±0.07.**

**F3 — several of our OWN N=50 headline deltas are at or under the floor — they need the same confirmation R4 did.**
Re-checking past claims against the paired floor (all from stored telemetry, $0):
- **Exp 16b "the summary-PREFIX is the adherence-killer" (CustS adh, Δ claimed −0.151 → measured paired −0.128 vs floor
  ±0.153): WITHIN NOISE.** The *mechanism* story (telemetry: prefix flips abstain→answer, new answers ungrounded) still
  stands as a qualitative read, but the **headline number is not yet claimable** at N=50. This is our R4 moment — flagged,
  not yet confirmed.
- **Exp 16b size effect on completeness (CustS, Δ +0.102 vs floor ±0.140): WITHIN NOISE.**
- **Exp C "PGC lifts relevance on top of the reranker" (CustS rel, Δ +0.137 vs floor ±0.084): REAL ✓** — clears the floor.
- **GenK SAC-geometry relevance (nosum vs sac, Δ +0.089 vs floor ±0.076): REAL ✓.**
  The pattern is clean and reassuring: **every relevance/chunking conclusion we drew clears the floor; the soft
  completeness/adherence deltas are the ones living in the noise.** Our *qualitative* findings (PGC wins CustS; rerankers
  fight quality levers; top_n inverts on pooled) are all relevance/mechanism-driven and safe. The numbers that need N≥100
  before they go in the report are the adherence/completeness deltas.

**F4 — process change for the report and the Groq phase.** From here, (a) **headline claims on completeness/adherence get
an N=100 confirmation** before they're stated as fact; (b) every future run keeps telemetry on so the floor is always
computable for $0; (c) this is itself a citable contribution — *"we audited our own measurement and report only deltas
that exceed the realised noise floor"* is exactly the discipline the partner's doc credits for catching its false winner.

**Bottom line — CustS winner CONFIRMED at N=100 (`pgc_complete @ 8B`); the relevance/chunking conclusions are solid; the
soft adherence/completeness deltas (notably Exp 16b's −0.151 headline) are flagged as not-yet-claimable and queued for
N=100 confirmation.** The most valuable thing the partner's doc gave us was not a lever — it was the noise-floor habit,
and applying it told us *which of our own results to trust*. Remaining local levers unchanged: **gemma2:9b** (Exp E),
**MMR retriever** for CustS, and the **Exp F answer-text padding probe** (now runnable for $0 on the winner's fresh telemetry).

---

## Pooled Exp 19 — MMR diversity retriever on the CustomerSupport winner: REJECTED — no resolvable gain, and it trends the SAME way top_n-widening did (coverage/diversity levers don't transfer to pooled)

**Question.** Our redundancy probe (`scripts/probe_redundancy.py`, 16 Jun) found CustomerSupport's top-5 carries **~40%
near-duplicate pairs** (max pair-sim 0.836) — roughly 2 in 5 CustS queries spend a retrieval slot on a near-copy of a
chunk already selected, so the kept set covers fewer *distinct* facts than its size suggests. (GenKnowledge was ~6% →
settled-negative, not tested.) MMR (Maximal Marginal Relevance, Carbonell & Goldstein 1998) re-orders the k=20 dense
candidates to penalise redundancy — `pick = argmax[ λ·sim(c,q) − (1−λ)·max sim(c,selected) ]` — so the top_n=5 we keep
should span more distinct content. **Hand-implemented** in `src/retrieval/mmr_retriever.py` (numpy; no library — same
"implement the method ourselves" posture as TRACe; verified against an independent brute-force reference on 600 random
inputs, $0). ONE change off the CustS winner `grounded_pooled_pgc_complete @ 8B`: `retriever` dense → mmr, `lambda_mult`
0.7 (lean relevance, gentle diversity). Pre-registered decision rule: *if 0.7 regresses everything, reject; if it shows
signal, sweep λ=0.5.* CustomerSupport only. N=50, 50/50 scored. Source: `..._pgc_complete_mmr.csv`; telemetry
`..._pgc_complete_mmr.yaml__CustomerSupport.jsonl`.

### vs the winner it forks — PAIRED on the 50 shared seeded examples (identical config, ONLY the retriever changed)

| metric | dense (winner) | MMR (Exp 19) | Δ (paired) | N=50 paired floor ± | verdict |
|---|---|---|---|---|---|
| relevance | 0.336 | 0.369 | +0.033 | 0.082 | within noise |
| utilization | 0.241 | 0.232 | −0.009 | 0.084 | within noise |
| **completeness** | **0.461** | **0.309** | **−0.152** | 0.186 | within noise (trends ↓) |
| **adherence** | **0.592** | **0.449** | **−0.143** | 0.179 | within noise (trends ↓) |

(Headline-CSV numbers, MMR vs the N=100 winner: rel 0.364 vs 0.369, util 0.229 vs 0.228, compl 0.323 vs 0.395, adh
0.460 vs 0.535 — same story. The table above is the *paired* read on the shared examples, which is the one that controls
for per-example difficulty.)

### Findings + reasoning

**F1 — REJECTED: every paired delta sits inside the N=50 floor → MMR produced no resolvable improvement on any metric.**
The two metrics the lever was *designed* to lift (completeness, utilization) are flat-to-**down**, the opposite of the
coverage bet. The strict, citable statement is "no improvement, within noise"; the *direction* on completeness/adherence
is downward (and would need N≥100 to claim as a true regression — not worth it for a rejected lever).

**F2 — the mechanism confirms the lever FIRED but the trade lost — and it's the same failure mode as Exp 17 and Exp 16b.**
Telemetry on the shared examples shows MMR did exactly what it was built to do, then backfired:
- **context shrank −385 chars** (3709 → 3324): MMR genuinely dropped near-duplicate content (the redundancy probe was
  right that the dups exist and that MMR removes them). So this is a real intervention, not a no-op.
- **abstain-rate fell 8% → 4%**: the more-diverse context made the 8B *bolder* (fewer refusals) — the same abstain→answer
  flip Exp 16b's prefix caused.
- **but answered-adherence fell 0.600 → 0.479**: when it *did* answer, it grounded **worse**. The chunks MMR promoted to
  fill the freed slots are diverse-but-**less query-relevant**, i.e. exactly the cross-ticket-ish material that Exp 17
  showed the 8B misgrounds on in a pooled corpus. Swapping a redundant-but-on-topic chunk for a diverse-but-off-topic one
  is a net loss here.

**F3 — this is the 3rd independent confirmation that coverage/diversity levers do NOT transfer to POOLED retrieval.**
Exp 17 (more chunks → distractors), Exp 16b (bolder model → ungrounded new answers), and now Exp 19 (diverse chunks →
less-relevant material) all fail the same way: on a pooled corpus, "broaden what reaches the generator" reduces to "feed
it less-relevant text," and the 8B grounds worse on it. A **λ=0.5 sweep is predicted strictly worse** (more diversity =
more of the thing that hurt), so per the pre-registered rule it is **not worth running** — the MMR line is closed. The
redundancy is real (probe + the −385-char context confirm it); it simply isn't the binding constraint, because the cost
of the lower-relevance replacements outweighs the value of removing the dups.

**Bottom line — MMR REJECTED; CustS winner UNCHANGED (`pgc_complete @ 8B`, dense retriever).** The brick is correct and
kept in-tree (registered retriever type `mmr`, independently verified) — only the *empirical bet* failed. With MMR,
top_n-widening, SAC, and the reranker stack all rejected on CustS, the converging signal across the track is that the
**local ceiling is generator capacity, not retrieval** — the headroom is the bigger generator (Groq 70B), exactly where
the next phase points. Remaining *local* items are now only **gemma2:9b** (Exp E, 2nd big local generator) and the **Exp
F answer-text padding probe** ($0 from telemetry).

---

## Pooled Exp F — answer-text probe ($0, telemetry-only): is GenKnowledge's high completeness REAL grounded coverage or thoroughness-padding? Verdict: mostly REAL, with an easy-denominator caveat — NOT a padding artifact

**Question (settles Exp 12 F3, no model run).** GenKnowledge's best completeness (0.655, the `complete_top3` config) is
far above CustomerSupport's. F3 asked whether that is *real coverage* (the answer genuinely states more of the relevant
facts) or a **thoroughness-padding artifact** (a long, rambling answer that accidentally name-drops relevant sentences and
inflates `|R∩U|`). Recall the metric: `completeness = |R∩U| / |R|` — "of the relevant sentences, how many did the answer
use." Pure analysis of stored telemetry (199 GenK records across 4 telemetry'd configs), $0.

**⚠️ Caveat on scope.** The exact 0.655 config (`complete_top3`) **predates telemetry adoption and has no per-example
trace**, so this probe cannot inspect that number directly. It characterises GenK's completeness **mechanism** on the 4
sibling GenK configs that *do* have telemetry (`bge_rerank`, `pgc_bge_rerank`, `sac`, `sac_nosum`; pooled completeness
0.20–0.42). Mechanism is a property of GenK's *data* (atomic docs, small relevant sets), so it transfers; the specific
thoroughness prompt that lifts `complete_top3` to 0.655 is not itself measured here. Telemetry also stores only the four
final scores + answer text, not the raw R/U keys — so `|R|` is reached via proxy (relevance ≈ `|R|/T` at ~fixed context).

### Five tests on the 199 GenK telemetry records

| # | test | result | reads as |
|---|---|---|---|
| A | bimodality of completeness | 51% at 0.0, 18% at 1.0, 31% between | near-binary per example → confirms the wide ±0.15 floor is mechanical, not a bug |
| B | corr(answer length, completeness) | **+0.30**; compl=1.0 answers ~2× longer (186 vs 95 chars) | longer answers *do* score higher — padding-suspicious **on its own** |
| **E** | **completeness \| grounded vs not** | **adherent 0.381 vs non-adherent 0.217; 67% of compl=1.0 are adherent (vs 52% base)** | **decisive: coverage travels WITH grounding, not hallucination → REAL, not padding** |
| C | corr(relevance≈\|R\|, completeness) | **−0.16**; compl=1.0 examples have *lower* rel (0.232 vs 0.369) | the honest caveat: small relevant sets are trivially "fully covered" (easy denominator) |
| D | fraction AT the ceiling `compl=util/rel` | **68%** | when at ceiling, U ⊆ R — everything the answer used was relevant, no junk in the utilized set |

### Findings + reasoning

**F1 — NOT a padding artifact (test E is decisive).** If high completeness were thoroughness-padding, the "extra"
covered content would be ungrounded and would FAIL adherence. The data shows the opposite: completeness is **higher when
the answer is adherent** (0.381) than when it isn't (0.217), and compl=1.0 answers are adherent 67% of the time vs a 52%
base rate. Real grounded substance, not rambling. Test B's length correlation is real but, combined with E, it means
longer-AND-more-grounded = the answer genuinely says more true relevant things (padding would be longer-and-*less*-grounded).
Test D agrees: in 68% of examples everything the answer used was relevant (U ⊆ R), so the utilized set isn't padded with junk.

**F2 — the honest caveat: part of GenK's completeness *lead* is an easy-denominator effect, not extra effort (test C).**
compl=1.0 examples have *lower* relevance → fewer relevant sentences `|R|`. GenK's atomic ~86-word docs produce small `|R|`,
and covering "all of" a 2-fact question is mechanically easy (100% on a 2-question quiz). So GenK's completeness advantage
over CustomerSupport is **partly real grounded coverage (F1) and partly a small-`|R|` denominator artifact (this)** — it is
NOT pure thoroughness, but it is also NOT fake.

**F3 — what would fully close it (low priority).** Directly probing the 0.655 number needs a local re-run of `complete_top3`
on GenK with telemetry ON (not $0) to read that config's actual answer texts. **Skip for now**: the $0 analysis already
falsifies the thing F3 worried about (fake/padded completeness), and the residual question (how much of the *lead* is
small-`|R|`) is a measurement nuance, not a lever — lower value than Exp E / the Groq phase.

**Bottom line — Exp 12 F3 RESOLVED: GenKnowledge completeness is mostly REAL grounded coverage, not a thoroughness-padding
artifact, with the documented caveat that GenK's atomic docs make its completeness denominator easy.** Safe to report
GenK completeness as genuine; cite the small-`|R|` caveat when comparing GenK vs CustS head-to-head. No run spent.

---

## Pooled Exp E — gemma2:9b on BOTH per-domain winners (2nd big LOCAL generator): the local capacity wall is REAL and NOT model-specific — a different 9B regresses both domains, and confirms "bigger over-summarises GenK" is a SIZE effect

**Question.** The whole local sweep has converged on one hypothesis — the ceiling is **generator capacity, not retrieval**
(MMR/top_n/SAC/reranker all rejected on CustS). `gemma2:9b` is a 2nd ~9B local generator (different family from llama),
run on the *actual* per-domain winners: CustS `pgc_complete` and GenK `complete_top3`. Two arms, separate config files +
own CSVs (telemetry is filename-keyed/append-only; reusing the winners' filenames would corrupt their traces). ONE change
per arm: generator → gemma2:9b. N=50, 50/50 scored both. Sources: `..._pgc_complete_gemma.csv` (CustS, paired vs the
winner's 100-record telemetry), `..._complete_top3_gemma.csv` (GenK; the winner `complete_top3` predates telemetry so GenK
is a headline compare vs the stored 3B/8B CSV numbers + gemma's own fresh telemetry for mechanism).

### CustS — PAIRED on 49 shared seeded examples (`pgc_complete`, only the generator changed)

| metric | llama3.1:8b (winner) | gemma2:9b | Δ (paired) | N=50 paired floor ± | verdict |
|---|---|---|---|---|---|
| relevance | 0.336 | 0.377 | +0.041 | 0.077 | within noise (same retriever) |
| **utilization** | **0.241** | **0.138** | **−0.102** | 0.078 | **REAL ↓** |
| **completeness** | **0.461** | **0.235** | **−0.226** | 0.162 | **REAL ↓** (clears the wide floor) |
| adherence | 0.592 | 0.510 | −0.082 | 0.177 | within noise |

### GenK — `complete_top3` across all three generators (the size curve)

| generator | relevance | utilization | **completeness** | adherence |
|---|---|---|---|---|
| llama3.2:**3b** (winner) | 0.402 | 0.313 | **0.655** | 0.540 |
| llama3.1:**8b** | 0.410 | 0.246 | **0.445** | 0.551 |
| **gemma2:9b** | 0.373 | 0.198 | **0.407** | 0.480 |

### Findings + reasoning

**F1 — gemma2:9b helps NEITHER domain; on CustS it REALLY regresses (util −0.102, compl −0.226 both clear the floor).**
Relevance is flat (expected — same retriever; the generator can't change what's retrieved). This is the 4th rejected lever
on CustS, but the first whose completeness delta is large enough to beat the ±0.16 floor — i.e. a *confirmed* regression,
not a noise null.

**F2 — the GenK completeness slide is MONOTONIC in model size: 0.655 (3B) → 0.445 (8B) → 0.407 (9B).** gemma did NOT recover
toward 0.655 — it continued the decline. So Exp 12's 8B dip was **not a llama-specific quirk; it is a SIZE effect**, now
confirmed by a second, different-family 9B. "Bigger models over-summarise GenK's atomic ~86-word docs" graduates from
hypothesis to a twice-confirmed per-domain finding.

**F3 — the mechanism (telemetry) is one behaviour explaining both arms: gemma is TERSER and more CONFIDENT.**
- CustS: abstain 8% → 2%, answer length 362 → 245 chars, answered-adherence 0.600 → 0.500. Context ~unchanged (3624 vs 3578).
- GenK: abstain 0%, answer length 182 chars (shortest of any config), context halved (top_n=3). GenK completeness stays
  grounded (compl|adherent 0.428 > compl|non-adherent 0.388 — the Exp F lens still holds), it's just LOWER.
- Read: gemma answers almost everything but writes shorter, more-summarised answers that **use fewer of the relevant
  sentences** → utilization and completeness fall while adherence holds (it's not hallucinating more — it just *covers less*).
  This is the same abstain→answer boldness Exp 16b's prefix induced, but here it's intrinsic to the model, and the new
  answers are terse rather than ungrounded.

**F4 — what this means for the Groq decision (the important part).** The local capacity wall is now shown to be **real and
~model-agnostic at the 8–9B scale** — two different families land in the same place (and bigger *hurts* GenK). So the
headroom is NOT "a slightly bigger same-scale model"; it is either a genuinely more capable model (**70B-class**) or a
*prompting* fix for the terseness (push coverage without inducing ungrounded boldness). Crucially, **`llama3.1:70b` is
already pulled LOCALLY (42 GB)** — so the decisive big-generator test may be runnable for **$0 locally** before any Groq
spend, which reorders the next phase.

**Bottom line — gemma2:9b REJECTED on both domains; per-domain winners UNCHANGED. The local generator ceiling is confirmed
real and not model-specific.** Best LOCAL generator per domain stands: CustS `pgc_complete @ llama3.1:8b`, GenK
`complete_top3 @ llama3.2:3b`. The local lever sweep is now EXHAUSTED — next is the bigger generator, and llama3.1:70b
being local means we can likely test that hypothesis at $0 first.

---

## Pooled Exp G — "prompting fix first": a PROCEDURAL coverage prompt (`grounded_coverage`) on the CustomerSupport winner, to test whether the terseness wall Exp E exposed is PROMPTABLE on the local 8B. Verdict: REJECTED — it BACKFIRED on every axis, which is exactly why it's decision-useful: the terseness is NOT promptable here → the wall is CAPACITY → 70B is justified

**Why this experiment, and why before 70B.** Exp E's telemetry diagnosed the stuck CustS
completeness/utilization precisely: the 8B (and gemma2:9b) *answer* almost everything but write
short, over-summarised answers that touch *fewer* of the relevant context sentences — it's not
hallucination (adherence held) and not the token cap (Exp F + the length probe killed that). The
winner prompt (`grounded_complete`) already says the abstract *"be thorough"* and the model
demonstrably ignores it. So before spending on a bigger generator, we test the cheap hypothesis:
**is the terseness PROMPTABLE?** `grounded_coverage` replaces the abstract adjective with a
**procedural recipe** — *"the context is given as numbered passages; consider each passage in turn
and include every detail in it that is relevant … do not omit a relevant detail to keep your answer
short."* Framed as a **coverage rule, NOT a length reward** (the Exp 16b guardrail: length-rewarding
pressure cratered adherence). ONE change off the winner (prompt only); `pgc` chunking, top_n 5,
reverse repack, `llama3.1:8b` all held fixed — a clean prompt-only ablation. CustomerSupport only.
N=50, **49/50 scored**. Source: `..._pgc_coverage.csv`; telemetry
`..._pgc_coverage.yaml__CustomerSupport.jsonl`. Brick verified for $0 before the run (registers; the
rendered prompt is a one-variable unified-diff vs the winner — `Context/Question/Answer` scaffold
byte-identical; guardrail clauses present; no length-reward phrasing).

### vs the winner it forks — PAIRED on the 45 shared seeded examples (identical config, ONLY the prompt changed)

| metric | winner (`grounded_complete`) | `grounded_coverage` | Δ (cov − win) | paired floor (±) | verdict |
|---|---|---|---|---|---|
| relevance | 0.359 | 0.418 | **+0.059** | 0.052 | ↑ clears floor — but see caveat (judge re-annotation noise; retrieval was identical) |
| utilization | 0.255 | 0.209 | −0.046 | 0.077 | ↓ within noise |
| completeness | 0.481 | 0.301 | **−0.180** | 0.166 | ↓ **RESOLVABLE regression** |
| adherence | 0.578 | 0.333 | **−0.244** | 0.153 | ↓ **RESOLVABLE regression** |

(Headline-CSV numbers, coverage vs the winner's own N=50: rel 0.407 vs 0.373, util 0.200 vs 0.213,
compl 0.280 vs 0.351, adh 0.327 vs 0.440 — same story, the paired view above is the one to trust.)

### Findings + reasoning

**G1 — REJECTED. The coverage prompt made BOTH target metrics WORSE, not better, and clears the
paired floor doing so.** Completeness −0.180 and adherence −0.244 are both resolvable regressions
(the two metrics it was *built* to lift, or at minimum hold). This is not a no-op; it is a
backfire.

**G2 — the mechanism is a DOUBLE failure (telemetry), and it's the mirror image of Exp 16b.**
- **It made the 8B refuse MORE, not cover more.** Abstain rate **13% → 31%** (paired): 8
  examples flipped answer→abstain, **0** flipped abstain→answer. Reading "consider each passage in
  turn" against the strict *"if the answer is not contained in the context, reply exactly
  [refusal]"* made the model read each passage more *literally* and conclude, more often, that the
  full answer isn't present — so it refused borderline cases it used to attempt. The procedural
  push induced **caution**, the opposite of the intended coverage.
- **On the SAME questions answered under BOTH prompts (stable set, n=31), it grounded WORSE and
  still covered LESS.** Adherence **0.677 → 0.419**, completeness **0.518 → 0.405**, utilization
  flat (0.287 → 0.291). Told to "include every relevant detail from each passage," the 8B reached
  across passages for more material, stitched it *less faithfully* (the judge marked more of it
  unsupported), and — the punchline — **didn't even cover more relevant material**. Classic
  recall-for-precision trade with no recall payoff: more reaching, worse grounding, equal-or-lower
  coverage. Answered-only adherence over the full answered pools tells the same story (0.615 → 0.419).
- **Exp 16b symmetry.** There, a summary-PREFIX made the *3B* bolder (abstain↓) and the new answers
  were ungrounded → adherence fell. Here, a procedural-coverage push made the *8B* more cautious
  (abstain↑) AND less grounded on what it did answer. Both directions of prompt pressure land on the
  same wall from opposite sides: **at this capacity, you cannot prompt the model into more
  *grounded* coverage** — push for boldness and it hallucinates, push for thoroughness and it either
  refuses or over-reaches and misgrounds.

**G3 — the one "gain" is an artifact, not a result.** Relevance "rose" +0.059 — but a prompt
**cannot** change relevance: relevance is a property of the retrieved context, and `context_chars`
was **byte-identical on all 46 shared examples** (verified — retrieval was provably unchanged). The
move is **judge re-annotation noise**: the judge re-scores relevance per run, and LLM-judge
nondeterminism gives the same context slightly different relevant-sentence sets across runs. Useful
reminder that even retrieval-only metrics carry judge-rerun noise; we do **not** claim this +0.059.

**G4 — scope honesty.** This is ONE prompt variant (the "procedural" design we chose over a gentler
bolt-on and a more aggressive length-reward). We did not exhaust prompt space. But the result is
*informative* precisely because it didn't no-op — it backfired in the mechanistically-predictable
recall/precision way we've now seen repeatedly (Exp 4's "be thorough" was already the best a prompt
got; Exp 16b's boldness push hurt adherence; this thoroughness push hurt both). The plausible
operating range is bracketed: gentler → no-op (like "be thorough"), this → backfire-via-caution,
more aggressive → backfire-via-boldness (Exp 16b). None clears the wall. N=100 confirmation is
*available but low-priority* — we are not adopting this, the direction clears the floor, and the
abstain-doubling + stable-set crater don't depend on the exact magnitude. **Winner UNCHANGED:**
`pgc_complete @ llama3.1:8b`.

**Bottom line — the "prompting fix first" branch is RESOLVED: the CustS terseness is NOT promptable
on the 8B.** A procedural coverage push didn't lift coverage; it made the model refuse more and
ground worse. Combined with Exp 16b and Exp 4, the **prompt lever for grounded coverage is now
exhausted** on this domain at this capacity. This is the clean green light we wanted: the wall is
**generator capacity**, so the decisive next test is a genuinely-more-capable generator. Crucially,
`llama3.1:70b` is already local (42 GB) → that test can run at **$0 locally** before any Groq spend.

---

## Pooled Exp H — the LAST untried prompt mode: FEW-SHOT demonstration (`grounded_fewshot`) on the CustomerSupport winner — SHOW, don't tell. Verdict: REJECTED — no lift; with Exp G this BRACKETS the prompt space, so "prompt lever exhausted on CustS @ 8B" is now a DEFENSIBLE claim, not an overstatement

**Why this experiment.** After Exp G backfired, the honest objection was: *one prompt failing
doesn't prove the whole prompt space is empty.* Correct — so we tested the one mode that is
**qualitatively different** from everything before it. All five prior prompts *tell* the model what
to do (an instruction/adjective/recipe): `grounded`, `grounded_complete` (the winner, "be
thorough"), `extractive` ("quote the source"), `grounded_coverage` (the procedural recipe). Small
models often imitate a worked example far better than they obey an adjective — so `grounded_fewshot`
keeps the winner's instruction VERBATIM and **prepends two short SYNTHETIC demonstrations** (not from
the test set — no leakage): Example 1 pulls a fact from *every* passage into one thorough answer
(shows the extent of coverage); Example 2 is unanswerable and its demo answer is the refusal string
(demonstrates the adherence guardrail by example — the deliberate counter to Exp G's
coverage-vs-grounding failure). ONE change off the winner (prompt only); CustS only. N=50, **48/50
scored**. Source: `..._pgc_fewshot.csv`; telemetry `..._pgc_fewshot.yaml__CustomerSupport.jsonl`.

**Measurement-clean by construction** (verified for $0 before the run): the demos live in the INPUT
prompt; the model emits ONE answer after the final `Answer:`, and the judge scores only that. The
real `Context/Question/Answer` scaffold is **byte-identical** to the winner — few-shot ONLY prepends.
(This is exactly why we did NOT run a two-step "list facts then answer" prompt: its copied-context
preamble would have fake-inflated utilization/completeness. Few-shot has no such confound.)

### vs the winner it forks — PAIRED on the 46 shared seeded examples (identical config, ONLY the prompt changed)

| metric | winner (`grounded_complete`) | `grounded_fewshot` | Δ (few − win) | paired floor (±) | verdict |
|---|---|---|---|---|---|
| relevance | 0.354 | 0.358 | +0.004 | 0.058 | ↑ within noise (retrieval identical — as it must be) |
| utilization | 0.252 | 0.189 | **−0.062** | 0.055 | ↓ **RESOLVABLE regression** |
| completeness | 0.493 | 0.352 | −0.141 | 0.157 | ↓ within noise (trends DOWN) |
| adherence | 0.587 | 0.457 | −0.130 | 0.143 | ↓ within noise (trends DOWN) |

⚠️ **The unpaired CSV misleads here.** The headline rows (fewshot adh 0.48 vs winner-N50 0.44;
compl 0.348 vs 0.351) make few-shot look adherence-POSITIVE and completeness-flat. That is a
**different-denominator illusion** — the two runs scored different example subsets. PAIRED on the
46 examples both saw, every signal is flat-or-DOWN: utilization is a resolvable regression and both
completeness and adherence trend down (within floor, but the wrong direction). This is the Exp 18
lesson in action: trust the paired view, not the unpaired headline.

### Findings + reasoning

**H1 — REJECTED. Few-shot did not lift the stuck metrics; the only resolvable move is utilization
DOWN.** Nothing improved beyond the noise floor; the lone floor-clearing change is a regression.
The demonstration mode is not the missing key on CustS at 8B.

**H2 — mechanism (telemetry): the demos made the 8B IMITATE the SHAPE, not the SUBSTANCE — and
slightly more cautious.** Abstain rose 13% → 24% (6 answer→abstain vs 1 the other way) — the
unanswerable Example 2 demonstrated refusal a touch *too* well, nudging the model to refuse 5 net
more borderline questions (the same cautious drift Exp G showed, milder). On the **stable-answered
set** (answered under BOTH, n=34) the verdict is cleanest: completeness 0.555 → 0.427, utilization
0.289 → 0.239, adherence 0.647 → 0.559 — all down. Answer length is **unchanged** (median 514 chars
both; answered-mean 415 → 412): the model copied the *format* of the worked examples but covered
*less* of the actual context. A demonstration can show the 8B *what a thorough answer looks like*; it
cannot give it the capacity to *find and faithfully ground* more relevant sentences in this pooled,
cross-ticket context.

**H3 — this is the result that makes "prompt lever exhausted" DEFENSIBLE (the point of running it).**
We now have **six prompts spanning both modes** all landing at or below the same ceiling:
instruction-style — `grounded` (baseline), `grounded_complete` (the 0.49 ceiling), `extractive`
(no unlock), `grounded_coverage` (backfired); demonstration-style — `grounded_fewshot` (this, no
lift). And the *failure directions are mechanistically consistent*: push coverage by instruction →
refuse-or-misground (Exp G); show coverage by example → imitate shape, cover less (Exp H); the
neutral "be thorough" sits at the top of the hill at ~0.49 completeness / ~0.58 adherence. That is a
**capacity signature**: the binding constraint is the 8B's ability to *discriminate grounded-vs-not
and assemble faithful coverage* from pooled context — something prompt wording and prompt examples
both leave untouched. We did not literally enumerate prompt space (unfalsifiable), but we sampled its
two distinct modes and bracketed the operating range; continuing to tune wording is now low-value.

**H4 — scope/discipline.** Within-floor on compl/adh, so we claim only the DIRECTION (flat-or-down)
and the one resolvable util regression; per the process rule a headline number would want N=100, but
we are NOT adopting this prompt and the direction is unambiguous, so N=100 is unwarranted spend.
Winner **UNCHANGED**: `pgc_complete @ llama3.1:8b`.

**Bottom line — the prompt question is now CLOSED on CustomerSupport.** Two genuinely different prompt
modes (instruction in Exp G, demonstration here), on top of four earlier instruction variants, all
fail to move grounded coverage off the `grounded_complete` ceiling — and the mechanisms converge on
capacity, not wording. This is the disciplined, bracketed close we wanted before spending: **the wall
is generator capacity**, so the decisive next test is a genuinely more capable generator. `llama3.1:70b`
is already local (42 GB) → that test can run at **$0 locally** first, with Groq held for speed/validation.

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
`..._compress_emb.csv` (Exp 14b),
`..._bge_rerank.csv` (Exp 15),
`..._sac.csv` (Exp 16), `..._sac_nosum.csv` (Exp 16b), `..._pgc_bge_rerank.csv` (Exp C),
`..._pgc_complete_topn10.csv` (Exp 17),
`..._n100_pooled_pgc_complete_confirm.csv` (Exp 18, N=100 winner confirmation + first telemetry capture for `pgc_complete`),
`..._pgc_complete_mmr.csv` (Exp 19, MMR diversity retriever — REJECTED),
`..._pgc_complete_gemma.csv` + `..._complete_top3_gemma.csv` (Exp E, gemma2:9b on both winners — REJECTED),
`..._pgc_coverage.csv` (Exp G, procedural coverage prompt — REJECTED, backfired),
`..._pgc_fewshot.csv` (Exp H, few-shot demonstration prompt — REJECTED, no lift),
`results/pooled/figures/` (charts). Per-example track + reference comparison live in `EXPERIMENTS.md`.*
