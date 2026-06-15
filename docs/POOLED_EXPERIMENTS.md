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

### Pooled track — status + next
1. ✅ 4A BGE null · 4B complete · 5A complete+rerank (averaged) · 5B complete+top3 (GenK win) ·
   6 hybrid (negative) · 7 chunking (PGC = CustomerSupport retrieval win) · **8 pgc+complete (STACKS on
   coverage; trades adherence).**
2. **Per-domain bests (updated):** GenKnowledge → `complete+top3` (compl 0.655). CustomerSupport →
   **`pgc_complete` if optimizing coverage (compl 0.351, rel 0.373)**, or plain `complete` if optimizing
   adherence (0.540). GenKnowledge wants generation-side levers; CustomerSupport wants PGC chunking +
   the completeness prompt — a clean, documented domain split.
3. **Levers now broadly explored on the 3B** (chunking, embedder, retriever, reranker, prompt, top_n,
   and the orthogonal combine). **Next = bigger generator** (`--gen-model llama3.1:8b`) on the per-domain
   winners — does the generation-side capacity lift the adherence that the coverage-maximizing combos
   traded away? (User's order: small-gen levers first — now done — then scale up.) Low-odds leftover:
   extractive prompt.

> ⚠️ **INFRA NOTE (14 Jun):** the PGC run first SEGFAULTED on CustomerSupport (14,968 chunks). Root
> cause: faiss-cpu + PyTorch each bundle their own OpenMP runtime; loaded together they trip
> `OMP: Error #15` → hard crash during faiss search under load. Fixed in `src/__init__.py`
> (`KMP_DUPLICATE_LIB_OK=TRUE` + `OMP_NUM_THREADS=1`, set before faiss/torch load). Any large pooled
> index would hit this — teammates included.

---

*Data sources: `results/pooled/ragbench_matrix_n50_pooled.csv` (Exp 1–3),
`..._bge.csv` + `..._complete.csv` (Exp 4), `..._complete_rerank.csv` + `..._complete_top3.csv` (Exp 5),
`..._hybrid.csv` + `..._hybrid_dense_heavy.csv` (Exp 6),
`..._chunk256.csv` + `..._chunk128.csv` + `..._pgc.csv` (Exp 7), `..._pgc_complete.csv` (Exp 8),
`results/pooled/figures/` (charts). Per-example track + reference comparison live in `EXPERIMENTS.md`.*
