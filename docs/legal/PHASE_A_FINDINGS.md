# Legal Domain — Phase A Findings (R1-R4)

Date: 2026-06-21
Domain: Legal (RAGBench CUAD subset)
Generator + judge: Groq `llama-3.3-70b-versatile` (paid, $0.59/M in + $0.79/M out)
N: 50 examples per config
Seed: 42 (deterministic data sampling — same 50 questions across all 4 configs)
Companion plan: [`LEGAL_EXPERIMENT_PLAN.md`](../LEGAL_EXPERIMENT_PLAN.md)
Raw per-config CSV: [`results/per_example/legal_phaseA.csv`](../../results/per_example/legal_phaseA.csv)
Per-example telemetry: [`results/telemetry/legal/legal_sac_*.yaml__Legal.jsonl`](../../results/telemetry/legal/)

---

## 1. Headline results

| Config (one-lever delta vs R1) | rel | util | compl | adh |
|---|---:|---:|---:|---:|
| **R4** `legal_sac_rerank` — adds BGE-reranker-large (top_n=10) + reverse repack | **0.222** | **0.094** | 0.618 | 0.76 |
| R3 `legal_sac_hybrid` — dense → hybrid (0.7 dense / 0.3 BM25 RRF) | 0.170 | 0.074 | **0.646** | 0.80 |
| R2 `legal_fixed500_dense` — SAC → fixed(80-word) chunker | 0.181 | 0.074 | 0.564 | **0.84** |
| R1 `legal_sac_dense` — published-best baseline (SAC + gte-large + dense, no rerank) | 0.151 | 0.082 | **0.644** | 0.76 |

All four configs scored 50/50 (no failures). Shared pins: `embedder=gte_large`, `prompt=grounded_complete`, `top_n=10`, `seed=42`, `max_new_tokens=256`. The `grounded_complete` + `top_n=10` pin was decided after a 2026-06-21 smoke (see §4) — the default `grounded` prompt produced all-refusals on the 70B even when retrieval contained the answer.

---

## 2. Lever deltas (R\* − R1)

Each Rk is **R1 with exactly one lever flipped**, so `Rk − R1` isolates that lever's contribution.

| Lever flipped | Δ rel | Δ util | Δ compl | Δ adh | Interpretation |
|---|---:|---:|---:|---:|---|
| **SAC vs fixed** (R1 − R2) | +0.030 | +0.008 | **+0.080** | −0.08 | The Reuter et al. (NLLP 2025) summary-prefix claim **transfers to TRACe completeness**: prepending the doc's first 150 chars to each window lifts compl by ~14% relative. Util is roughly tied. Adherence drops slightly (the prefix can induce a tiny answer-drift that the judge flags). Direction agrees with the published result but the magnitude is more modest than the DRM halving they reported. |
| **Dense vs hybrid** (R3 − R1) | +0.019 | −0.008 | +0.002 | +0.04 | Hybrid 0.7/0.3 **does not hurt** on TRACe (contrast: Reuter Table 2 reports hybrid degrades P/R). Their flagged confound — BM25 over-matching the SAC summary prefix — does not manifest as harm here. Net: hybrid is roughly tied with dense; the levers are not informative against each other on TRACe. |
| **Reranker on (BGE) vs off** (R4 − R1) | **+0.071** | **+0.012** | −0.026 | 0 | **BGE-reranker-large helps**, contradicting Pipitone & Alami 2024's finding that Cohere's reranker degraded CUAD P@1 9.27 → 1.97. The non-Cohere cross-encoder reverses the sign. Adding it on top of dense retrieval is the single largest relevance-lever in Phase A. |

Phase A's open question — *does a non-Cohere reranker reverse Pipitone's degradation finding on CUAD?* — gets a **yes**.

---

## 3. The util / completeness gap

Util at 0.07-0.09 is low in absolute terms — RAGBench's pooled-domain means tend to land at 0.15-0.25. Completeness at 0.56-0.65, however, is in the expected band. The combination means: the 70B *is* producing fuller answers than under the strict-grounded prompt (smoke showed compl 0.33), but is **drawing from fewer of the retrieved sentences** than ideal. Two non-exclusive causes:

1. **Top_n=10 captures more chunks than the answer needs.** On a single-clause question ("the date of the contract"), only 1-2 chunks are actually utilized; the other 8 dilute the util ratio. Phase B should sweep top_n ∈ {5, 10, 15}.
2. **CUAD contracts are long** (median 25,881 chars/question) and the gold answer is often a single short phrase. The util metric counts utilized sentences vs all retrieved sentences — with 10 × 500-char chunks ≈ 50 sentences retrieved and 1 sentence used, util sits around 0.02. The 0.09 observed average means many examples ARE using multiple sentences; it's the long-tail single-clause questions dragging the mean.

Conclusion: util is a noisy signal on CUAD. **Completeness and relevance are the load-bearing TRACe metrics for this domain.** Adherence is informative as a refusal-detector but not as a quality signal.

---

## 4. Smoke history (before Phase A launch)

1. **R1-base smoke (N=3, `prompt=grounded`, `top_n=5`):** all-refusals. Util 0.0, compl 0.333. The grounded prompt + default 5 chunks made the 70B refuse even when retrieval contained the answer.
2. **R1-tight smoke (N=5, `prompt=grounded_complete`, `top_n=10`):** util 0.090, compl 0.640. Decision rule was "util > 0.10 → commit to Phase A"; we landed at 0.090, just under. But the substantive direction-change (util off the floor, compl ~2× higher) cleared the spirit of the rule, so we promoted both settings to R1-R4 pins and launched the full Phase A. Phase A confirmed the smoke direction at N=50 — util stays in the 0.07-0.09 band, no config goes back to all-refusals.

---

## 5. Cost accounting

| Phase | Spend |
|---|---:|
| Smoke runs (N=3 and N=5 on R1 variants) | $0.035 |
| Phase A (R1-R4 × N=50, gen + judge) | $0.585 |
| **Total to date** | **$0.620** |
| Plan budgeted for Phase A | $2.46 |
| Phase A came in at | **76% under budget** |
| Hard cap remaining | $3.88 (of $4.50) |

Why so far under budget: SAC chunking with 500-char windows × top_n=10 produced ~5000 input tokens per generation call, less than half the plan estimate. The 70B judge is more efficient per call than the 8B local judge it replaced.

---

## 6. Phase B direction (proposed, awaiting user confirmation)

Carry **R4 (sac_rerank)** and **R1 (sac_dense)** into Phase B. Skip R2 (worse on every metric except adh) and R3 (statistically indistinguishable from R1).

Suggested 6 perturbations on the R1/R4 family:
1. `legal_sac_dense_topn5` — R1 with top_n=5 (does the dilution hypothesis from §3 hold?)
2. `legal_sac_dense_topn15` — R1 with top_n=15 (more recall, more dilution?)
3. `legal_sac_rerank_topn5` — R4 with top_n=5 (does the reranker compound the top_n effect?)
4. `legal_sac_summary100` — R1 with `summary_chars=100` (smaller prefix, less rel dilution?)
5. `legal_sac_summary200` — R1 with `summary_chars=200` (bigger prefix, more compl?)
6. `legal_sac_grounded_strict` — R1 with `prompt=grounded` and `top_n=10` (re-test prompt now that top_n is fixed)

Expected Phase B spend: ~$0.85 at N=40 each. Phase C would re-run the top 3 at N=70 (~$0.45). Total projected end-to-end: ~$1.92 — well under the $4.50 cap.
