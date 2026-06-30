r"""MMRRetriever — re-rank dense candidates for DIVERSITY via Maximal Marginal Relevance.

Pure dense retrieval optimises ONE thing: query-relevance. So when a corpus contains
near-duplicate chunks (the same FAQ pasted across tickets, boilerplate, repeated
policy text), the top-k can fill up with several copies of the same fact — slots
spent on redundancy instead of coverage. Our redundancy probe found exactly this on
CustomerSupport (≈40% of queries had a near-duplicate pair in the top-5; GenKnowledge
only ≈6%, which is why MMR is a CustomerSupport-only lever).

MMR (Carbonell & Goldstein, 1998) fixes that by scoring each candidate against the
query AND against what's already been picked, then greedily taking the best of that
trade-off:

    pick = argmax_{c not yet selected} [ λ · sim(c, query)  −  (1−λ) · max_{s in selected} sim(c, s) ]
                                          \___relevance___/      \________redundancy________/

  - λ = 1.0  → pure relevance (identical to DenseRetriever's order)
  - λ = 0.0  → pure diversity (ignore the query, just spread out)
  - 0 < λ < 1 → "trade a little relevance for coverage" — the bet we're testing

WHY we must re-embed the candidate texts here:
  The index's search() returns RetrievedChunk(chunk, score, rank) — the per-chunk
  query similarity (`score`) but NOT the stored vectors. MMR's redundancy term needs
  candidate-to-CANDIDATE similarity, i.e. vectors. So we re-embed the ~k candidate
  texts with the SAME embedder (same model → same vector space) — only ~20 short texts
  per query, batched, so the cost is negligible — and compute all similarities from
  those vectors. We re-embed the query too and unit-normalise everything defensively,
  so MMR is correct (dot == cosine) regardless of whether the index normalised.

WHAT we hand downstream:
  We return the candidates RE-ORDERED by MMR. The pipeline truncates to top_n
  (candidates[:top_n]) AFTER us, so re-ordering is exactly how we make the kept set
  diverse without touching the truncation logic. On each returned chunk we stamp
  rank = the MMR position (the authoritative order) and KEEP score = the query
  relevance cosine (a real, interpretable number, comparable to dense/hybrid scores).
  rank and score are intentionally NOT co-monotonic — that's the point: a chunk can
  rank lower than its raw relevance because a more-relevant near-duplicate already
  claimed a slot, and telemetry can SEE that trade-off.

LIGHT at import. Registered as retriever type "mmr". Built specially in the pipeline
(it wraps embedder+index like dense/hybrid), not via a bare registry call.
"""

import numpy as np

from ..registry import register
from ..indexing import RetrievedChunk
from .dense_retriever import DenseRetriever


@register("retriever", "mmr")
class MMRRetriever:
    def __init__(self, embedder, index, *, lambda_mult: float = 0.7):
        # lambda_mult: relevance↔diversity knob in [0, 1]. Default 0.7 = lean relevance,
        # gentle diversity ("trade a LITTLE relevance for coverage"). 'lambda' is a Python
        # keyword, so the config/kwarg is named lambda_mult (matches LangChain's MMR).
        assert 0.0 <= lambda_mult <= 1.0, f"lambda_mult must be in [0,1], got {lambda_mult}"
        self.dense = DenseRetriever(embedder=embedder, index=index)
        self.embedder = embedder            # re-embed candidate texts for the diversity term
        self.lambda_mult = float(lambda_mult)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Fetch k dense candidates, then greedily re-order them by MMR."""
        hits = self.dense.retrieve(query, k=k)
        if len(hits) <= 1:
            return hits                     # nothing to diversify

        # One batch embed: [query, cand_0, cand_1, ...] → split row 0 off as the query.
        texts = [query] + [h.chunk.text for h in hits]
        vecs = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        # Defensive unit-normalise so dot product == cosine even if the embedder didn't.
        vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        qvec, cvecs = vecs[0], vecs[1:]     # cvecs: (n, dim)

        rel = cvecs @ qvec                  # (n,)   cosine(candidate, query)
        sim = cvecs @ cvecs.T               # (n, n) cosine(candidate, candidate)

        # Greedy MMR: first pick = most relevant (selected is empty → no redundancy term).
        n = len(hits)
        first = int(np.argmax(rel))
        selected = [first]
        remaining = [i for i in range(n) if i != first]
        while remaining:
            # redundancy[i] = how close candidate i is to its NEAREST already-picked chunk.
            redundancy = sim[np.ix_(remaining, selected)].max(axis=1)
            mmr = self.lambda_mult * rel[remaining] - (1.0 - self.lambda_mult) * redundancy
            pick = remaining[int(np.argmax(mmr))]
            selected.append(pick)
            remaining.remove(pick)

        # Re-stamp rank = MMR order; KEEP score = query-relevance cosine (see module docstring).
        return [RetrievedChunk(chunk=hits[idx].chunk, score=float(rel[idx]), rank=rank)
                for rank, idx in enumerate(selected)]
