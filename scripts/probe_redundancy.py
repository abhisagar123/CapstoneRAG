"""Pre-check for MMR (Exp 15): is our retrieved set actually REDUNDANT?

MMR (Maximal Marginal Relevance) only helps when the top-k chunks fed to the
generator are NEAR-DUPLICATES of each other — it trades a little relevance to
swap a duplicate for something that adds new information. If our retrieved sets
are already diverse, MMR is a no-op (or a slight loss, since it lowers relevance
to buy diversity we don't need).

So before building an MMR retriever, MEASURE the redundancy. This script reuses
the REAL pipeline (same chunker / embedder / FAISS index / dense retriever as the
pooled baseline), retrieves the top-k for every query, and reports how similar the
retrieved chunks are TO EACH OTHER.

  No generator. No judge. No network beyond the dataset pull.
  Just: chunk -> embed -> index -> retrieve -> measure chunk<->chunk similarity.

The embedder runs with normalize=True (unit vectors), so cosine == dot product;
the vectors already live in the FAISS index, so we read them straight back out.

How to read the output (the decision rule):
  mean pairwise cosine among the top_n chunks, averaged over queries
    >= 0.6   HIGH redundancy  -> MMR has real room; build Exp 15.
    0.4-0.6  MODERATE         -> MMR might help at the margin; judgement call.
    <  0.4   LOW              -> sets already diverse; MMR is a no-op. Skip it.
We also print the share of query-sets that contain at least one near-duplicate
PAIR (cosine >= 0.9) — the single clearest "MMR would have something to remove"
signal, and the max pair per set.

Run (CPU is fine — MiniLM is tiny):
  .venv-eda/bin/python scripts/probe_redundancy.py --domains GenKnowledge CustomerSupport --n 50 --k 20 --top-n 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import src  # noqa: F401 — registers light components
from src.embeddings import load_embedders
from src.chunking import load_chunkers
from src.config import from_dict
from src.pipeline import build_pipeline
from src.data_loader import load_domain


# The pooled baseline config (mirrors configs/grounded_pooled_norerank.yaml):
# fixed-512 chunks, MiniLM, FAISS pooled, dense k=20. We stop at retrieve(), so
# prompt/generator are placeholders that are never invoked.
BASE = {
    "chunker":   {"type": "fixed", "size": 512, "overlap": 50},
    "embedder":  {"type": "minilm"},
    "index":     {"type": "faiss", "corpus_mode": "pooled"},
    "retriever": {"type": "dense", "k": 20},
    "reranker":  {"type": "none", "top_n": 5},
    "repacker":  {"type": "reverse"},
    "prompt":    {"type": "grounded"},
    "generator": {"type": "echo"},     # never called — we stop after retrieve()
    "splitter":  {"type": "regex"},
}


def pairwise_stats(vectors: np.ndarray) -> dict:
    """Given (m, dim) UNIT vectors, summarise their pairwise cosine similarities.

    Returns mean / max over the m*(m-1)/2 unique off-diagonal pairs. With unit
    vectors, the full cosine matrix is just V @ V.T; we mask out the diagonal
    (a chunk is trivially identical to itself) and the duplicate lower triangle.
    """
    m = len(vectors)
    if m < 2:
        return {"mean": float("nan"), "max": float("nan"), "n_pairs": 0}
    sims = vectors @ vectors.T                       # (m, m) cosine, since unit-norm
    iu = np.triu_indices(m, k=1)                      # upper triangle, no diagonal
    pair_sims = sims[iu]
    return {"mean": float(pair_sims.mean()),
            "max": float(pair_sims.max()),
            "n_pairs": int(len(pair_sims))}


def probe_domain(domain: str, n: int, k: int, top_n: int, dup_thresh: float) -> dict:
    """Build the pooled index once for `domain`, retrieve every query, and gather
    chunk<->chunk redundancy stats at both the k (retrieved) and top_n (sent to the
    generator) levels."""
    cfg = from_dict({**BASE, "domain": domain,
                     "retriever": {"type": "dense", "k": k},
                     "reranker": {"type": "none", "top_n": top_n}})
    pipe = build_pipeline(cfg)

    examples = load_domain(domain, split="test", n=n, seed=42)

    # Pooled: index the WHOLE sampled corpus once (union of every example's docs).
    pipe.reset_index()
    corpus = [d for ex in examples for d in ex["documents"]]
    n_chunks = pipe.index_documents(corpus)

    # The FAISS index holds the chunk vectors in a parallel list; pull them back so
    # we can compute chunk<->chunk cosine without re-embedding. reconstruct_n reads
    # the stored (already-normalized) vectors straight out of the flat index.
    faiss_index = pipe.index._index
    all_vecs = faiss_index.reconstruct_n(0, faiss_index.ntotal)   # (n_chunks, dim), unit-norm

    topn_means, topn_maxes, k_means = [], [], []
    n_with_dup = 0
    for ex in examples:
        retrieved = pipe.retriever.retrieve(ex["question"], k=k)
        if len(retrieved) < 2:
            continue
        # Map each retrieved chunk back to its row in the index (identity by the
        # exact chunk object the index stored), then grab its stored vector.
        id_of = {id(ch): i for i, ch in enumerate(pipe.index._chunks)}
        rows = [id_of[id(rc.chunk)] for rc in retrieved]
        vecs_k = all_vecs[rows]                       # the k retrieved, in rank order
        vecs_top = vecs_k[:top_n]                      # the subset that reaches the generator

        k_means.append(pairwise_stats(vecs_k)["mean"])
        st = pairwise_stats(vecs_top)
        topn_means.append(st["mean"])
        topn_maxes.append(st["max"])
        if st["max"] >= dup_thresh:
            n_with_dup += 1

    n_q = len(topn_means)
    return {
        "domain": domain,
        "n_queries": n_q,
        "n_chunks_indexed": n_chunks,
        "topn_mean": float(np.mean(topn_means)) if n_q else float("nan"),
        "topn_max_mean": float(np.mean(topn_maxes)) if n_q else float("nan"),
        "k_mean": float(np.mean(k_means)) if n_q else float("nan"),
        "dup_share": (n_with_dup / n_q) if n_q else float("nan"),
        "dup_thresh": dup_thresh,
        "top_n": top_n, "k": k,
    }


def verdict(topn_mean: float) -> str:
    if topn_mean >= 0.6:
        return "HIGH redundancy -> MMR has real room. Worth building Exp 15."
    if topn_mean >= 0.4:
        return "MODERATE -> MMR might help at the margin. Judgement call."
    return "LOW redundancy -> retrieved set already diverse. MMR likely a no-op; skip."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["GenKnowledge", "CustomerSupport"])
    ap.add_argument("--n", type=int, default=50, help="examples per domain (seeded sample)")
    ap.add_argument("--k", type=int, default=20, help="retriever candidate count")
    ap.add_argument("--top-n", type=int, default=5, help="chunks that reach the generator")
    ap.add_argument("--dup-thresh", type=float, default=0.9, help="cosine for a near-dup PAIR")
    args = ap.parse_args()

    load_embedders()                  # registers the heavy MiniLM embedder
    load_chunkers()                   # registers heavy chunkers (fixed is light, but be safe)

    print(f"\nMMR pre-check — chunk<->chunk redundancy in the retrieved set")
    print(f"(pooled, fixed-512, MiniLM, dense; n={args.n}/domain, k={args.k}, top_n={args.top_n})\n")

    rows = []
    for domain in args.domains:
        print(f"  indexing + retrieving {domain} ...", flush=True)
        rows.append(probe_domain(domain, args.n, args.k, args.top_n, args.dup_thresh))

    print("\n" + "=" * 78)
    hdr = f"{'domain':<16}{'queries':>8}{'chunks':>8}{'top_n mean':>12}{'top_n max':>11}{'k mean':>9}{'dup%':>7}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        print(f"{r['domain']:<16}{r['n_queries']:>8}{r['n_chunks_indexed']:>8}"
              f"{r['topn_mean']:>12.3f}{r['topn_max_mean']:>11.3f}{r['k_mean']:>9.3f}"
              f"{r['dup_share']*100:>6.0f}%")
    print("=" * 78)
    print("\nColumn meanings:")
    print("  top_n mean = avg pairwise cosine among the chunks SENT TO THE GENERATOR (the headline)")
    print("  top_n max  = avg of the SINGLE most-similar pair per query (how close the closest two are)")
    print("  k mean     = same mean, over the wider k-candidate pool MMR would pick from")
    print(f"  dup%       = share of queries with a near-duplicate pair (cosine >= {args.dup_thresh})")
    print("\nVerdict (by top_n mean):")
    for r in rows:
        print(f"  {r['domain']:<16} {verdict(r['topn_mean'])}")
    print()


if __name__ == "__main__":
    main()
