"""Pre-check for Exp 15 (raise max_new_tokens): are the 8B's CustomerSupport
answers actually being TRUNCATED at the 256-token cap?

Per-example Exp 6A found raising the cap 256->512 was a NULL — but it verified
that on the 3B generator (answers were 3-62 words, nowhere near the cap). Pooled
Exp 12 then changed the regime: the 8B on CustomerSupport's winner (pgc_complete)
writes THOROUGH answers (it "used more context" — util +0.034, completeness
+0.122). That is exactly where a 256-token output cap could start to bite — and
it was never measured. This probe measures it directly.

We do NOT need the judge here — only the GENERATOR's output length. Ollama hands
back two fields our generator normally discards, which make this exact (not a
word-count guess):
  done_reason == "length"  -> the answer hit the num_predict cap (TRUNCATED)
  done_reason == "stop"    -> the model finished its sentence on its own
  eval_count               -> the model's OWN count of tokens it generated

Method: reproduce Exp 12's pipeline FAITHFULLY — same config (pgc_complete), same
pooled index, same seed so the sample is a subset of the Exp 12 run — build the
real prompt via the real pipeline components, then make a metadata-capturing
Ollama call at the SAME 256-token cap Exp 12 used.

Decision rule:
  many answers done_reason="length" / eval_count clustered at ~256  -> cap BINDS
      -> build Exp 15 (raise to 512, full N=50, watch completeness + adherence).
  all "stop", eval_count well under ~200                            -> NULL (like 3B)
      -> close the line; the cap is not constraining the 8B either.

Run (8B is pulled locally; generation only, a few minutes; NO judge, NO git):
  .venv-eda/bin/python scripts/probe_answer_length.py --domain CustomerSupport --n 15 --max-new-tokens 256
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401 — registers light components (pgc chunker, grounded_complete prompt)
from src.embeddings import load_embedders
from src.generation import load_generators
from src.config import from_dict
from src.pipeline import build_pipeline
from src.data_loader import load_domain


# CustomerSupport's Exp 12 WINNER, verbatim from configs/grounded_pooled_pgc_complete.yaml,
# with the 8B generator (the --gen-model llama3.1:8b override Exp 12 used).
WINNER = {
    "domain": "Legal",                                    # tag only; we pass examples explicitly
    "chunker":   {"type": "pgc", "paragraphs": 2, "overlap": 1},
    "embedder":  {"type": "minilm"},
    "index":     {"type": "faiss", "corpus_mode": "pooled"},
    "retriever": {"type": "dense", "k": 20},
    "reranker":  {"type": "none", "top_n": 5},
    "repacker":  {"type": "reverse"},
    "prompt":    {"type": "grounded_complete"},
    "generator": {"type": "ollama", "model": "llama3.1:8b", "max_new_tokens": 256},
    "splitter":  {"type": "regex"},
    "seed": 42,
}


def _dedup(docs):
    """Order-preserving dedup — mirrors runner._dedup_docs (the pooled corpus is the union)."""
    seen, out = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d); out.append(d)
    return out


def build_prompt_for(pipe, query: str) -> str:
    """Replicate Pipeline.answer()'s chain UP TO the prompt (retrieve -> rerank ->
    repack -> [summarize] -> build), reusing the pipeline's own components so this is
    byte-identical to what Exp 12 fed the generator. We stop before generate() so we
    can make our own metadata-capturing call."""
    candidates = pipe.retriever.retrieve(query, k=pipe.k)
    if pipe.reranker is not None:
        chunks = pipe.reranker.rerank(query, candidates, top_n=pipe.top_n)
    else:
        chunks = candidates[: pipe.top_n]
    if pipe.repacker is not None:
        chunks = pipe.repacker.pack(chunks)
    if pipe.summarizer is not None:
        chunks = pipe.summarizer.compress(query, chunks)
    return pipe.prompt_builder.build(query, chunks)


def generate_with_meta(prompt: str, model: str, max_new_tokens: int,
                       host: str = "http://localhost:11434", num_ctx: int = 16384) -> dict:
    """Same Ollama call as OllamaGenerator.generate, but KEEP the metadata fields
    (done_reason, eval_count) the production generator drops."""
    import httpx
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.0, "num_predict": max_new_tokens, "num_ctx": num_ctx}}
    resp = httpx.post(f"{host}/api/generate", json=payload, timeout=600.0)
    resp.raise_for_status()
    d = resp.json()
    return {"text": d.get("response", ""),
            "done_reason": d.get("done_reason"),
            "eval_count": d.get("eval_count")}        # tokens the model generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="CustomerSupport")
    ap.add_argument("--n", type=int, default=15, help="examples to probe (seeded subset of Exp 12)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="the cap Exp 12 used")
    ap.add_argument("--model", default="llama3.1:8b")
    args = ap.parse_args()

    load_embedders()                                  # heavy minilm (pgc chunker is light)
    load_generators()                                 # registers the "ollama" generator (gated, like minilm)

    cfg = from_dict({**WINNER, "generator": {"type": "ollama", "model": args.model,
                                             "max_new_tokens": args.max_new_tokens}})
    pipe = build_pipeline(cfg)

    examples = load_domain(args.domain, split="test", n=args.n, seed=42)
    pipe.reset_index()
    corpus = _dedup([d for ex in examples for d in ex["documents"]])
    n_chunks = pipe.index_documents(corpus)

    print(f"\nAnswer-length probe — {args.domain}, pgc_complete @ {args.model}, "
          f"cap={args.max_new_tokens} (Exp 12 setting)")
    print(f"pooled index: {n_chunks} chunks from {len(corpus)} unique docs; n={len(examples)} questions\n")

    near = max(8, int(0.94 * args.max_new_tokens))    # "within ~6% of the cap" = effectively at it
    rows = []
    for i, ex in enumerate(examples):
        prompt = build_prompt_for(pipe, ex["question"])
        out = generate_with_meta(prompt, args.model, args.max_new_tokens)
        toks = out["eval_count"] or 0
        truncated = (out["done_reason"] == "length") or (toks >= near)
        rows.append({"toks": toks, "reason": out["done_reason"], "trunc": truncated})
        flag = "  <-- TRUNCATED" if truncated else ""
        print(f"  q{i+1:>2}: {toks:>4} tok  done={out['done_reason'] or '?':<6}{flag}")

    toks_list = [r["toks"] for r in rows]
    n_trunc = sum(r["trunc"] for r in rows)
    n_len_reason = sum(r["reason"] == "length" for r in rows)
    toks_list_sorted = sorted(toks_list)
    median = toks_list_sorted[len(toks_list_sorted) // 2] if toks_list_sorted else 0

    print("\n" + "=" * 64)
    print(f"  answers probed         : {len(rows)}")
    print(f"  median tokens          : {median}")
    print(f"  max tokens             : {max(toks_list) if toks_list else 0}   (cap = {args.max_new_tokens})")
    print(f"  done_reason='length'   : {n_len_reason}/{len(rows)}   (Ollama says it hit the cap)")
    print(f"  truncated (len OR >=94%): {n_trunc}/{len(rows)}")
    print("=" * 64)
    share = n_trunc / len(rows) if rows else 0
    if share >= 0.20:
        print(f"\n  VERDICT: cap BINDS ({share*100:.0f}% truncated) -> raising max_new_tokens is a")
        print(f"           real lever. Build Exp 15 (cap 512, full N=50, watch completeness+adherence).")
    else:
        print(f"\n  VERDICT: cap does NOT bind ({share*100:.0f}% truncated; median {median} << {args.max_new_tokens}).")
        print(f"           Same null as the 3B (per-example Exp 6A). Close the line — not a lever.")
    print()


if __name__ == "__main__":
    main()
