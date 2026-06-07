"""ExperimentRunner — run configs over domains and write the results matrix.

This produces the headline RAGBench deliverable: one CSV row per (config × domain)
with the config's settings + its mean TRACe scores. "An experiment = a config."

Per-experiment flow (for N examples of a domain):
    pipeline.answer(question) -> segment(context, answer) -> [judge] -> TRACe score
Aggregate the N examples' scores -> one summary row.

THE JUDGE IS PLUGGABLE + OPTIONAL (it's deferred — needs a strong model / key):
  - judge=None  -> run pipeline + segment only; row records that scoring is pending
    (still useful: confirms the pipeline runs end-to-end and logs counts).
  - judge=fn    -> fn(question, keyed) returns RAGBench-style labels; we score with
    the validated TRACe math. Drop the OSS/OpenAI judge in here later, no other change.

Robustness (Colab dies mid-sweep):
  - rows are written to the CSV AS each experiment finishes (not all at the end),
  - configs whose (config_id, domain) is already in the CSV are SKIPPED,
  so a re-run resumes where it stopped.
"""

import csv
import json
import os

from .pipeline import build_pipeline
from .segmentation import OutputSegmenter
from .evaluator.trace import (
    total_doc_sentences, relevance, utilization, completeness, adherence,
)


def config_id(cfg) -> str:
    """A short stable identifier for a config — the stage types joined.
    e.g. 'fixed|minilm|faiss|dense|cross_encoder|reverse|grounded|hf'."""
    parts = [cfg.chunker.type, cfg.embedder.type, cfg.index.type, cfg.retriever.type,
             cfg.reranker.type if cfg.reranker else "none",
             cfg.repacker.type if cfg.repacker else "none",
             cfg.prompt.type, cfg.generator.type]
    return "|".join(parts)


def _score_one(example, answer: str, segmenter: OutputSegmenter, judge) -> dict | None:
    """Segment our output, ask the judge for labels, compute the 4 TRACe scores.
    Returns None if no judge (scoring deferred)."""
    if judge is None:
        return None
    # Build keyed sentences from OUR context (the retrieved chunk texts) + answer.
    keyed = segmenter.segment(example["_context_texts"], answer)
    labels = judge(example["question"], keyed)        # judge returns RAGBench-style label dict
    total = total_doc_sentences(keyed["documents_sentences"])
    R = labels["all_relevant_sentence_keys"]
    U = labels["all_utilized_sentence_keys"]
    return {
        "relevance": relevance(R, total),
        "utilization": utilization(U, total),
        "completeness": completeness(R, U),
        "adherence": adherence(labels["unsupported_response_sentence_keys"]),
    }


def _mean(values: list) -> float | str:
    """Mean of a list. Booleans count as 0/1 (so adherence becomes a True-rate).
    Empty -> "" (blank CSV cell, e.g. when scoring is pending)."""
    nums = [float(v) for v in values if isinstance(v, (int, float, bool))]
    return sum(nums) / len(nums) if nums else ""


def run_experiment(cfg, examples, *, segmenter: OutputSegmenter, judge=None) -> dict:
    """Run ONE config over a list of examples; return one aggregated result row.

    `examples`: RAGBench dicts (need 'question' and 'documents'). Each example's
    own documents are indexed per-example (corpus_mode 'per_example').
    """
    pipe = build_pipeline(cfg)
    n, n_scored = 0, 0
    score_lists = {"relevance": [], "utilization": [], "completeness": [], "adherence": []}

    for ex in examples:
        # Per-example corpus: clear the index, index just this question's docs, answer.
        pipe.reset_index()
        pipe.index_documents(ex["documents"])
        out = pipe.answer(ex["question"])
        n += 1

        if judge is not None:
            ex_ctx = {**ex, "_context_texts": [rc.chunk.text for rc in out["sources"]]}
            s = _score_one(ex_ctx, out["answer"], segmenter, judge)
            if s is not None:
                n_scored += 1
                for k in score_lists:
                    score_lists[k].append(s[k])

    return {
        "config_id": config_id(cfg),
        "domain": cfg.domain,
        "chunker": cfg.chunker.type, "embedder": cfg.embedder.type,
        "index": cfg.index.type, "retriever": cfg.retriever.type,
        "reranker": cfg.reranker.type if cfg.reranker else "none",
        "repacker": cfg.repacker.type if cfg.repacker else "none",
        "prompt": cfg.prompt.type, "generator": cfg.generator.type,
        "n": n, "n_scored": n_scored,
        "relevance": _mean(score_lists["relevance"]),
        "utilization": _mean(score_lists["utilization"]),
        "completeness": _mean(score_lists["completeness"]),
        "adherence": _mean(score_lists["adherence"]),
        "scoring": "done" if n_scored else "pending (no judge)",
    }


FIELDNAMES = ["config_id", "domain", "chunker", "embedder", "index", "retriever",
              "reranker", "repacker", "prompt", "generator", "n", "n_scored",
              "relevance", "utilization", "completeness", "adherence", "scoring"]


def _already_done(out_csv: str) -> set:
    """Return the set of (config_id, domain) pairs already present in the CSV."""
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv, newline="") as f:
        return {(r["config_id"], r["domain"]) for r in csv.DictReader(f)}


def run_matrix(configs, examples_for, out_csv: str, *, segmenter=None, judge=None) -> None:
    """Run many configs, writing one row each to out_csv AS THEY FINISH (resumable).

    `examples_for(cfg)` -> the list of examples to run for that config (lets the
    caller load each config's domain). Already-done (config_id, domain) pairs are
    skipped so a re-run resumes after a Colab disconnect.
    """
    done = _already_done(out_csv)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for cfg in configs:
            key = (config_id(cfg), cfg.domain)
            if key in done:
                print(f"skip (already done): {key}")
                continue
            row = run_experiment(cfg, examples_for(cfg), segmenter=segmenter, judge=judge)
            writer.writerow(row)
            f.flush()                                   # persist immediately (Colab-safe)
            print(f"done: {key}  n={row['n']}  scoring={row['scoring']}")
