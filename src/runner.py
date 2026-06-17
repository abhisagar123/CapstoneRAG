"""ExperimentRunner — run configs over domains and write the results matrix.

This produces the headline RAGBench deliverable: one CSV row per (config × domain)
with the config's settings + its mean TRACe scores. "An experiment = a config."

Per-experiment flow (for N examples of a domain):
    pipeline.answer(question) -> segment(context, answer) -> [judge] -> TRACe score
Aggregate the N examples' scores -> one summary row.

THE JUDGE IS PLUGGABLE + OPTIONAL (it's deferred — needs a strong model / key):
  - judge=None   -> run pipeline + segment only; row records that scoring is pending
    (still useful: confirms the pipeline runs end-to-end and logs counts).
  - judge=Judge  -> a Judge object (judge/base.py): judge.label(question, keyed)
    returns RAGBench-style labels, which scores_from_label() maps to the validated
    TRACe math. Drop the OSS/OpenAI judge in here later, no other change.

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
from .judge.base import scores_from_label


def config_id(cfg) -> str:
    """A short stable identifier for a config — the stage types joined.
    e.g. 'fixed|minilm|faiss|dense|cross_encoder|reverse|none|grounded|hf'.
    (Position 7 is the summarizer; 'none' when no compression stage.)"""
    parts = [cfg.chunker.type, cfg.embedder.type, cfg.index.type, cfg.retriever.type,
             cfg.reranker.type if cfg.reranker else "none",
             cfg.repacker.type if cfg.repacker else "none",
             getattr(cfg, "summarizer", None).type if getattr(cfg, "summarizer", None) else "none",
             cfg.prompt.type, cfg.generator.type]
    return "|".join(parts)


def _score_one(example, answer: str, segmenter: OutputSegmenter, judge) -> dict | None:
    """Segment our output, ask the judge for labels, compute the 4 TRACe scores.
    Returns None if no judge (scoring deferred).

    The judge is a Judge object (judge/base.py): call its `.label(question, keyed)`
    method, then map the label JSON to the 4 scores with `scores_from_label` — the
    same validated adapter the judge-validation harness uses, so scoring goes through
    one path. (Adherence comes from the judge's `overall_supported` bool, §9.6.)
    """
    if judge is None:
        return None
    # Build keyed sentences from OUR context (the retrieved chunk texts) + answer.
    keyed = segmenter.segment(example["_context_texts"], answer)
    try:
        label = judge.label(example["question"], keyed)   # judge returns RAGBench-style label dict
        return scores_from_label(keyed, label)
    except Exception as e:
        # The judge can fail in many ways: unparseable JSON (ValueError/KeyError) OR a
        # network timeout/drop (e.g. httpx.ReadTimeout on a giant context). ANY of these
        # must skip this ONE answer, not crash the whole config's matrix row; the caller
        # counts it via (n - n_scored). (KeyboardInterrupt isn't an Exception, so Ctrl-C
        # still stops cleanly.) Same robustness rule as evaluator/judge_validate.py.
        print(f"  [skip] judge failed on one example ({type(e).__name__}: {e})")
        return None


def _mean(values: list) -> float | str:
    """Mean of a list. Booleans count as 0/1 (so adherence becomes a True-rate).
    Empty -> "" (blank CSV cell, e.g. when scoring is pending)."""
    nums = [float(v) for v in values if isinstance(v, (int, float, bool))]
    return sum(nums) / len(nums) if nums else ""

def _sum_counts(values: list) -> float | str:
    """Sum of a list of numeric micro-values. Same blank-on-empty behavior as _mean().
    Use this INSTEAD of _mean() below if you want TOTAL counts across all N examples
    for a config, rather than a per-example average."""
    nums = [float(v) for v in values if isinstance(v, (int, float, bool))]
    return sum(nums) if nums else ""


def _dedup_docs(docs) -> list[str]:
    """Unique documents, order-preserving (the pooled corpus is the dedup'd union)."""
    seen, out = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def run_experiment(cfg, examples, *, segmenter: OutputSegmenter, judge=None,
                   corpus_docs=None) -> dict:
    """Run ONE config over a list of examples; return one aggregated result row.

    `examples`: RAGBench dicts (need 'question' and 'documents'). We ANSWER + SCORE
    these N examples.

    Corpus mode (read from cfg.index.params['corpus_mode']) decides WHAT the retriever
    searches over:
      * 'per_example' (default): for each question, the index holds ONLY that question's
        own documents (matches RAGBench's reference-score universe). The index is reset
        between examples. This is the validated path used by Exp 1-6.
      * 'pooled': the index is built ONCE from a shared corpus and every question retrieves
        from it (real needle-in-haystack RAG). The corpus is `corpus_docs` if given (e.g.
        the FULL domain's documents), else the dedup'd union of the examples' own docs.
        ⚠️ In pooled mode our retrieval universe ≠ the reference's per-question universe, so
        relevance/utilization/completeness are valid as OUR-OWN scores but are NOT
        apples-to-apples vs the RAGBench reference (adherence still is). We compute all four
        regardless; whether to compare to reference is decided later from the results.
    """
    pipe = build_pipeline(cfg)
    pooled = cfg.index.params.get("corpus_mode") == "pooled"
    n, n_scored = 0, 0

    score_lists = {"relevance": [], "utilization": [], "completeness": [], "adherence": [],
                   "relevant_count": [], "total_sentences": [], "utilized_count": [],
                   "overlap_count": [], "unsupported_count": []}

    if pooled:
        # Build the shared index ONCE, then never reset between questions.
        docs = corpus_docs if corpus_docs is not None else [d for ex in examples for d in ex["documents"]]
        pipe.reset_index()
        n_chunks = pipe.index_documents(_dedup_docs(docs))
        print(f"   [pooled] indexed {n_chunks} chunks from {len(_dedup_docs(docs))} unique docs", flush=True)

    for ex in examples:
        if not pooled:
            # Per-example corpus: clear the index, index just this question's docs.
            pipe.reset_index()
            pipe.index_documents(ex["documents"])
        # pooled: the shared index built above is reused for every question (no reset).
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
        "summarizer": getattr(cfg, "summarizer", None).type if getattr(cfg, "summarizer", None) else "none",
        "prompt": cfg.prompt.type, "generator": cfg.generator.type,
        "n": n, "n_scored": n_scored,
        "relevance": _mean(score_lists["relevance"]),
        "utilization": _mean(score_lists["utilization"]),
        "completeness": _mean(score_lists["completeness"]),
        "adherence": _mean(score_lists["adherence"]),
        # Micro-values behind each score (see judge/base.py scores_from_label).
        # Using _mean() to match the same per-example-average convention as the 4
        # scores above. Swap to _sum_counts(...) on any line below if you'd rather
        # see TOTALS across all N examples for that column instead.
        "relevant_count": _mean(score_lists["relevant_count"]),
        "total_sentences": _mean(score_lists["total_sentences"]),
        "utilized_count": _mean(score_lists["utilized_count"]),
        "overlap_count": _mean(score_lists["overlap_count"]),
        "unsupported_count": _mean(score_lists["unsupported_count"]),
        "scoring": "done" if n_scored else "pending (no judge)",
    }

FIELDNAMES = ["config_id", "domain", "chunker", "embedder", "index", "retriever",
              "reranker", "repacker", "summarizer", "prompt", "generator", "n", "n_scored",
              "relevance", "utilization", "completeness", "adherence",
              "relevant_count", "total_sentences", "utilized_count", "overlap_count",
              "unsupported_count",
              "scoring"]


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


# ── named-matrix runner (shared by nb04 AND scripts/run_matrix.py) ─────────────────
# Like run_matrix but (a) carries a config_NAME column (the YAML filename) so configs
# differing only in a param don't collide, and (b) writes each row by RE-OPENING the
# file (open->write->close) — survives a mid-run file swap, same fix as the judge sweep.

NAMED_FIELDNAMES = ["config_name"] + FIELDNAMES


def build_grid(raw_configs: dict, domains, *, generator_override: dict | None = None):
    """Expand {config_name: raw_yaml_dict} × domains into [(config_name, PipelineConfig)].

    Each config is cloned per domain (overriding `domain`), optionally replacing the
    generator stage (e.g. to point all configs at one local model). Asserts no two
    (config_id, domain) pairs collide — config_id is built from stage TYPES, so a
    param-only difference would silently overwrite a row; this fails loudly instead.
    """
    import copy
    from collections import Counter
    from .config import from_dict

    grid = []
    for name, raw in raw_configs.items():
        for dom in domains:
            d = copy.deepcopy(raw)
            d["domain"] = dom
            if generator_override is not None:
                d["generator"] = dict(generator_override)
            grid.append((name, from_dict(d, do_validate=True)))

    clash = [k for k, c in Counter((config_id(cfg), cfg.domain) for _, cfg in grid).items() if c > 1]
    if clash:
        raise ValueError(f"config_id collision (would overwrite matrix rows): {clash}")
    return grid


def _named_done(out_csv: str) -> set:
    """(config_name, domain) pairs already in the CSV — for resume. Re-read each call."""
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv, newline="") as f:
        return {(r["config_name"], r["domain"]) for r in csv.DictReader(f)}


def run_named_matrix(grid, examples_for, out_csv: str, *, segmenter=None, judge=None,
                     corpus_for=None) -> None:
    """Run a [(config_name, cfg)] grid -> one CSV row per (config_name, domain).

    Resumable AND file-swap-safe: re-reads the done-set and re-opens the file per row
    (open->write->close), so a git/IDE replacement mid-run can't orphan writes (the
    bug that hit the judge sweep). examples_for(cfg) supplies that config's examples.

    `corpus_for(cfg)` (optional): for POOLED configs, returns the shared corpus docs to
    index once (e.g. the full domain's documents). None (default) → per-example mode uses
    each question's own docs. Ignored by per_example configs.
    """
    # Ensure the output dir exists (out_csv now lives under results/{per_example,pooled}/ —
    # a fresh clone won't have the subdir until first write).
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for name, cfg in grid:
        if (name, cfg.domain) in _named_done(out_csv):
            print(f"skip (done): {name} / {cfg.domain}")
            continue
        print(f"running {name} on {cfg.domain} ...", flush=True)
        corpus = corpus_for(cfg) if corpus_for is not None else None
        row = {"config_name": name,
               **run_experiment(cfg, examples_for(cfg), segmenter=segmenter, judge=judge,
                                corpus_docs=corpus)}
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:          # re-open PER ROW (swap-safe)
            w = csv.DictWriter(f, fieldnames=NAMED_FIELDNAMES)
            if write_header:
                w.writeheader()
            w.writerow(row)
            f.flush()
        print(f"   -> rel={row['relevance']}  util={row['utilization']}  "
              f"compl={row['completeness']}  adh={row['adherence']}  (scored {row['n_scored']}/{row['n']})")
    print(f"done. wrote {out_csv}")
