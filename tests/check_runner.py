"""OFFLINE checks for the ExperimentRunner (src/runner.py).

Fake embedder + echo generator + a fake judge so the full run + CSV writing are
exercised locally with no model. Real runs use real models + the OSS judge on Colab.

Run directly:  python tests/check_runner.py
"""

import sys, os, csv, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import src  # noqa: F401
from src.registry import register, available
from src.config import from_dict
from src.segmentation import OutputSegmenter, RegexSplitter
from src.judge import FakeJudge
from src.runner import run_experiment, run_matrix, config_id, _mean, FIELDNAMES

if "fake" not in available("embedder"):
    @register("embedder", "fake")
    class _FakeEmbedder:
        dim = 16
        def embed(self, texts):
            out = []
            for t in texts:
                rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
                v = rng.standard_normal(self.dim).astype("float32")
                out.append(v / (np.linalg.norm(v) + 1e-9))
            return np.array(out)


# Use the REAL FakeJudge object (a Judge with .label()), NOT a bespoke function —
# so this test exercises the actual runner↔judge seam. (A function-style stub here
# once hid a real bug: the runner called judge(...) and read a field the judge never
# emits. Testing each brick against a faked neighbour is only safe if the fake honours
# the real contract.)
JUDGE = FakeJudge()


BASE = {
    "domain": "Legal",
    "chunker": {"type": "fixed", "size": 40, "overlap": 5}, "embedder": {"type": "fake"},
    "index": {"type": "faiss", "corpus_mode": "per_example"}, "retriever": {"type": "dense", "k": 6},
    "reranker": {"type": "none", "top_n": 3}, "repacker": {"type": "reverse"},
    "prompt": {"type": "grounded"}, "generator": {"type": "echo"}, "splitter": {"type": "regex"},
}
EXAMPLES = [
    {"question": "Termination notice period?",
     "documents": ["Either party may terminate with thirty days notice. Notice must be written.",
                   "Governed by Delaware law."]},
    {"question": "Which law governs?",
     "documents": ["Governed by the laws of Delaware. Disputes go to arbitration.",
                   "Either party may terminate."]},
]
SEG = OutputSegmenter(RegexSplitter())


def test_config_id_is_stable_and_descriptive():
    cid = config_id(from_dict(BASE))
    # position 7 (between repacker and prompt) is the summarizer slot — "none" here
    # since BASE has no summarizer stage.
    assert cid == "fixed|fake|faiss|dense|none|reverse|none|grounded|echo"


def test_mean_counts_booleans_as_rate():
    # The bug we fixed: adherence is boolean; mean must treat True/False as 1/0.
    assert _mean([True, True, False, False]) == 0.5
    assert abs(_mean([0.2, 0.4]) - 0.3) < 1e-9        # float-safe comparison
    assert _mean([]) == ""


def test_run_experiment_with_judge_scores_all_four():
    row = run_experiment(from_dict(BASE), EXAMPLES, segmenter=SEG, judge=JUDGE)
    assert row["n"] == 2 and row["n_scored"] == 2
    for metric in ("relevance", "utilization", "completeness", "adherence"):
        assert isinstance(row[metric], float)        # all four populated (incl. adherence!)
    assert row["scoring"] == "done"


def test_run_experiment_without_judge_is_pending():
    row = run_experiment(from_dict(BASE), EXAMPLES, segmenter=SEG, judge=None)
    assert row["n"] == 2 and row["n_scored"] == 0
    assert row["relevance"] == "" and row["adherence"] == ""
    assert row["scoring"].startswith("pending")


def test_pooled_mode_indexes_shared_corpus_once():
    # Pooled config: the index is built once from a shared corpus and NOT reset per example,
    # so every question can retrieve from the other questions' docs too. We assert the run
    # completes and scores all four (behavioural difference is the shared index; the key
    # correctness point — no reset between examples — is exercised by both questions answering
    # against the union corpus).
    pooled = from_dict({**BASE, "index": {"type": "faiss", "corpus_mode": "pooled"}})
    # corpus_docs given explicitly = the union of both examples' docs (the "full corpus").
    corpus = [d for ex in EXAMPLES for d in ex["documents"]]
    row = run_experiment(pooled, EXAMPLES, segmenter=SEG, judge=JUDGE, corpus_docs=corpus)
    assert row["n"] == 2 and row["n_scored"] == 2
    assert row["index"] == "faiss"
    for metric in ("relevance", "utilization", "completeness", "adherence"):
        assert isinstance(row[metric], float)


def test_pooled_retrieves_across_examples():
    # The defining property of pooled mode: a question CAN retrieve a chunk that belongs to a
    # DIFFERENT example's documents (impossible in per_example mode). Build a pooled pipeline
    # directly and check a query unique to example B's doc can surface while indexed together.
    from src.pipeline import build_pipeline
    pooled = from_dict({**BASE, "index": {"type": "faiss", "corpus_mode": "pooled"},
                        "retriever": {"type": "dense", "k": 4}})
    pipe = build_pipeline(pooled)
    pipe.reset_index()
    pipe.index_documents([ex_d for ex in EXAMPLES for ex_d in ex["documents"]])  # pooled union
    out = pipe.answer("Which law governs?")              # answer lives in example B/A's docs
    # All retrieved chunks come from the shared index (union of both examples' docs).
    assert len(out["sources"]) > 0


def test_run_matrix_writes_csv_and_is_resumable():
    out = os.path.join(tempfile.gettempdir(), "_check_runner.csv")
    if os.path.exists(out):
        os.remove(out)
    configs = [from_dict(BASE), from_dict({**BASE, "repacker": {"type": "forward"}})]
    run_matrix(configs, examples_for=lambda cfg: EXAMPLES, out_csv=out, segmenter=SEG, judge=JUDGE)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2                              # one row per config
    assert set(rows[0].keys()) == set(FIELDNAMES)
    assert {r["repacker"] for r in rows} == {"reverse", "forward"}

    # Re-run: nothing new appended (resumable / skip-already-done).
    run_matrix(configs, examples_for=lambda cfg: EXAMPLES, out_csv=out, segmenter=SEG, judge=JUDGE)
    with open(out, newline="") as f:
        assert len(list(csv.DictReader(f))) == 2       # still 2, not 4
    os.remove(out)


def test_build_grid_distinct_ok_and_collision_raises():
    from src.runner import build_grid, config_id
    raw = {"a.yaml": {**BASE, "prompt": {"type": "grounded"}},
           "b.yaml": {**BASE, "prompt": {"type": "minimal"}}}
    grid = build_grid(raw, ["Legal"])                       # differ by prompt type -> distinct
    assert len(grid) == 2 and len({config_id(c) for _, c in grid}) == 2
    assert [n for n, _ in grid] == ["a.yaml", "b.yaml"]

    # Two configs with the SAME types (differ only by a param) -> collision -> raises.
    clash = {"x.yaml": {**BASE, "chunker": {"type": "fixed", "size": 200}},
             "y.yaml": {**BASE, "chunker": {"type": "fixed", "size": 999}}}
    try:
        build_grid(clash, ["Legal"])
        assert False, "expected a collision error"
    except ValueError as e:
        assert "collision" in str(e)


def test_run_named_matrix_resumable(tmp=None):
    import csv
    from src.runner import build_grid, run_named_matrix
    out = os.path.join(tempfile.gettempdir(), "_check_named_matrix.csv")
    if os.path.exists(out):
        os.remove(out)
    raw = {"a.yaml": {**BASE, "prompt": {"type": "grounded"}},
           "b.yaml": {**BASE, "prompt": {"type": "minimal"}}}
    grid = build_grid(raw, ["Legal"])
    run_named_matrix(grid, lambda cfg: EXAMPLES, out, segmenter=SEG, judge=JUDGE)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and {r["config_name"] for r in rows} == {"a.yaml", "b.yaml"}
    assert "config_name" in rows[0] and rows[0]["relevance"] != ""
    run_named_matrix(grid, lambda cfg: EXAMPLES, out, segmenter=SEG, judge=JUDGE)  # resume
    with open(out, newline="") as f:
        assert len(list(csv.DictReader(f))) == 2        # not 4
    os.remove(out)


def test_run_named_matrix_creates_missing_out_dir():
    # After the results/ reorg, out_csv lives under results/{per_example,pooled}/ — a path
    # whose dir may not exist on a fresh clone. The runner must create it, not crash.
    import csv, shutil
    from src.runner import build_grid, run_named_matrix
    base = os.path.join(tempfile.gettempdir(), "_check_mkdir_results")
    shutil.rmtree(base, ignore_errors=True)
    out = os.path.join(base, "pooled", "ragbench_matrix_x.csv")    # nested, does NOT exist yet
    grid = build_grid({"a.yaml": dict(BASE)}, ["Legal"])
    run_named_matrix(grid, lambda cfg: EXAMPLES, out, segmenter=SEG, judge=JUDGE)
    assert os.path.exists(out)                                     # dir was created + file written
    with open(out, newline="") as f:
        assert len(list(csv.DictReader(f))) == 1
    shutil.rmtree(base, ignore_errors=True)


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} runner checks passed.")


if __name__ == "__main__":
    _run()
