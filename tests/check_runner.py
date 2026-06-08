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
    assert cid == "fixed|fake|faiss|dense|none|reverse|grounded|echo"


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


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} runner checks passed.")


if __name__ == "__main__":
    _run()
