"""Fast, OFFLINE regression checks for the TRACe math (src/evaluator/trace.py).

The TRACe math is the graded core of the project; this guards it against future
refactors. Pure arithmetic on hardcoded inputs — no dataset download, no network,
runs in milliseconds.

Run directly:   python tests/check_trace.py
Or (if pytest is ever added) it is auto-discovered: the test_* functions use asserts.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator.trace import (
    total_doc_sentences, relevance, utilization, completeness, adherence,
)


def test_total_doc_sentences():
    # 3 docs with 2, 3, 1 sentences -> 6. Each sentence is a [key, text] pair.
    docs = [
        [["0a", "x"], ["0b", "y"]],
        [["1a", "x"], ["1b", "y"], ["1c", "z"]],
        [["2a", "x"]],
    ]
    assert total_doc_sentences(docs) == 6
    assert total_doc_sentences([]) == 0


def test_relevance_and_utilization():
    # 11 relevant of 18 -> 0.6111...;  9 utilized of 18 -> 0.5  (the validated pubmedqa[0] case)
    R = [f"k{i}" for i in range(11)]
    U = [f"k{i}" for i in range(9)]
    assert abs(relevance(R, 18) - 11 / 18) < 1e-12
    assert abs(utilization(U, 18) - 9 / 18) < 1e-12
    # zero-denominator guard
    assert relevance([], 0) == 0.0
    assert utilization([], 0) == 0.0


def test_relevance_uses_list_not_set():
    # EDA gotcha: duplicates are COUNTED (list semantics), to match reference scores.
    assert relevance(["a", "a", "b"], 10) == 3 / 10      # not 2/10
    assert utilization(["a", "a"], 10) == 2 / 10


def test_completeness():
    # |R∩U| / |R|. R={a,b,c,d}, U={a,b} -> 2/4 = 0.5
    assert completeness(["a", "b", "c", "d"], ["a", "b", "x"]) == 0.5
    # intersection is set-like even though counts are list-like
    assert completeness(["a", "b"], ["a", "a", "b"]) == 1.0
    # |R|==0 -> defined as 1.0 (frequent in CUAD)
    assert completeness([], ["a"]) == 1.0


def test_adherence():
    # Adherence == (unsupported list is empty). NOT derived from fully_supported flags.
    assert adherence([]) is True
    assert adherence(["a"]) is False
    assert adherence(["a", "b"]) is False


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} trace checks passed.")


if __name__ == "__main__":
    _run_all()
