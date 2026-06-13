"""Checks for the reference-comparison layer (src/evaluator/compare.py).

Split into:
  * OFFLINE checks (the join + gap math) — no network: we inject a fake `reference_fn`
    so build_comparison never touches the dataset. Run always.
  * SMOKE checks (real reference_means over RAGBench) — need the dataset; run when
    DATASET=1 is set.

Run directly:  python tests/check_compare.py            # offline only
               DATASET=1 python tests/check_compare.py  # + dataset smoke
"""

import sys, os, csv, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator.compare import (
    build_comparison, comparison_fieldnames, write_comparison_csv,
    reference_means, METRICS, _to_float, _mean_reference_fields,
    load_matrix_rows, plot_matrix,
)

WANT_DATASET = os.environ.get("DATASET") == "1"

# A tiny fake matrix CSV: 2 strategies × 1 domain, plus one row with a blank score
# (scoring pending) to exercise the None path.
MATRIX_ROWS = [
    {"config_name": "a.yaml", "domain": "GenKnowledge", "n": "10",
     "relevance": "0.40", "utilization": "0.20", "completeness": "0.30", "adherence": "0.80"},
    {"config_name": "b.yaml", "domain": "GenKnowledge", "n": "10",
     "relevance": "0.25", "utilization": "0.15", "completeness": "0.50", "adherence": "0.60"},
    {"config_name": "c.yaml", "domain": "GenKnowledge", "n": "10",
     "relevance": "", "utilization": "", "completeness": "", "adherence": ""},  # pending
]

# Fake reference: every metric's reference mean is 0.30 (constant baseline for the domain).
FAKE_REF = {"relevance": 0.30, "utilization": 0.30, "completeness": 0.30,
            "adherence": 0.30, "_n_ref": 10}


def _fake_reference_fn(domain, n=None, seed=42, split="test"):
    return dict(FAKE_REF)


def _write_matrix(rows):
    path = os.path.join(tempfile.gettempdir(), "_check_compare_matrix.csv")
    cols = ["config_name", "domain", "n", "relevance", "utilization", "completeness", "adherence"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------- OFFLINE (no network) ----------------

def test_to_float_handles_blanks_and_junk():
    assert _to_float("0.5") == 0.5
    assert _to_float("") is None
    assert _to_float(None) is None
    assert _to_float("n/a") is None


def test_mean_reference_fields_counts_adherence_as_rate():
    # 3 examples; adherence True/True/False -> rate 2/3. relevance mean = 0.3.
    examples = [
        {"relevance_score": 0.2, "utilization_score": 0.1, "completeness_score": 0.5, "adherence_score": True},
        {"relevance_score": 0.3, "utilization_score": 0.2, "completeness_score": 0.5, "adherence_score": True},
        {"relevance_score": 0.4, "utilization_score": 0.3, "completeness_score": 0.5, "adherence_score": False},
    ]
    means = _mean_reference_fields(examples)
    assert abs(means["relevance"] - 0.3) < 1e-9
    assert abs(means["adherence"] - 2 / 3) < 1e-9       # bool averaged as a rate
    assert means["_n_ref"] == 3


def test_build_comparison_adds_ref_and_gap():
    path = _write_matrix(MATRIX_ROWS)
    rows = build_comparison(path, reference_fn=_fake_reference_fn)
    assert len(rows) == 3
    a = rows[0]
    # ours=0.40, ref=0.30 -> gap +0.10 (strategy a beats the baseline on relevance)
    assert abs(a["relevance_ref"] - 0.30) < 1e-9
    assert abs(a["relevance_gap"] - 0.10) < 1e-9
    # b: ours=0.25 < ref 0.30 -> negative gap (below baseline -> retriever lagging)
    assert rows[1]["relevance_gap"] < 0
    os.remove(path)


def test_build_comparison_blank_scores_have_no_gap():
    path = _write_matrix(MATRIX_ROWS)
    rows = build_comparison(path, reference_fn=_fake_reference_fn)
    pending = rows[2]                                   # the c.yaml row (scoring pending)
    for m in METRICS:
        assert pending[f"{m}_ref"] is not None          # reference still computable
        assert pending[f"{m}_gap"] is None              # but no gap without our score
    os.remove(path)


def test_reference_means_cached_per_domain():
    # Same (domain, n) must reuse one reference computation, not re-call per row.
    calls = []

    def counting_fn(domain, n=None, seed=42, split="test"):
        calls.append((domain, n))
        return dict(FAKE_REF)

    path = _write_matrix(MATRIX_ROWS)
    build_comparison(path, reference_fn=counting_fn)
    assert calls == [("GenKnowledge", 10)]              # 3 rows, ONE reference call
    os.remove(path)


def test_match_n_false_uses_full_split():
    calls = []

    def counting_fn(domain, n=None, seed=42, split="test"):
        calls.append((domain, n))
        return dict(FAKE_REF)

    path = _write_matrix(MATRIX_ROWS)
    build_comparison(path, match_n=False, reference_fn=counting_fn)
    assert calls == [("GenKnowledge", None)]            # n=None -> whole-split baseline
    os.remove(path)


def test_fieldnames_group_metric_triples_after_base():
    path = _write_matrix(MATRIX_ROWS)
    rows = build_comparison(path, reference_fn=_fake_reference_fn)
    names = comparison_fieldnames(rows)
    # original columns first, untouched
    assert names[:3] == ["config_name", "domain", "n"]
    # then each metric's ref+gap pair, in METRICS order
    assert names[-8:] == [
        "relevance_ref", "relevance_gap", "utilization_ref", "utilization_gap",
        "completeness_ref", "completeness_gap", "adherence_ref", "adherence_gap"]
    os.remove(path)


def test_plot_comparison_writes_one_png_per_metric():
    # matplotlib is optional; skip cleanly if it isn't installed (the data layer is
    # what matters offline). When present, every metric should yield a PNG.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("  (skip plot test — matplotlib not installed)")
        return
    from src.evaluator.compare import plot_comparison
    path = _write_matrix(MATRIX_ROWS)
    rows = build_comparison(path, reference_fn=_fake_reference_fn)
    out_dir = os.path.join(tempfile.gettempdir(), "_check_compare_figs")
    pngs = plot_comparison(rows, out_dir, baseline_label="test")
    assert len(pngs) == len(METRICS)
    for p in pngs:
        assert os.path.exists(p) and os.path.getsize(p) > 0
        os.remove(p)
    os.remove(path)


def test_plot_import_does_not_force_matplotlib():
    # Heavy-dep discipline: importing compare.py must NOT pull matplotlib at module level.
    # NOTE: popping matplotlib mid-suite leaves it half-initialized for later tests that
    # plot (matplotlib.use() then crashes on the stale module). So snapshot every matplotlib
    # submodule, run the check, then RESTORE — this test must not corrupt global state.
    import sys, importlib
    saved = {k: v for k, v in sys.modules.items() if k == "matplotlib" or k.startswith("matplotlib.")}
    for k in list(saved):
        del sys.modules[k]
    try:
        importlib.reload(importlib.import_module("src.evaluator.compare"))
        assert "matplotlib" not in sys.modules
    finally:
        sys.modules.update(saved)                  # restore so later plot tests work


def test_load_matrix_rows_merges_and_dedupes():
    # Pooled runs split across files; load_matrix_rows merges them, de-duping on
    # (config_name, domain) keeping the LAST occurrence (a re-run supersedes the old row).
    p1 = _write_matrix([
        {"config_name": "a.yaml", "domain": "GenKnowledge", "n": "50",
         "relevance": "0.30", "utilization": "0.10", "completeness": "0.40", "adherence": "0.50"}])
    p2 = os.path.join(tempfile.gettempdir(), "_check_compare_matrix2.csv")
    cols = ["config_name", "domain", "n", "relevance", "utilization", "completeness", "adherence"]
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        w.writerow({"config_name": "b.yaml", "domain": "GenKnowledge", "n": "50",
                    "relevance": "0.40", "utilization": "0.20", "completeness": "0.50", "adherence": "0.60"})
        # a re-run of a.yaml with a DIFFERENT score -> should supersede p1's a.yaml row
        w.writerow({"config_name": "a.yaml", "domain": "GenKnowledge", "n": "50",
                    "relevance": "0.99", "utilization": "0.10", "completeness": "0.40", "adherence": "0.50"})
    rows = load_matrix_rows(p1, p2)
    assert len(rows) == 2                                       # a + b, a de-duped
    a = next(r for r in rows if r["config_name"] == "a.yaml")
    assert a["relevance"] == "0.99"                            # last occurrence wins
    os.remove(p1); os.remove(p2)


def test_plot_matrix_one_png_per_domain():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("  (skip plot_matrix test — matplotlib not installed)")
        return
    rows = [
        {"config_name": "a.yaml", "domain": "GenKnowledge", "relevance": "0.3",
         "utilization": "0.1", "completeness": "0.4", "adherence": "0.5"},
        {"config_name": "b.yaml", "domain": "GenKnowledge", "relevance": "0.4",
         "utilization": "0.2", "completeness": "0.5", "adherence": "0.6"},
        {"config_name": "a.yaml", "domain": "CustomerSupport", "relevance": "0.2",
         "utilization": "0.1", "completeness": "0.1", "adherence": "0.5"},
    ]
    out_dir = os.path.join(tempfile.gettempdir(), "_check_matrix_figs")
    pngs = plot_matrix(rows, out_dir, title_prefix="pooled — ")
    assert len(pngs) == 2                                       # one per domain
    for p in pngs:
        assert os.path.exists(p) and os.path.getsize(p) > 0
        os.remove(p)


def test_write_comparison_csv_roundtrips():
    path = _write_matrix(MATRIX_ROWS)
    rows = build_comparison(path, reference_fn=_fake_reference_fn)
    out = os.path.join(tempfile.gettempdir(), "_check_compare_out.csv")
    write_comparison_csv(rows, out)
    with open(out, newline="") as f:
        back = list(csv.DictReader(f))
    assert len(back) == 3
    assert "relevance_gap" in back[0]
    assert abs(float(back[0]["relevance_gap"]) - 0.10) < 1e-9
    os.remove(path); os.remove(out)


# ---------------- SMOKE (need dataset) ----------------

def test_reference_means_real_domain():
    means = reference_means("GenKnowledge", n=10, seed=42)
    assert means["_n_ref"] == 10
    for m in METRICS:
        assert 0.0 <= means[m] <= 1.0                   # all TRACe scores are fractions/rates


def test_reference_means_matches_matrix_sample():
    # The whole point: same seed+n as the matrix -> a fair head-to-head baseline.
    # Reproducible across calls.
    a = reference_means("CustomerSupport", n=10, seed=42)
    b = reference_means("CustomerSupport", n=10, seed=42)
    assert a == b


def _run():
    offline = [v for k, v in sorted(globals().items())
               if k.startswith("test_") and callable(v) and "real" not in k and "matches_matrix" not in k]
    smoke = [globals()[k] for k in ("test_reference_means_real_domain",
                                    "test_reference_means_matches_matrix_sample")]

    for fn in offline:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"{len(offline)} offline checks passed.")

    if WANT_DATASET:
        print("\nrunning dataset smoke checks (DATASET=1)...")
        for fn in smoke:
            fn(); print(f"  ✅ {fn.__name__}")
        print(f"{len(smoke)} smoke checks passed.")
    else:
        print("\n(skipped dataset smoke checks — set DATASET=1 to run them)")


if __name__ == "__main__":
    _run()
