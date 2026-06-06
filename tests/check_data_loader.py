"""Checks for the data loader (src/data_loader.py).

Split into:
  * OFFLINE checks (maps, guards) — no network, run always.
  * SMOKE checks (real load_domain calls) — need the RAGBench dataset; run when
    DATASET=1 is set, e.g.:  DATASET=1 python tests/check_data_loader.py

Run directly:  python tests/check_data_loader.py            # offline only
               DATASET=1 python tests/check_data_loader.py  # + dataset smoke
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import configs_for, load_domain, load_config, DOMAINS, VALID_SPLITS

WANT_DATASET = os.environ.get("DATASET") == "1"


# ---------------- OFFLINE (no network) ----------------

def test_domain_map():
    assert configs_for("Legal") == ["cuad"]
    assert configs_for("Biomedical") == ["pubmedqa", "covidqa"]
    assert set(DOMAINS) == {"Biomedical", "GenKnowledge", "Legal", "CustomerSupport", "Finance"}


def test_unknown_domain_raises():
    try:
        configs_for("Medical")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_split_guard_rejects_bad_split():
    for bad in ("holdout", "dev", "Test", ""):
        try:
            load_domain("Legal", split=bad)
            assert False, f"expected ValueError for split {bad!r}"
        except ValueError:
            pass
    assert "test" in VALID_SPLITS and "validation" in VALID_SPLITS


# ---------------- SMOKE (need dataset) ----------------

def test_load_single_config_domain():
    rows = load_domain("Legal", split="test", n=10)
    assert len(rows) == 10
    assert all(r["_config"] == "cuad" for r in rows)
    assert "question" in rows[0] and "documents" in rows[0]


def test_load_multi_config_pools_and_tags():
    rows = load_domain("Biomedical", split="test", n=40)
    assert len(rows) == 40
    assert {r["_config"] for r in rows} <= {"pubmedqa", "covidqa"}


def test_sampling_is_reproducible():
    a = [r["id"] for r in load_domain("Biomedical", split="test", n=30, seed=1)]
    b = [r["id"] for r in load_domain("Biomedical", split="test", n=30, seed=1)]
    c = [r["id"] for r in load_domain("Biomedical", split="test", n=30, seed=2)]
    assert a == b and a != c


def test_full_split_size():
    assert len(load_config("cuad", split="test", n=None)) == 510   # per EDA


def _run():
    offline = [v for k, v in sorted(globals().items())
               if k.startswith("test_") and callable(v)
               and not k.startswith(("test_load", "test_sampling", "test_full"))]
    smoke = [globals()[k] for k in (
        "test_load_single_config_domain", "test_load_multi_config_pools_and_tags",
        "test_sampling_is_reproducible", "test_full_split_size")]

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
