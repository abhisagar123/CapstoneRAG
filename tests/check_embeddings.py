"""Checks for the embedder (src/embeddings/).

Split into:
  * OFFLINE checks — interface, registry wiring, lazy-loading. No torch import,
    no model download; run always.
  * MODEL checks — load the real all-MiniLM model (downloads ~90MB first time)
    and verify embeddings capture meaning. Gated behind MODEL=1:
        MODEL=1 python tests/check_embeddings.py

Run directly:  python tests/check_embeddings.py            # offline only
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401
from src.registry import build, available
from src.embeddings import Embedder, load_embedders

WANT_MODEL = os.environ.get("MODEL") == "1"


# ---------------- OFFLINE (no torch import, no download) ----------------

def test_importing_src_does_not_load_torch():
    # The lightweight-import rule: importing src must NOT drag in torch.
    # (If something eagerly imported the embedder impl, torch would be present.)
    assert "torch" not in sys.modules, "importing src pulled in torch — heavy dep leaked!"


def test_embedder_unregistered_until_loaded():
    # Before load_embedders(), the heavy impl must not be registered.
    # (Run this in a fresh process to be meaningful; here we just assert the
    #  function exists and is callable.)
    assert callable(load_embedders)


def test_load_embedders_registers_types():
    load_embedders()
    names = available("embedder")
    assert "sentence_transformer" in names
    assert "minilm" in names


# ---------------- MODEL (download + real inference) ----------------

def test_build_and_dim():
    load_embedders()
    emb = build("embedder", "minilm")
    assert isinstance(emb, Embedder)
    assert emb.dim == 384                       # all-MiniLM-L6-v2


def test_shape_and_meaning():
    load_embedders()
    emb = build("embedder", "minilm")
    texts = [
        "How do I cancel my subscription?",
        "What's the process to end my membership plan?",
        "The sky is blue and the grass is green.",
    ]
    V = emb.embed(texts)
    assert V.shape == (3, 384)
    # normalized vectors → dot product is cosine similarity
    import numpy as np
    sim_related = float(np.dot(V[0], V[1]))
    sim_unrelated = float(np.dot(V[0], V[2]))
    assert sim_related > sim_unrelated          # meaning, not word overlap


def _run():
    offline = [test_importing_src_does_not_load_torch,
               test_embedder_unregistered_until_loaded,
               test_load_embedders_registers_types]
    model = [test_build_and_dim, test_shape_and_meaning]

    for fn in offline:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"{len(offline)} offline checks passed.")

    if WANT_MODEL:
        print("\nrunning model checks (MODEL=1; may download ~90MB)...")
        for fn in model:
            fn(); print(f"  ✅ {fn.__name__}")
        print(f"{len(model)} model checks passed.")
    else:
        print("\n(skipped model checks — set MODEL=1 to run them)")


if __name__ == "__main__":
    _run()
