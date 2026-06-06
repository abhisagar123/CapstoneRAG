"""OFFLINE checks for the chunker + the registry swap mechanism (src/chunking.py,
src/registry.py). No network, no dataset — pure logic, runs in milliseconds.

Run directly:  python tests/check_chunking.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401 — triggers component registration via src/__init__.py
from src.registry import build, available, register, REGISTRY
from src.chunking import Chunk, Chunker, FixedChunker, NoOpChunker


def test_components_registered_on_import():
    # Importing src must populate the registry (the decorator-runs-on-import gotcha).
    assert "fixed" in available("chunker")
    assert "none" in available("chunker")


def test_fixed_chunker_splits_long_doc():
    doc = " ".join(f"w{i}" for i in range(1000))   # 1000 words
    chunks = FixedChunker(size=200, overlap=20).chunk([doc])
    assert len(chunks) > 1                          # long doc -> many chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text.split()) <= 200 for c in chunks)
    assert [c.chunk_id for c in chunks[:3]] == ["0-0", "0-1", "0-2"]


def test_fixed_chunker_overlap_is_honored():
    doc = " ".join(f"w{i}" for i in range(250))
    chunks = FixedChunker(size=100, overlap=30).chunk([doc])
    assert chunks[0].text.split()[-30:] == chunks[1].text.split()[:30]


def test_short_doc_yields_one_chunk():
    chunks = FixedChunker(size=512, overlap=50).chunk(["just three words"])
    assert len(chunks) == 1
    assert chunks[0].text == "just three words"


def test_noop_chunker_returns_whole_doc():
    docs = ["doc one text", "doc two text"]
    chunks = NoOpChunker().chunk(docs)
    assert len(chunks) == 2                          # one chunk per document
    assert chunks[0].text == "doc one text"
    assert [c.chunk_id for c in chunks] == ["0-0", "1-0"]


def test_empty_documents_skipped():
    assert FixedChunker().chunk(["", "   "]) == []
    assert NoOpChunker().chunk(["", "   "]) == []


def test_doc_ids_custom_and_validated():
    chunks = NoOpChunker().chunk(["a", "b"], doc_ids=["alpha", "beta"])
    assert {c.doc_id for c in chunks} == {"alpha", "beta"}
    try:
        NoOpChunker().chunk(["a", "b"], doc_ids=["only-one"])
        assert False, "expected ValueError on mismatched doc_ids"
    except ValueError:
        pass


def test_invalid_params_rejected():
    for bad in [dict(size=0), dict(size=100, overlap=100), dict(size=100, overlap=-1)]:
        try:
            FixedChunker(**bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def test_registry_swap_by_string():
    # The whole point: pick an implementation by name, no class reference.
    fixed = build("chunker", "fixed", {"size": 100, "overlap": 10})
    none = build("chunker", "none")
    assert isinstance(fixed, FixedChunker) and isinstance(none, NoOpChunker)
    assert isinstance(fixed, Chunker) and isinstance(none, Chunker)


def test_registry_unknown_type_errors():
    try:
        build("chunker", "does-not-exist")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_registry_rejects_duplicate_name():
    try:
        @register("chunker", "fixed")               # 'fixed' already taken by FixedChunker
        class Clash:
            pass
        assert False, "expected ValueError on duplicate registration"
    except ValueError:
        pass


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} chunking checks passed.")


if __name__ == "__main__":
    _run()
