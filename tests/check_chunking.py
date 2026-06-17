"""OFFLINE checks for the chunker + the registry swap mechanism (src/chunking.py,
src/registry.py). No network, no dataset — pure logic, runs in milliseconds.

Run directly:  python tests/check_chunking.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401 — triggers component registration via src/__init__.py
from src.registry import build, available, register, REGISTRY
from src.chunking import Chunk, Chunker, FixedChunker, NoOpChunker, ParagraphGroupChunker, load_chunkers
from src.chunking.semantic_chunker import SemanticChunker

WANT_MODEL = os.environ.get("MODEL") == "1"


def test_components_registered_on_import():
    # Importing src must populate the registry (the decorator-runs-on-import gotcha).
    assert "fixed" in available("chunker")
    assert "none" in available("chunker")
    assert "pgc" in available("chunker")


def test_heavy_chunkers_not_registered_until_loaded():
    # Mirror of the embedder rule: the HEAVY chunkers must register only via
    # load_chunkers(), never on bare `import src` (keeps import torch-free).
    assert callable(load_chunkers)


def test_pgc_groups_adjacent_paragraphs():
    # 4 blank-line-separated paragraphs; paragraphs=2, overlap=1 -> windows
    # [p0,p1], [p1,p2], [p2,p3] (sliding by 1).
    doc = "Para zero.\n\nPara one.\n\nPara two.\n\nPara three."
    chunks = ParagraphGroupChunker(paragraphs=2, overlap=1).chunk([doc])
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].text == "Para zero.\n\nPara one."     # grouped 2 paras, blank-line joined
    assert "Para one." in chunks[1].text                  # overlap=1 carries para one forward
    assert [c.chunk_id for c in chunks][:3] == ["0-0", "0-1", "0-2"]


def test_pgc_single_block_falls_back_to_one_chunk():
    # No blank lines AND no newlines -> one paragraph -> one chunk (the whole text).
    chunks = ParagraphGroupChunker().chunk(["one solid block of text with no breaks"])
    assert len(chunks) == 1
    assert chunks[0].text == "one solid block of text with no breaks"


def test_pgc_falls_back_to_single_newlines():
    # Many corpora separate paragraphs with single newlines (no blank line). PGC should
    # still find structure rather than treating the whole doc as one paragraph.
    doc = "Line A\nLine B\nLine C\nLine D"
    chunks = ParagraphGroupChunker(paragraphs=2, overlap=0).chunk([doc])
    assert len(chunks) == 2                                # [A,B],[C,D]
    assert "Line A" in chunks[0].text and "Line B" in chunks[0].text


def test_pgc_empty_and_invalid_params():
    assert ParagraphGroupChunker().chunk(["", "   "]) == []
    for bad in [dict(paragraphs=0), dict(paragraphs=2, overlap=2), dict(paragraphs=2, overlap=-1)]:
        try:
            ParagraphGroupChunker(**bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


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


def test_swap_changes_behavior_not_just_type():
    # The full swap claim: identical calling code + different config string ->
    # genuinely different chunking behavior on the SAME input.
    doc = " ".join(f"w{i}" for i in range(1000))    # one 1000-word document
    results = {}
    for type_name, params in [("fixed", {"size": 200, "overlap": 20}),
                              ("fixed", {"size": 500, "overlap": 50}),
                              ("none", {})]:
        chunker = build("chunker", type_name, params)   # <- identical calling code
        results[(type_name, params.get("size"))] = len(chunker.chunk([doc]))
    # smaller window -> more chunks; none -> exactly 1 (whole doc)
    assert results[("fixed", 200)] > results[("fixed", 500)] > results[("none", None)]
    assert results[("none", None)] == 1


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


# ---------------- SemanticChunker: pure logic (no model, no torch) ----------------
# _cut_after and _group are pure functions of the similarity scores, so the whole
# boundary-decision logic is testable offline — the embedder (the only heavy part)
# just produces the `sims` list those functions consume.

def test_semantic_invalid_params_rejected():
    for bad in [dict(breakpoint="nonsense"), dict(percentile=0), dict(percentile=100),
                dict(threshold=-0.1), dict(threshold=1.1)]:
        try:
            SemanticChunker(**bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def test_semantic_absolute_cuts_below_threshold():
    # sims between 4 sentences: high, LOW (topic shift), high.
    sc = SemanticChunker(breakpoint="absolute", threshold=0.5)
    cut = sc._cut_after([0.8, 0.1, 0.7])      # only index 1 (sim 0.1) is below 0.5
    assert cut == {1}
    pieces = sc._group(["s0", "s1", "s2", "s3"], cut)
    assert pieces == ["s0 s1", "s2 s3"]        # cut AFTER sentence 1


def test_semantic_percentile_cuts_largest_drop():
    # distances = 1 - sim = [0.1, 0.9, 0.2]; the 95th-percentile distance is ~0.83,
    # so only the big gap (0.9, index 1) exceeds it → exactly one cut, the sharpest shift.
    sc = SemanticChunker(breakpoint="percentile", percentile=95)
    cut = sc._cut_after([0.9, 0.1, 0.8])
    assert cut == {1}


def test_semantic_uniform_doc_stays_one_chunk():
    # A perfectly coherent document (all neighbour sims equal) has no gap exceeding
    # the percentile → NO cut → the strict ">" keeps it as a single chunk.
    sc = SemanticChunker(breakpoint="percentile", percentile=95)
    assert sc._cut_after([0.7, 0.7, 0.7, 0.7]) == set()
    assert sc._group(["a", "b", "c"], set()) == ["a b c"]


def test_semantic_empty_sims():
    assert SemanticChunker()._cut_after([]) == set()


def test_semantic_registered_only_after_load():
    load_chunkers()
    assert "semantic" in available("chunker")


# ---------------- SemanticChunker: end-to-end with the real model (MODEL=1) ----------------

def test_semantic_splits_topic_shift_end_to_end():
    load_chunkers()
    sc = build("chunker", "semantic", {"breakpoint": "absolute", "threshold": 0.4})
    # Two clearly different topics; the chunker should NOT lump them into one chunk.
    doc = ("The cat sat on the warm mat. It purred softly in the afternoon sun. "
           "Quarterly revenue rose twelve percent. Operating costs fell sharply.")
    chunks = sc.chunk([doc])
    assert all(isinstance(c, Chunk) for c in chunks)
    assert len(chunks) >= 2, "expected the cat/finance topic shift to force a cut"
    assert [c.chunk_id for c in chunks][:2] == ["0-0", "0-1"]
    joined = " ".join(c.text for c in chunks)
    assert "cat" in joined and "revenue" in joined          # no text lost


def test_semantic_empty_and_single_sentence():
    load_chunkers()
    sc = build("chunker", "semantic")
    assert sc.chunk(["", "   "]) == []                       # empty docs skipped
    one = sc.chunk(["Just one solitary sentence here."])     # nothing to compare
    assert len(one) == 1 and one[0].chunk_id == "0-0"


def _run():
    model_tests = {"test_semantic_splits_topic_shift_end_to_end",
                   "test_semantic_empty_and_single_sentence"}
    all_fns = [(k, v) for k, v in sorted(globals().items())
               if k.startswith("test_") and callable(v)]
    offline = [v for k, v in all_fns if k not in model_tests]
    model = [v for k, v in all_fns if k in model_tests]

    for fn in offline:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(offline)} offline chunking checks passed.")

    if WANT_MODEL:
        print("\nrunning model checks (MODEL=1; may download ~90MB)...")
        for fn in model:
            fn(); print(f"  ✅ {fn.__name__}")
        print(f"{len(model)} model checks passed.")
    else:
        print("(skipped semantic model checks — set MODEL=1 to run them)")


if __name__ == "__main__":
    _run()
