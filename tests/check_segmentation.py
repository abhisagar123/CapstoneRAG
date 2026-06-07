"""Checks for segmentation (src/segmentation/) — the pipeline→evaluator bridge.

  * OFFLINE — regex splitter, the OutputSegmenter keying scheme, registry. No deps.
  * NLTK   — NltkSplitter (needs punkt data). Gated behind NLTK=1:
        NLTK=1 python tests/check_segmentation.py

Run directly:  python tests/check_segmentation.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401
from src.registry import build, available
from src.segmentation import SentenceSplitter, OutputSegmenter, RegexSplitter, load_nltk_splitter
from src.segmentation.base import _letters

WANT_NLTK = os.environ.get("NLTK") == "1"


# ---------------- OFFLINE ----------------

def test_regex_registered_nltk_gated():
    assert "regex" in available("splitter")
    assert "nltk" not in available("splitter")     # gated until load_nltk_splitter()


def test_letters_scheme():
    assert _letters(3) == ["a", "b", "c"]
    assert _letters(27)[25:] == ["z", "aa"]        # rolls over past 26 correctly
    assert _letters(0) == []


def test_regex_splits_basic():
    s = RegexSplitter()
    assert isinstance(s, SentenceSplitter)
    out = s.split("First sentence here. Second one follows! A third?")
    assert out == ["First sentence here.", "Second one follows!", "A third?"]
    assert s.split("") == [] and s.split("   ") == []


def test_segmenter_document_keys():
    seg = OutputSegmenter(RegexSplitter())
    docs = ["Alpha here. Beta here.", "Gamma here."]
    out = seg.segment_documents(docs)
    keys = [s[0] for doc in out for s in doc]
    assert keys == ["0a", "0b", "1a"]              # {doc_idx}{letter}


def test_segmenter_response_keys():
    seg = OutputSegmenter(RegexSplitter())
    out = seg.segment_response("One thing. Two thing. Three thing.")
    assert [s[0] for s in out] == ["a", "b", "c"]  # plain letters


def test_segment_matches_ragbench_shape():
    seg = OutputSegmenter(RegexSplitter())
    out = seg.segment(["Doc one. Doc two."], "Ans one. Ans two.")
    # documents_sentences: [doc][sent][key, text]; response_sentences: [sent][key, text]
    assert out["documents_sentences"] == [[["0a", "Doc one."], ["0b", "Doc two."]]]
    assert out["response_sentences"] == [["a", "Ans one."], ["b", "Ans two."]]


def test_build_via_registry():
    seg = OutputSegmenter(build("splitter", "regex"))
    assert seg.segment_response("Hi there.") == [["a", "Hi there."]]


# ---------------- NLTK (needs punkt data) ----------------

def test_nltk_handles_abbreviations_better():
    load_nltk_splitter()
    assert "nltk" in available("splitter")
    nltk_s = build("splitter", "nltk")
    out = nltk_s.split("Section 8.2 applies. See Dr. Smith. The fee is $1.5M.")
    assert len(out) == 3                            # not fooled by "Dr." / "8.2" / "$1.5M"
    assert "See Dr. Smith." in out


def _run():
    offline = [test_regex_registered_nltk_gated, test_letters_scheme,
               test_regex_splits_basic, test_segmenter_document_keys,
               test_segmenter_response_keys, test_segment_matches_ragbench_shape,
               test_build_via_registry]
    nltk_tests = [test_nltk_handles_abbreviations_better]

    for fn in offline:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"{len(offline)} offline checks passed.")

    if WANT_NLTK:
        print("\nrunning NLTK checks (NLTK=1)...")
        for fn in nltk_tests:
            fn(); print(f"  ✅ {fn.__name__}")
        print(f"{len(nltk_tests)} NLTK checks passed.")
    else:
        print("\n(skipped NLTK checks — set NLTK=1 to run them)")


if __name__ == "__main__":
    _run()
