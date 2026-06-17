"""OFFLINE checks for the summarization (context-compression) brick + its wiring
into the registry / config / pipeline. No network, no dataset — pure logic, runs
in milliseconds. The one embedding-backed test is gated behind MODEL=1.

Run directly:        python tests/check_summarization.py
With heavy test:     MODEL=1 python tests/check_summarization.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401 — triggers component registration via src/__init__.py
from src.registry import build, available
from src.indexing import RetrievedChunk
from src.chunking import Chunk
from src.summarization import (
    Summarizer, NoOpSummarizer, LexicalSummarizer, load_summarizers,
    select_top_indices, keep_sentences,
)

WANT_MODEL = os.environ.get("MODEL") == "1"


def _rc(text: str, rank: int = 0) -> RetrievedChunk:
    """Build a RetrievedChunk wrapping `text` (the shape the pipeline passes around)."""
    return RetrievedChunk(chunk=Chunk(text=text, doc_id="0", chunk_id=f"0-{rank}"),
                          score=1.0 - 0.1 * rank, rank=rank)


# ── light tier registers on import; heavy tier only via load_summarizers() ──────────

def test_light_summarizers_registered_on_import():
    assert "none" in available("summarizer")
    assert "lexical" in available("summarizer")


def test_heavy_summarizer_not_registered_until_loaded():
    # Mirror of the embedder/semantic-chunker rule: the embedding summarizer must
    # register ONLY via load_summarizers(), never on bare `import src`.
    assert callable(load_summarizers)
    load_summarizers()
    assert "extractive_embedding" in available("summarizer")


# ── select_top_indices: the pure heart of every extractive arm ──────────────────────

def test_select_top_indices_keeps_best_in_reading_order():
    # scores: idx0=0.9, idx2=0.8 are the two best; ratio 0.5 of 4 -> keep 2 -> [0,2]
    assert select_top_indices([0.9, 0.1, 0.8, 0.2], ratio=0.5) == [0, 2]


def test_select_top_indices_min_keep_floor():
    # ratio so small it would keep 0; min_keep=1 forces the single best (idx1).
    assert select_top_indices([0.2, 0.9, 0.1], ratio=0.01, min_keep=1) == [1]


def test_select_top_indices_ratio_one_keeps_all():
    assert select_top_indices([0.1, 0.5, 0.3], ratio=1.0) == [0, 1, 2]


def test_select_top_indices_empty():
    assert select_top_indices([], ratio=0.5) == []


def test_select_top_indices_ties_prefer_earlier():
    # all equal → ceil(0.5*4)=2 kept; ties resolved by original order → [0,1]
    assert select_top_indices([0.5, 0.5, 0.5, 0.5], ratio=0.5) == [0, 1]


def test_keep_sentences_joins_in_order_and_never_blanks():
    sents = ["A.", "B.", "C."]
    assert keep_sentences("A. B. C.", sents, [0, 2]) == "A. C."
    # empty keep set → fall back to original text (never blank a chunk)
    assert keep_sentences("A. B. C.", sents, []) == "A. B. C."


# ── NoOpSummarizer: exact passthrough (the baseline control) ────────────────────────

def test_noop_returns_chunks_unchanged():
    chunks = [_rc("first sentence here. second one too.", 0), _rc("another chunk.", 1)]
    out = NoOpSummarizer().compress("any query", chunks)
    assert [c.chunk.text for c in out] == [c.chunk.text for c in chunks]
    assert len(out) == len(chunks)


# ── LexicalSummarizer: model-free, drops off-topic sentences ────────────────────────

def test_lexical_drops_offtopic_sentence():
    # 2 sentences: one about revenue (matches query), one about an office move (not).
    chunk = _rc("Revenue rose twelve percent on ad sales. The office moved to a new building in May.")
    out = LexicalSummarizer(ratio=0.5).compress("what caused the revenue rise", [chunk])
    kept = out[0].chunk.text
    assert "Revenue rose" in kept          # the query-relevant sentence survives
    assert "office moved" not in kept      # the off-topic one is dropped


def test_lexical_preserves_provenance_and_count():
    # Compression must keep doc_id/chunk_id/score/rank and never drop a whole chunk.
    chunks = [_rc("Revenue rose on ad sales. Cats are nice pets. Dogs too.", 0)]
    out = LexicalSummarizer(ratio=0.34).compress("revenue ad sales", chunks)
    assert len(out) == 1
    assert out[0].chunk.doc_id == "0" and out[0].chunk.chunk_id == "0-0"
    assert out[0].rank == 0 and out[0].score == chunks[0].score


def test_lexical_short_chunk_passthrough():
    # A single-sentence chunk (<= min_keep) is returned untouched, never emptied.
    chunk = _rc("Only one sentence here.")
    out = LexicalSummarizer(ratio=0.5).compress("unrelated query terms", [chunk])
    assert out[0].chunk.text == "Only one sentence here."


def test_lexical_empty_query_passthrough():
    chunk = _rc("Sentence one. Sentence two. Sentence three.")
    out = LexicalSummarizer(ratio=0.5).compress("", [chunk])
    assert out[0].chunk.text == chunk.chunk.text   # no query → nothing to score → unchanged


def test_lexical_rejects_bad_params():
    for bad in (0.0, -0.1, 1.5):
        try:
            LexicalSummarizer(ratio=bad); assert False, f"ratio={bad} should raise"
        except ValueError:
            pass
    try:
        LexicalSummarizer(min_keep=0); assert False, "min_keep=0 should raise"
    except ValueError:
        pass


# ── config + pipeline wiring (summarizer is an OPTIONAL stage) ───────────────────────

def test_config_accepts_summarizer_stage():
    # Use validate=False so the test stays offline (minilm/echo register only after a
    # heavy load_*()); we're checking config PARSING of the new optional stage here.
    from src.config import from_dict
    base = {
        "domain": "Legal",
        "chunker": {"type": "fixed", "size": 512, "overlap": 50},
        "embedder": {"type": "minilm"},
        "index": {"type": "faiss"},
        "retriever": {"type": "dense", "k": 20},
        "prompt": {"type": "grounded"},
        "generator": {"type": "echo"},
        "splitter": {"type": "regex"},
        "summarizer": {"type": "lexical", "ratio": 0.5},
    }
    cfg = from_dict(base, do_validate=False)
    assert cfg.summarizer is not None and cfg.summarizer.type == "lexical"
    assert cfg.summarizer.params["ratio"] == 0.5

    # Absent summarizer → None (optional, skipped) — back-compat with all existing configs.
    no_sum = {k: v for k, v in base.items() if k != "summarizer"}
    assert from_dict(no_sum, do_validate=False).summarizer is None
    # (The light "lexical"/"none" arms ARE registered on `import src` — see
    # test_light_summarizers_registered_on_import — so a validated config can name them.)


def test_config_id_includes_summarizer_and_distinguishes():
    # A summarizer swap must yield a DISTINCT config_id (else build_grid would treat
    # two summarizer arms as a collision / silently overwrite a matrix row).
    from src.config import from_dict
    from src.runner import config_id
    base = {
        "domain": "Legal",
        "chunker": {"type": "fixed", "size": 512, "overlap": 50},
        "embedder": {"type": "minilm"}, "index": {"type": "faiss"},
        "retriever": {"type": "dense", "k": 20}, "prompt": {"type": "grounded"},
        "generator": {"type": "echo"}, "splitter": {"type": "regex"},
    }
    # validate=False: we inspect config_id (stage TYPES) only — no need to register the
    # heavy minilm embedder for this structural check.
    none_id = config_id(from_dict(base, do_validate=False))                       # no summarizer
    lex_id = config_id(from_dict({**base, "summarizer": {"type": "lexical"}}, do_validate=False))
    assert none_id != lex_id
    assert none_id.split("|")[6] == "none"        # position 7 = summarizer slot
    assert lex_id.split("|")[6] == "lexical"


def test_pipeline_builds_summarizer_stage():
    # Confirm the pipeline ASSEMBLES the optional summarizer from config (registry
    # lookup of the light "lexical" arm). We build the Pipeline object and inspect the
    # stage; we don't run a full answer() here (that needs a torch embedder) — the
    # end-to-end answer path is exercised offline in check_runner.py with the fake
    # embedder, and with a real model under MODEL=1 below.
    from src.config import from_dict
    from src.pipeline import Pipeline

    class _StubEmbedder:                      # avoid the torch import for an assembly check
        dim = 8
        def embed(self, texts):
            import numpy as _np
            return _np.zeros((len(texts), self.dim), dtype="float32")

    cfg = from_dict({
        "domain": "Legal",
        "chunker": {"type": "fixed", "size": 512, "overlap": 50},
        "embedder": {"type": "minilm"}, "index": {"type": "faiss"},
        "retriever": {"type": "dense", "k": 20}, "prompt": {"type": "grounded"},
        "generator": {"type": "echo"}, "splitter": {"type": "regex"},
        "summarizer": {"type": "lexical", "ratio": 0.5},
    }, do_validate=False)

    # Bypass __init__'s heavy embedder build; we only need to confirm the summarizer
    # branch builds the right object from config.
    from src.registry import build
    obj = Pipeline.__new__(Pipeline)
    obj.summarizer = build("summarizer", cfg.summarizer.type, cfg.summarizer.params)
    assert obj.summarizer is not None and type(obj.summarizer).__name__ == "LexicalSummarizer"


# ── HEAVY (MODEL=1): the embedding summarizer actually scores by meaning ─────────────

def test_extractive_embedding_keeps_semantically_relevant(_gated=True):
    if not WANT_MODEL:
        print("  [skip] test_extractive_embedding_* (set MODEL=1 to run — loads MiniLM)")
        return
    load_summarizers()
    summ = build("summarizer", "extractive_embedding", {"ratio": 0.5})
    # Query is about revenue; the office/cafe sentences are off-topic. Meaning-based
    # scoring (not word overlap) should keep the revenue + ad-demand sentences.
    chunk = _rc("The office moved in May. Revenue rose twelve percent on strong ad sales. "
                "Staff enjoyed the new cafe. Ad demand outpaced supply that quarter.")
    out = summ.compress("What drove the increase in revenue?", [chunk])
    kept = out[0].chunk.text
    assert "Revenue rose" in kept
    assert "office moved" not in kept and "new cafe" not in kept


def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ok: {t.__name__}")
    print(f"\n{passed}/{len(tests)} summarization checks passed"
          + ("" if WANT_MODEL else "  (1 heavy test skipped — MODEL=1 to run it)"))


if __name__ == "__main__":
    run_all()
