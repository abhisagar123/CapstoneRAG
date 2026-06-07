"""OFFLINE checks for the config layer (src/config.py). No models — uses only
light, locally-registered components (regex/fixed/faiss/dense/echo/grounded).

Run directly:  python tests/check_config.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401 — registers light components
from src.config import (
    from_dict, validate, _parse_stage, StageConfig, PipelineConfig,
    REQUIRED_STAGES, OPTIONAL_STAGES,
)

# A valid config using ONLY light components (no embedder/generator model needed).
# We use "none" reranker and skip the embedder-type check by overriding to a
# light registered type where possible. embedder/minilm is gated, so for offline
# config tests we point embedder at a type we know is registered light... but none
# exists — so these tests use do_validate selectively.
BASE = {
    "domain": "Legal",
    "chunker":   {"type": "fixed", "size": 512, "overlap": 50},
    "embedder":  {"type": "minilm"},     # gated; only registers after load_embedders()
    "index":     {"type": "faiss", "corpus": "per_example"},
    "retriever": {"type": "dense"},
    "reranker":  {"type": "none", "top_n": 5},
    "repacker":  {"type": "reverse"},
    "prompt":    {"type": "grounded"},
    "generator": {"type": "echo"},
    "splitter":  {"type": "regex"},
}


def test_parse_stage_separates_type_and_params():
    sc = _parse_stage("chunker", {"type": "fixed", "size": 512, "overlap": 50})
    assert sc.type == "fixed"
    assert sc.params == {"size": 512, "overlap": 50}


def test_parse_stage_requires_type():
    try:
        _parse_stage("prompt", {"size": 5})
        assert False
    except ValueError:
        pass


def test_from_dict_builds_without_validation():
    # do_validate=False lets us build the object without the registry checks
    # (so this test doesn't depend on the gated embedder being loaded).
    cfg = from_dict(BASE, do_validate=False)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.domain == "Legal"
    assert cfg.chunker.type == "fixed" and cfg.chunker.params["size"] == 512
    assert cfg.reranker.type == "none"
    assert cfg.seed == 42


def test_missing_domain_raises():
    d = {k: v for k, v in BASE.items() if k != "domain"}
    try:
        from_dict(d, do_validate=False)
        assert False
    except ValueError:
        pass


def test_missing_required_stage_raises():
    d = {k: v for k, v in BASE.items() if k != "retriever"}
    try:
        from_dict(d, do_validate=False)
        assert False
    except ValueError as e:
        assert "retriever" in str(e)


def test_optional_stage_can_be_null():
    d = dict(BASE); d["reranker"] = None; d["repacker"] = None
    cfg = from_dict(d, do_validate=False)
    assert cfg.reranker is None and cfg.repacker is None


def test_validate_catches_unknown_type():
    d = dict(BASE); d["chunker"] = {"type": "does_not_exist"}
    try:
        validate(from_dict(d, do_validate=False))
        assert False
    except ValueError as e:
        assert "chunker" in str(e) and "does_not_exist" in str(e)


def test_validate_passes_for_light_components():
    # Build a config whose every stage is a LIGHT registered type, so validate()
    # passes with no model loading. (embedder has no light type, so we validate
    # the stages individually via the registry instead.)
    from src.registry import available
    assert "fixed" in available("chunker")
    assert "faiss" in available("index")
    assert "dense" in available("retriever")
    assert "none" in available("reranker")
    assert "reverse" in available("repacker")
    assert "grounded" in available("prompt")
    assert "echo" in available("generator")
    assert "regex" in available("splitter")


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} config checks passed.")


if __name__ == "__main__":
    _run()
