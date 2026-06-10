"""OFFLINE checks for the judge (src/judge/). Fake judge + parsing + adapter — no model.

The real HuggingFaceJudge is validated on Colab (it needs the model + the
judge_validate harness against reference scores).

Run directly:  python tests/check_judge.py
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src  # noqa: F401
from src.registry import build, available
from src.judge import (
    Judge, FakeJudge, load_judges, build_prompt, parse_label_json,
    scores_from_label, APPENDIX_7_4_PROMPT,
)
from src.segmentation import OutputSegmenter, RegexSplitter

SEG = OutputSegmenter(RegexSplitter())
CTX = ["Either party may terminate with thirty days notice. Notice must be written.",
       "Governed by Delaware law."]
ANS = "The notice period is thirty days. It must be written."
KEYED = SEG.segment(CTX, ANS)


def test_fake_registered_hf_gated():
    assert "fake" in available("judge")
    assert "hf" not in available("judge")
    assert "torch" not in sys.modules


def test_prompt_uses_appendix_template_and_fills_slots():
    p = build_prompt("What is the notice period?", KEYED)
    # verbatim Appendix-7.4 opening + our keyed content slotted in
    assert p.startswith("I asked someone to answer a question based on one or more documents.")
    assert "0a. Either party may terminate" in p          # doc sentence keyed
    assert "a. The notice period is thirty days." in p    # response sentence keyed
    assert "What is the notice period?" in p


def test_fake_judge_returns_ragbench_schema():
    label = FakeJudge().label("q?", KEYED)
    assert isinstance(FakeJudge(), Judge)
    for field in ("all_relevant_sentence_keys", "all_utilized_sentence_keys",
                  "overall_supported", "sentence_support_information"):
        assert field in label
    # one support entry per response sentence
    assert len(label["sentence_support_information"]) == len(KEYED["response_sentences"])


def test_scores_from_label_uses_trace_math():
    # relevant_frac=1.0 -> ALL doc sentences relevant -> relevance == 1.0.
    label = FakeJudge(relevant_frac=1.0, supported=True).label("q?", KEYED)
    s = scores_from_label(KEYED, label)
    assert s["relevance"] == 1.0                            # |R| == total -> 1.0
    assert 0.0 < s["utilization"] <= 1.0                    # some utilized (FakeJudge keeps >=1)
    assert s["adherence"] is True                          # straight from overall_supported
    # adherence follows overall_supported, regardless of keys.
    label2 = FakeJudge(supported=False).label("q?", KEYED)
    assert scores_from_label(KEYED, label2)["adherence"] is False


def test_parse_json_plain():
    assert parse_label_json('{"all_relevant_sentence_keys": ["0a"]}') == {"all_relevant_sentence_keys": ["0a"]}


def test_parse_json_from_fence_and_prose():
    messy = 'Here you go:\n```json\n{"overall_supported": true}\n```\ndone'
    assert parse_label_json(messy) == {"overall_supported": True}


def test_parse_json_raises_on_garbage():
    try:
        parse_label_json("no json here at all")
        assert False
    except ValueError:
        pass


def test_conservative_addendum_is_optional_and_appended():
    # Baseline must be the VERBATIM Appendix-7.4 prompt (no addendum) — clean A/B control.
    base = build_prompt("q?", KEYED)
    cons = build_prompt("q?", KEYED, conservative=True)
    from src.judge import CONSERVATIVE_ADDENDUM
    assert CONSERVATIVE_ADDENDUM not in base                 # baseline untouched
    assert base.startswith("I asked someone to answer")      # still the real prompt
    assert cons.startswith(base) and cons.endswith(CONSERVATIVE_ADDENDUM)  # extends, not alters
    assert "strict and conservative" in cons


def test_parse_json_salvages_unescaped_inner_quotes():
    # The real Colab failure: a 7B model writes an unescaped " inside a value.
    messy = '{"relevance_explanation": "the term "net 30" applies", "overall_supported": true}'
    out = parse_label_json(messy)                      # strict parse fails -> salvage pass
    assert out["overall_supported"] is True
    assert "net 30" in out["relevance_explanation"]    # content preserved, quotes escaped


def test_parse_json_salvage_keeps_lists_and_clean_json_intact():
    # Salvage must NOT corrupt clean JSON or list values.
    clean = '{"all_relevant_sentence_keys": ["0a","1b"], "overall_supported": false}'
    assert parse_label_json(clean) == {"all_relevant_sentence_keys": ["0a", "1b"],
                                       "overall_supported": False}


def test_score_one_skips_a_judge_that_raises():
    # A judge raising ValueError on bad JSON must be SKIPPED (return None), not crash —
    # so one bad answer can't kill a whole config's matrix row.
    from src.runner import _score_one
    from src.segmentation import OutputSegmenter, RegexSplitter

    class _BadJudge:
        def label(self, question, keyed):
            raise ValueError("judge produced unparseable JSON after retries")

    seg = OutputSegmenter(RegexSplitter())
    ex = {"question": "q?", "_context_texts": ["A sentence. Another."]}
    assert _score_one(ex, "Some answer.", seg, _BadJudge()) is None    # skipped, no raise


def test_build_via_registry():
    assert isinstance(build("judge", "fake"), FakeJudge)


def test_load_judges_registers_hf():
    load_judges()
    assert "hf" in available("judge")


def test_load_judges_registers_ollama():
    load_judges()
    assert "ollama" in available("judge")              # local-server judge


def test_sweep_csv_path_encodes_model_variant_n():
    # The filename scheme that keeps parallel runs from colliding + lets the verdict glob+merge.
    from src.evaluator.judge_validate import sweep_csv_path
    p = sweep_csv_path("meta-llama/Llama-3.1-8B-Instruct", "conservative", 50)
    assert p == "results/judge_validation__meta-llama-llama-3-1-8b-instruct__conservative__n50.csv"
    # different model / variant / n -> different file (no clobbering)
    assert sweep_csv_path("mistral", "baseline", 50) != p


def test_flatten_report_matches_sweep_cols():
    # The flattened row must have EXACTLY the CSV columns (schema guard — both nb03 and
    # the local script depend on this).
    from src.evaluator.judge_validate import _flatten_report, SWEEP_COLS
    fake_report = {"config": "covidqa", "n": 5, "n_scored": 5, "n_failed": 0,
                   "relevance": {"rmse": 0.1, "mean_abs_err": 0.08, "signed_err": 0.05},
                   "utilization": {"rmse": 0.1, "mean_abs_err": 0.08, "signed_err": 0.05},
                   "completeness": {"rmse": 0.1, "mean_abs_err": 0.08, "signed_err": 0.0},
                   "adherence": {"accuracy": 0.8, "over_flag": 1, "under_flag": 0}}
    row = _flatten_report(fake_report, "llama3.1:8b", "baseline", "Biomedical")
    assert set(row.keys()) == set(SWEEP_COLS)           # exact schema match
    assert row["relevance_signed"] == 0.05 and row["adherence_over_flag"] == 1


def test_ollama_judge_posts_and_parses():
    # OFFLINE: mock httpx.post so we exercise the judge with NO server. Confirms it
    # sends the Appendix-7.4 prompt (with conservative steer) and parses the JSON.
    import httpx
    from src.judge import CONSERVATIVE_ADDENDUM
    from src.judge.ollama_judge import OllamaJudge

    captured = {}
    label_json = ('{"all_relevant_sentence_keys": ["0a"], '
                  '"all_utilized_sentence_keys": ["0a"], "overall_supported": true}')

    class _FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"response": label_json}

    def _fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        return _FakeResp()

    orig = httpx.post
    httpx.post = _fake_post
    try:
        j = OllamaJudge(model="llama3.1:8b", conservative=True)
        out = j.label("What is the notice period?", KEYED)
    finally:
        httpx.post = orig

    assert out["overall_supported"] is True            # parsed the label JSON
    assert out["all_relevant_sentence_keys"] == ["0a"]
    # conservative steer + verbatim Appendix-7.4 both went into the prompt
    assert CONSERVATIVE_ADDENDUM in captured["payload"]["prompt"]
    assert captured["payload"]["prompt"].startswith("I asked someone to answer")
    assert captured["payload"]["format"] == "json"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} judge checks passed.")


if __name__ == "__main__":
    _run()
