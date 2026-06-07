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


def test_build_via_registry():
    assert isinstance(build("judge", "fake"), FakeJudge)


def test_load_judges_registers_hf():
    load_judges()
    assert "hf" in available("judge")


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} judge checks passed.")


if __name__ == "__main__":
    _run()
