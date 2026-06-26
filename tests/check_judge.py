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


class _Timeout(Exception):
    """Stand-in for httpx.ReadTimeout — a NON-(ValueError/KeyError) judge failure."""


def test_score_one_skips_a_judge_that_raises():
    # A judge that raises must be SKIPPED (return None), not crash — so one bad answer
    # can't kill a whole config's matrix row. Covers BOTH bad-JSON (ValueError) AND a
    # network timeout (the CUAD failure that actually crashed a run).
    from src.runner import _score_one
    from src.segmentation import OutputSegmenter, RegexSplitter
    seg = OutputSegmenter(RegexSplitter())
    ex = {"question": "q?", "_context_texts": ["A sentence. Another."]}

    class _BadJSON:
        def label(self, q, k): raise ValueError("unparseable JSON after retries")

    class _Slow:
        def label(self, q, k): raise _Timeout("timed out")

    assert _score_one(ex, "Some answer.", seg, _BadJSON()) is None     # JSON failure -> skip
    assert _score_one(ex, "Some answer.", seg, _Slow()) is None        # timeout -> skip (was a crash)


def test_judge_one_example_skips_timeout():
    # The sweep's parallel unit must also treat a timeout as skip-and-count, not fatal.
    from src.evaluator.judge_validate import _judge_one_example
    ex = {"question": "q?",
          "documents_sentences": [[["0a", "A."]]], "response_sentences": [["a", "A."]],
          "relevance_score": 0.0, "utilization_score": 0.0,
          "completeness_score": 1.0, "adherence_score": True}

    class _Slow:
        def label(self, q, k): raise _Timeout("timed out")

    scores, reference = _judge_one_example(_Slow(), ex, "cuad")
    assert scores is None                              # skipped, not raised
    assert reference["adherence"] is True              # reference still returned for alignment


def test_build_via_registry():
    assert isinstance(build("judge", "fake"), FakeJudge)


def test_load_judges_registers_hf():
    load_judges()
    assert "hf" in available("judge")


def test_load_judges_registers_ollama():
    load_judges()
    assert "ollama" in available("judge")              # local-server judge


def test_judge_one_example_pairs_scores_with_own_reference():
    # The parallel unit of work: must return THIS example's scores paired with THIS
    # example's reference — never mixing examples (the across/within-domain safety unit).
    from src.evaluator.judge_validate import _judge_one_example
    ex = {"question": "q?",
          "documents_sentences": [[["0a", "Cats are mammals."], ["0b", "Sky is blue."]]],
          "response_sentences": [["a", "Cats are mammals."]],
          "relevance_score": 0.5, "utilization_score": 0.5,
          "completeness_score": 1.0, "adherence_score": True}
    scores, reference = _judge_one_example(FakeJudge(relevant_frac=1.0), ex, "covidqa")
    assert reference == {"relevance": 0.5, "utilization": 0.5,
                         "completeness": 1.0, "adherence": True}   # this example's own ref
    assert scores is not None and "relevance" in scores

    class _Bad:
        def label(self, q, k): raise ValueError("bad json")
    scores2, reference2 = _judge_one_example(_Bad(), ex, "covidqa")
    assert scores2 is None                                 # failure -> None (skip+count path)
    assert reference2["adherence"] is True                 # reference still returned


def test_validate_judge_accepts_workers_param():
    # Signature guard: run_validation_sweep + validate_judge both take workers.
    import inspect
    from src.evaluator.judge_validate import validate_judge, run_validation_sweep
    assert "workers" in inspect.signature(validate_judge).parameters
    assert "workers" in inspect.signature(run_validation_sweep).parameters


def test_sweep_csv_path_encodes_model_variant_n():
    # The filename scheme that keeps parallel runs from colliding + lets the verdict glob+merge.
    from src.evaluator.judge_validate import sweep_csv_path
    p = sweep_csv_path("meta-llama/Llama-3.1-8B-Instruct", "conservative", 50)
    assert p == "results/validation/judge_validation__meta-llama-llama-3-1-8b-instruct__conservative__n50.csv"
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


def test_load_judges_registers_groq():
    load_judges()
    assert "groq" in available("judge")                # hosted-API validation judge


def test_groq_judge_posts_chat_schema_and_parses():
    # OFFLINE: mock httpx.post. Groq speaks the OpenAI CHAT schema, not Ollama's — verify
    # messages[] in, choices[0].message.content out, json_object mode, conservative steer.
    import httpx
    from src.judge import CONSERVATIVE_ADDENDUM
    from src.judge.groq_judge import GroqJudge, GROQ_URL

    captured = {}
    label_json = ('{"all_relevant_sentence_keys": ["0a"], '
                  '"all_utilized_sentence_keys": ["0a"], "overall_supported": true}')

    class _FakeResp:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": label_json}, "finish_reason": "stop"}]}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return _FakeResp()

    orig = httpx.post
    httpx.post = _fake_post
    try:
        j = GroqJudge(model="llama-3.3-70b-versatile", conservative=True, api_key="test-key")
        out = j.label("What is the notice period?", KEYED)
    finally:
        httpx.post = orig

    assert out["overall_supported"] is True            # parsed choices[0].message.content
    assert out["all_relevant_sentence_keys"] == ["0a"]
    assert captured["url"] == GROQ_URL
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "llama-3.3-70b-versatile"
    msg = captured["payload"]["messages"][0]
    assert msg["role"] == "user"
    # verbatim Appendix-7.4 prompt + conservative steer both went into the chat message
    assert msg["content"].startswith("I asked someone to answer")
    assert CONSERVATIVE_ADDENDUM in msg["content"]
    assert captured["payload"]["response_format"] == {"type": "json_object"}   # OpenAI json mode
    assert captured["payload"]["temperature"] == 0.0                           # greedy first pass
    assert "prompt" not in captured["payload"]                                 # NOT Ollama's schema
    assert "num_ctx" not in captured["payload"]                                # Groq sizes ctx server-side


def test_groq_judge_requires_key_lazily():
    # No key at construction is fine (offline config validation builds the object); the
    # clear error fires only when label() actually needs the API.
    from src.judge.groq_judge import GroqJudge
    j = GroqJudge(api_key=None)
    try:
        j.label("q?", KEYED)
        assert False, "expected RuntimeError when GROQ_API_KEY is missing"
    except RuntimeError as e:
        assert "GROQ_API_KEY" in str(e)


def test_groq_judge_retries_on_malformed_json_with_sampling():
    # The judge's distinctive path: first completion is unparseable -> retry, and the
    # retry must RAISE the temperature (sample) so the regeneration can differ.
    import httpx
    from src.judge.groq_judge import GroqJudge

    temps, calls = [], {"n": 0}
    good = ('{"all_relevant_sentence_keys": [], "all_utilized_sentence_keys": [], '
            '"overall_supported": false}')

    class _Resp:
        status_code = 200
        headers = {}
        def __init__(self, body): self._body = body
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": self._body}}]}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        temps.append(json["temperature"])
        return _Resp("not json at all") if calls["n"] == 1 else _Resp(good)

    orig = httpx.post
    httpx.post = _fake_post
    try:
        j = GroqJudge(api_key="test-key", max_retries=1)
        out = j.label("q?", KEYED)
    finally:
        httpx.post = orig

    assert out["overall_supported"] is False
    assert calls["n"] == 2                              # first failed, retried once
    assert temps[0] == 0.0 and temps[1] > 0.0           # greedy first, sampled retry


def test_groq_judge_throttles_when_rpm_set():
    # requests_per_minute > 0 must SPACE sends (proactive pacing under Groq's tokens/min
    # budget). We stub time so the test is instant but still asserts the sleep was requested.
    import time
    import src.judge.groq_judge as gj
    from src.judge.groq_judge import GroqJudge

    good = '{"all_relevant_sentence_keys": [], "all_utilized_sentence_keys": [], "overall_supported": true}'
    slept = []

    class _Resp:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": good}}]}

    import httpx
    orig_post, orig_sleep = httpx.post, time.sleep
    orig_last = gj._LAST_CALL[0]
    # Simulate "a request just happened" so the next send must wait ~the full interval.
    gj._LAST_CALL[0] = time.monotonic()
    httpx.post = lambda *a, **k: _Resp()
    time.sleep = lambda s: slept.append(s)
    try:
        j = GroqJudge(api_key="test-key", requests_per_minute=5.0)   # -> 12s min interval
        assert abs(j._min_interval - 12.0) < 1e-9
        j.label("q?", KEYED)                                         # must pace behind the recent call
    finally:
        httpx.post, time.sleep = orig_post, orig_sleep
        gj._LAST_CALL[0] = orig_last

    assert slept and slept[0] > 0                                    # it waited before sending
    assert slept[0] <= 12.0                                          # but no more than the interval


def test_groq_judge_no_throttle_by_default():
    # rpm unset (0) -> no pacing (offline tests / paid tier): _min_interval is 0.
    from src.judge.groq_judge import GroqJudge
    assert GroqJudge(api_key="k")._min_interval == 0.0


def test_groq_judge_fails_fast_on_daily_cap():
    # A 429 with a LONG Retry-After (the per-DAY token cap resets in ~20+ min) must RAISE
    # immediately, not park the run in a 20-minute time.sleep(). Regression test for the
    # "stuck sweep" bug: honouring Retry-After verbatim hung the whole validation run.
    import time
    import httpx
    from src.judge.groq_judge import GroqJudge

    slept = []

    class _Resp429Daily:
        status_code = 429
        headers = {"Retry-After": "1289"}              # ~21 min — a daily-cap wait
        text = '{"error":{"message":"...tokens per day (TPD): Limit 100000..."}}'
        def raise_for_status(self): raise AssertionError("should raise before this")
        def json(self): return {}

    orig_post, orig_sleep = httpx.post, time.sleep
    httpx.post = lambda *a, **k: _Resp429Daily()
    time.sleep = lambda s: slept.append(s)             # would record a long sleep if it happened
    try:
        j = GroqJudge(api_key="test-key", max_rate_limit_retries=5, max_backoff=90.0)
        try:
            j.label("q?", KEYED)
            assert False, "expected RuntimeError on a daily-cap-length Retry-After"
        except RuntimeError as e:
            assert "daily token cap" in str(e) or "TPD" in str(e)
    finally:
        httpx.post, time.sleep = orig_post, orig_sleep

    assert not slept, "must NOT sleep for a daily-cap-length wait (fail fast instead)"


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
    # num_ctx MUST be sent + large — else Ollama silently truncates long (CUAD) prompts.
    assert captured["payload"]["options"]["num_ctx"] >= 8192


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} judge checks passed.")


if __name__ == "__main__":
    _run()
