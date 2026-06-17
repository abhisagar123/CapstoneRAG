"""Judge — produce TRACe labels for our pipeline's output (the evaluator's hard half).

The math half (evaluator/trace.py) turns sentence labels into the 4 TRACe scores.
THIS half is the LLM that PRODUCES those labels: given a question + keyed documents
+ keyed answer, it decides which document sentences are relevant (R) and which the
answer utilized (U), and whether the answer is supported.

The judge defines GROUND TRUTH, so two things are non-negotiable:
  1. Use the EXACT RAGBench Appendix-7.4 labeling prompt (verbatim below) — the same
     prompt the dataset's GPT-4 reference labels came from, so we replicate that task.
  2. Validate the judge against the reference scores before trusting it (§9.4).

This file holds the shared pieces: the Judge interface, the prompt builder, robust
JSON extraction, and the adapter that maps judge JSON -> the math half's inputs
(so trace.py stays the single source of scoring truth). Concrete judges
(HuggingFaceJudge / later OpenAIJudge) live in sibling files.
"""

import json
import re
from typing import Protocol, runtime_checkable

from ..evaluator.trace import (
    total_doc_sentences, relevance, utilization, completeness,
)

# ── The EXACT RAGBench Appendix-7.4 GPT labeling prompt (paper pp.16-17) ──────────
# {documents} = doc sentences as keyed lines (0a, 0b, ...); {question}; {answer} =
# response sentences as keyed lines (a, b, ...). Reproduced verbatim so our judge
# performs the same labeling task as the dataset's GPT-4 annotator.
APPENDIX_7_4_PROMPT = '''I asked someone to answer a question based on one or more documents.
Your task is to review their response and assess whether or not each sentence
in that response is supported by text in the documents. And if so, which
sentences in the documents provide that support. You will also tell me which
of the documents contain useful information for answering the question, and
which of the documents the answer was sourced from.

Here are the documents, each of which is split into sentences. Alongside each
sentence is associated key, such as '0a.' or '0b.' that you can use to refer
to it:

```
{documents}
```

The question was:
```
{question}
```

Here is their response, split into sentences. Alongside each sentence is
associated key, such as 'a.' or 'b.' that you can use to refer to it. Note
that these keys are unique to the response, and are not related to the keys
in the documents:

```
{answer}
```

You must respond with a JSON object matching this schema:

```
{{
  "relevance_explanation": string,
  "all_relevant_sentence_keys": [string],
  "overall_supported_explanation": string,
  "overall_supported": boolean,
  "sentence_support_information": [
    {{
      "response_sentence_key": string,
      "explanation": string,
      "supporting_sentence_keys": [string],
      "fully_supported": boolean
    }},
  ],
  "all_utilized_sentence_keys": [string]
}}
```
The relevance_explanation field is a string explaining which documents
contain useful information for answering the question. Provide a step-by-step
breakdown of information provided in the documents and how it is useful for
answering the question.

The all_relevant_sentence_keys field is a list of all document sentences keys
(e.g. '0a') that are relevant to the question. Include every sentence that is
useful and relevant to the question, even if it was not used in the response,
or if only parts of the sentence are useful. Ignore the provided response when
making this judgement and base your judgement solely on the provided documents
and question. Omit sentences that, if removed from the document, would not
impact someone's ability to answer the question.

The overall_supported_explanation field is a string explaining why the response
*as a whole* is or is not supported by the documents. In this field, provide a
step-by-step breakdown of the claims made in the response and the support (or
lack thereof) for those claims in the documents. Begin by assessing each claim
separately, one by one; don't make any remarks about the response as a whole
until you have assessed all the claims in isolation.

The overall_supported field is a boolean indicating whether the response as a
whole is supported by the documents. This value should reflect the conclusion
you drew at the end of your step-by-step breakdown in overall_supported_explanation.

In the sentence_support_information field, provide information about the support
*for each sentence* in the response.

The sentence_support_information field is a list of objects, one for each sentence
in the response. Each object MUST have the following fields:
- response_sentence_key: a string identifying the sentence in the response.
This key is the same as the one used in the response above.
- explanation: a string explaining why the sentence is or is not supported by the
documents.
- supporting_sentence_keys: keys (e.g. '0a') of sentences from the documents that
support the response sentence. If the sentence is not supported, this list MUST
be empty. If the sentence is supported, this list MUST contain one or more keys.
In special cases where the sentence is supported, but not by any specific sentence,
you can use the string "supported_without_sentence" to indicate that the sentence
is generally supported by the documents. Consider cases where the sentence is
expressing inability to answer the question due to lack of relevant information in
the provided context as "supported_without_sentence". In cases where the sentence
is making a general statement (e.g. outlining the steps to produce an answer, or
summarizing previously stated sentences, or a transition sentence), use the
string "general". In cases where the sentence is correctly stating a well-known fact,
like a mathematical formula, use the string "well_known_fact". In cases where the
sentence is performing numerical reasoning (e.g. addition, multiplication), use
the string "numerical_reasoning".
- fully_supported: a boolean indicating whether the sentence is fully supported by
the documents.
  - This value should reflect the conclusion you drew at the end of your step-by-step
  breakdown in explanation.
  - If supporting_sentence_keys is an empty list, then fully_supported must be false.
'''


@runtime_checkable
class Judge(Protocol):
    """The contract every judge honours: keyed example -> RAGBench label JSON (dict)."""

    def label(self, question: str, keyed: dict) -> dict:
        """Given a question and a keyed example (documents_sentences +
        response_sentences, from the segmenter), return the parsed label JSON."""
        ...


# ── conservative steer (optional addendum — targets a MEASURED bias) ──────────────
# Validation (Qwen2.5-7B, N=50, 4 domains) showed the judge SYSTEMATICALLY over-marks:
# relevance signed-error was +0.06..+0.26 (always positive → too many sentences called
# relevant) and adherence over-flags swamped under-flags (e.g. covidqa 24 vs 3 → too
# eager to call answers unsupported). Both are the same behaviour: a LOOSER threshold
# than the GPT-4 reference. This addendum makes the prompt's existing "omit if removable"
# instruction louder, to pull that threshold toward the reference. Kept SEPARATE and
# appended (never edits the verbatim Appendix-7.4 prompt) so "baseline vs conservative"
# is a clean, reportable A/B. Used only when conservative=True.
CONSERVATIVE_ADDENDUM = '''

IMPORTANT — be strict and conservative in your judgements:
- For all_relevant_sentence_keys: include a sentence ONLY if it directly provides
  information needed to answer the question. EXCLUDE sentences that merely restate or
  rephrase the question, give titles/headings/citations/author names, or are about the
  general topic without contributing a fact used to answer it. If removing a sentence
  would not reduce someone's ability to answer the question, OMIT it.
- For overall_supported and fully_supported: judge an answer sentence as supported
  unless it CLEARLY contradicts or has no basis in the documents. Do not flag a sentence
  as unsupported merely because the wording differs from the documents or the support is
  paraphrased — only flag a genuine, clear lack of support.
When in doubt, prefer FEWER relevant keys and prefer marking answers as supported.'''


# ── prompt assembly ──────────────────────────────────────────────────────────────

def _render_doc_lines(documents_sentences) -> str:
    """Flatten [[ [key,text], ...], ...] into 'key. text' lines for the prompt."""
    lines = []
    for doc in documents_sentences:
        for key, text in doc:
            lines.append(f"{key}. {text}")
    return "\n".join(lines)


def _render_response_lines(response_sentences) -> str:
    return "\n".join(f"{key}. {text}" for key, text in response_sentences)


def build_prompt(question: str, keyed: dict, *, conservative: bool = False) -> str:
    """Fill the Appendix-7.4 template with this example's keyed docs + answer.

    conservative=True appends CONSERVATIVE_ADDENDUM (the verbatim Appendix-7.4 text is
    unchanged) to counter the measured over-marking bias — an A/B arm, off by default.
    """
    prompt = APPENDIX_7_4_PROMPT.format(
        documents=_render_doc_lines(keyed["documents_sentences"]),
        question=question,
        answer=_render_response_lines(keyed["response_sentences"]),
    )
    return prompt + CONSERVATIVE_ADDENDUM if conservative else prompt


# ── robust JSON extraction (OSS models often wrap JSON in prose/markdown) ─────────

def _extract_candidate(text: str) -> str:
    """Pull the JSON object substring out of a model's raw output (fence or outermost {...})."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in judge output")
    return text[start:end + 1]


def _escape_inner_quotes(s: str) -> str:
    """Salvage the #1 OSS-judge JSON failure: an UNescaped double-quote inside a
    string value, e.g.  "explanation": "the term "net 30" applies"  -> the inner
    quotes break json.loads ('Expecting , delimiter').

    Walk the text char by char tracking whether we're inside a string. A `"` is
    STRUCTURAL (keep as-is) only if, ignoring whitespace, the next non-space char
    is one of  : , } ]  (closing a value/key) OR it's opening a string right after
    one of  : , { [  . Any other `"` while inside a string is a literal the model
    forgot to escape -> we escape it to \\". Best-effort; if it still won't parse,
    the caller falls back to raising.
    """
    out = []
    in_str = False
    n = len(s)
    for i, ch in enumerate(s):
        if ch != '"':
            out.append(ch)
            continue
        if not in_str:
            # Opening a string (we only flip to in_str when the prev meaningful char
            # was a structural opener — but for robustness just treat it as opening).
            in_str = True
            out.append(ch)
            continue
        # We're inside a string and hit a `"`. Decide: structural close or literal?
        j = i + 1
        while j < n and s[j] in " \t\r\n":
            j += 1
        nxt = s[j] if j < n else ""
        if nxt in ":,}]" or nxt == "":
            in_str = False          # legitimate end of this string value
            out.append(ch)
        else:
            out.append('\\"')       # unescaped inner quote -> escape it
    return "".join(out)


def parse_label_json(text: str) -> dict:
    """Extract + parse the JSON object from a model's raw output.

    Three passes, increasingly forgiving (OSS judges are messy):
      1. strict parse of the extracted {...}            (fast path, clean models)
      2. salvage unescaped inner quotes, parse again    (the common 7B failure)
    Raises ValueError if all passes fail (the caller skips/counts that example)."""
    candidate = _extract_candidate(text)            # may raise: no object at all
    try:
        return json.loads(candidate)                # pass 1: strict
    except json.JSONDecodeError as first_err:
        try:
            return json.loads(_escape_inner_quotes(candidate))   # pass 2: salvage
        except json.JSONDecodeError:
            raise ValueError(f"judge output was not valid JSON: {first_err}")


# ── adapter: judge JSON -> the 4 TRACe scores (reuses the validated math) ─────────

def scores_from_label(keyed: dict, label: dict) -> dict:
    """Map judge label JSON to TRACe scores via the validated trace.py math.

    Also returns the micro-values behind each score for analysis.
    """
    total = total_doc_sentences(keyed["documents_sentences"])

    R = label.get("all_relevant_sentence_keys", [])
    U = label.get("all_utilized_sentence_keys", [])

    overlap = set(R) & set(U)

    sentence_info = label.get("sentence_support_information", [])
    unsupported_count = sum(
        1 for s in sentence_info
        if not s.get("fully_supported", False)
    )

    return {
        # TRACe scores
        "relevance": relevance(R, total),
        "utilization": utilization(U, total),
        "completeness": completeness(R, U),
        "adherence": bool(label.get("overall_supported", False)),

        # Micro-metrics
        "relevant_count": len(R),
        "total_sentences": total,
        "utilized_count": len(U),
        "overlap_count": len(overlap),
        "unsupported_count": unsupported_count,
    }
