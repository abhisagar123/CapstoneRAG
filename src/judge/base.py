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


def build_prompt(question: str, keyed: dict) -> str:
    """Fill the Appendix-7.4 template with this example's keyed docs + answer."""
    return APPENDIX_7_4_PROMPT.format(
        documents=_render_doc_lines(keyed["documents_sentences"]),
        question=question,
        answer=_render_response_lines(keyed["response_sentences"]),
    )


# ── robust JSON extraction (OSS models often wrap JSON in prose/markdown) ─────────

def parse_label_json(text: str) -> dict:
    """Extract the JSON object from a model's raw output. Tolerates ```json fences
    and surrounding prose by grabbing the outermost {...}. Raises ValueError if none."""
    # Strip a ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        # Fallback: outermost balanced-looking {...}
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in judge output")
        candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"judge output was not valid JSON: {e}")


# ── adapter: judge JSON -> the 4 TRACe scores (reuses the validated math) ─────────

def scores_from_label(keyed: dict, label: dict) -> dict:
    """Map judge label JSON to TRACe scores via the validated trace.py math.

    Adherence uses the judge's `overall_supported` boolean directly — that field
    IS what RAGBench stored as `adherence_score` (§9.6), the judge's holistic
    conclusion. Relevance/utilization/completeness use the R/U key lists with the
    same list-vs-set semantics as trace.py.
    """
    total = total_doc_sentences(keyed["documents_sentences"])
    R = label.get("all_relevant_sentence_keys", [])
    U = label.get("all_utilized_sentence_keys", [])
    return {
        "relevance": relevance(R, total),
        "utilization": utilization(U, total),
        "completeness": completeness(R, U),
        "adherence": bool(label.get("overall_supported", False)),
    }
