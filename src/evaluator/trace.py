"""TRACe metrics — the arithmetic ("math") half of our evaluator.

This module turns sentence-level labels (which document sentences are *relevant*
to the question, and which the answer actually *used*) into the four TRACe scores.

It contains NO LLM / API calls. Given the labels, the four scores are pure
arithmetic. We proved during EDA (see AI_CONTEXT.md §9.6) that these exact
formulas reproduce RAGBench's shipped "reference" scores, so this half can be
unit-tested to ~zero error against the dataset before we build the LLM judge.

Terminology used throughout:
    R  = the set/list of "relevant" document-sentence keys   (e.g. '1a', '3c')
    U  = the set/list of "utilized" document-sentence keys
    T  = total number of sentences across all documents (the denominator)
"""


def total_doc_sentences(documents_sentences) -> int:
    """Count T: how many sentences are there across all the documents?

    `documents_sentences` is the RAGBench field with shape:
        [ document ][ sentence ][ key, text ]
    i.e. a list of documents, where each document is a list of sentences,
    and each sentence is a [key, text] pair like ['1a', 'Micro-CT analysis ...'].

    To get the total we just add up the number of sentences in each document
    (like counting students classroom-by-classroom, then summing).
    """
    return sum(len(document) for document in documents_sentences)


def relevance(relevant_keys, total_sentences) -> float:
    """Context Relevance = relevant sentences / total sentences.

    "What fraction of the retrieved documents was actually useful for the
    question?" Diagnoses the *retriever*. (e.g. 11 / 18 = 0.611)

    NOTE (EDA gotcha, AI_CONTEXT.md §9.6): we count `relevant_keys` with the raw
    LIST length (duplicates kept) because that is how RAGBench computed its
    reference scores. Do NOT dedupe here or we drift from the official numbers.
    """
    if total_sentences == 0:
        return 0.0
    return len(relevant_keys) / total_sentences


def utilization(utilized_keys, total_sentences) -> float:
    """Context Utilization = utilized sentences / total sentences.

    "What fraction of the retrieved documents did the answer actually use?"
    Diagnoses the *generator/prompt*. Same raw-list-length rule as relevance().
    """
    if total_sentences == 0:
        return 0.0
    return len(utilized_keys) / total_sentences


def completeness(relevant_keys, utilized_keys) -> float:
    """Completeness = (relevant AND used) / relevant.

    "Of the useful material, how much did the answer actually cover (vs miss)?"

    - Denominator: number of relevant sentences (raw LIST length, duplicates kept
      — same §9.6 rule).
    - Numerator: the DISTINCT sentences that are both relevant and used, so we
      treat the overlap as a set intersection.
    - Edge case: if nothing is relevant (empty R) completeness is defined as 1.0
      (you can't miss what isn't there). RAGBench ships 1.0 in exactly these rows.
    """
    if len(relevant_keys) == 0:
        return 1.0
    overlap = set(relevant_keys) & set(utilized_keys)
    return len(overlap) / len(relevant_keys)


def adherence(unsupported_response_sentence_keys) -> bool:
    """Adherence = is the WHOLE answer grounded in the documents? (yes/no)

    This is the hallucination check. The answer is "adherent" only if NONE of
    its sentences is unsupported (like one made-up sentence failing an essay's
    fact-check — one bad sentence spoils the whole thing).

    EDA lesson (AI_CONTEXT.md §9.6 — see "adherence bug" note): we initially
    tried `all(s['fully_supported'])` from `sentence_support_information`, but
    that flag is `None` (left blank) in the *majority* of rows (e.g. finqa: 1306
    None vs 11 True), so it is NOT the reliable signal — that definition matched
    the shipped `adherence_score` only ~32% of the time. The dataset's true
    source of truth is the explicit `unsupported_response_sentence_keys` list:
    if it is empty, nothing was unsupported → adherent. This reproduces the
    shipped `adherence_score` on 100% of tested rows across 5 domains.
    """
    return len(unsupported_response_sentence_keys) == 0


def score_from_reference_labels(example) -> dict:
    """Compute all four TRACe scores for a RAGBench example from its GOLD labels.

    This is the "reference reproduction" path: feed in RAGBench's own ground-truth
    sentence labels (R, U, and the unsupported list) and we should get back the
    dataset's shipped reference scores. It's how we validate the math half
    (notebook 02_evaluator_validation).

    It also documents the CONTRACT for later: the LLM judge (the hard half we
    build next) will produce these same label fields for OUR pipeline's answers,
    and we'll call the identical four functions below. Same calculator, different
    source of labels.

    Expects a dict with the RAGBench fields:
        documents_sentences, all_relevant_sentence_keys,
        all_utilized_sentence_keys, unsupported_response_sentence_keys
    """
    total = total_doc_sentences(example["documents_sentences"])
    relevant = example["all_relevant_sentence_keys"]
    utilized = example["all_utilized_sentence_keys"]
    unsupported = example["unsupported_response_sentence_keys"]
    return {
        "relevance": relevance(relevant, total),
        "utilization": utilization(utilized, total),
        "completeness": completeness(relevant, utilized),
        "adherence": adherence(unsupported),
    }
