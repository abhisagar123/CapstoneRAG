# Pipeline Walkthrough — one question, end to end

**What this is:** a guided trace of a *single real question* through every stage of the
CapstoneRAG system — which function runs, what it receives, what it returns — from the raw
documents all the way to the four TRACe scores. Read it top to bottom to understand how the
pieces we built brick-by-brick actually fit together at run time.

The example below is a **real run** (config `grounded_norerank`, domain GenKnowledge,
generator `llama3.2:3b`, judge `llama3.1:8b`).

> **The question:** *"how to make a word in a cell as hyperlink"* (an Excel how-to)
> **Source:** 9 documents shipped with this RAGBench example.

---

## The 30-second map

```
  OFFLINE (build the index, once per example's docs)
    documents ──▶ Chunker ──▶ Embedder ──▶ Index
                  (split)     (→vectors)   (FAISS)

  ONLINE (answer the question)
    question ──▶ Retriever ──▶ Reranker ──▶ Repacker ──▶ PromptBuilder ──▶ Generator
                 (find k)      (re-score)   (reorder)    (wrap)            (LLM answer)
                                                                              │
  EVALUATE (score the answer)                                                 ▼
    {answer, sources} ──▶ OutputSegmenter ──▶ Judge ──▶ TRACe math ──▶ 4 scores
                          (key sentences)     (label)   (arithmetic)
```

Two analogies to hold onto:
- **Offline = stocking a library.** We cut the documents into index cards (chunks), give each
  a "meaning fingerprint" (embedding), and file them so we can find similar ones fast (index).
- **Online = a librarian answering a question.** Find candidate cards (retrieve), pick the best
  (rerank), arrange them (repack), write the question + cards onto a worksheet (prompt), and
  hand it to a writer who answers using only those cards (generate).
- **Evaluate = a grader.** Split the answer and sources into numbered sentences (segment), have
  an examiner mark which sentences were relevant/used/supported (judge), then compute the four
  fraction-based scores (TRACe math).

---

## How a run is assembled: config → registry → Pipeline

Before any stage runs, a **config** (a `configs/*.yaml` file) names which implementation to use
for each stage. `build_pipeline(config)` (in `src/pipeline.py`) reads it and, for each stage,
asks the **registry** for the class registered under that `type` string and constructs it.

```yaml
chunker:   { type: fixed,  size: 512, overlap: 50 }
embedder:  { type: minilm }
index:     { type: faiss, corpus_mode: per_example }
retriever: { type: dense, k: 20 }
reranker:  { type: none, top_n: 5 }     # grounded_norerank has NO reranker
repacker:  { type: reverse }
prompt:    { type: grounded }
generator: { type: ollama, model: llama3.2:3b }
splitter:  { type: regex }              # used by the segmenter, at eval time
```

So "an experiment = a config." Swapping `reranker: none` → `cross_encoder` is the *only* change
between two of our four strategies — that's what makes the results matrix interpretable.

---

## OFFLINE — building the index

### Stage 1 — Chunker  (`src/chunking/fixed_chunker.py`)

`Pipeline.index_documents(documents)` first calls `chunker.chunk(documents) -> list[Chunk]`.

- **In:** the 9 raw document strings.
- **Out:** a list of `Chunk` objects (`text`, `doc_id`, `chunk_id`, `meta`).
- **Here:** each of the 9 docs was short (< 512 words), so the `FixedChunker(size=512, overlap=50)`
  produced **9 chunks** (one per doc — nothing needed splitting). On a long-document domain like
  CUAD, one doc would explode into many chunks; that's where the chunking *strategy* matters.

### Stage 2 — Embedder  (`src/embeddings/sentence_transformer_embedder.py`)

`embedder.embed([chunk.text for chunk in chunks]) -> (n, dim) array`.

- **In:** the 9 chunk texts.
- **Out:** a `(9, 384)` float array — each chunk becomes a 384-number vector (`all-MiniLM-L6-v2`).
- **Plain meaning:** the vector is a "meaning fingerprint." Two chunks about hyperlinks land close
  together in this 384-dim space; a chunk about taxes lands far away. Vectors are **normalized**,
  so "closeness" = dot product = cosine similarity.

### Stage 3 — Index  (`src/indexing/faiss_index.py`)

`index.add(chunks, vectors)` stores them. `corpus_mode: per_example` means the index holds **only
this question's 9 chunks** (matching how RAGBench's reference data was built). Between examples the
runner calls `pipe.reset_index()` to start fresh.

> After Stage 3, `index_documents` returns `9` (chunks indexed). The library is stocked.

---

## ONLINE — answering the question

This is all inside `Pipeline.answer(query) -> {answer, sources, context}`.

### Stage 4 — Retriever  (`src/retrieval/dense_retriever.py`)

`retriever.retrieve(query, k) -> list[RetrievedChunk]`.

- **How:** it embeds the *question* with the **same** embedder (so question and chunks live in the
  same space), then `index.search(query_vector, k)` returns the `k` nearest chunks by cosine score.
- **In:** `"how to make a word in a cell as hyperlink"`, `k=20` (but only 9 chunks exist, so all 9).
- **Out:** chunks ranked by score. **Top chunk scored 0.624** — *"Here we will show you how to
  create a hyperlink to another document. With your Excel document open click on the cell..."* —
  exactly on topic. Each `RetrievedChunk` carries `(chunk, score, rank)`.

### Stage 5 — Reranker  (`src/reranking/`)

`reranker.rerank(query, chunks, top_n) -> list[RetrievedChunk]`.

- **`grounded_norerank` uses `type: none`** → `NoOpReranker`, which just **truncates to the top
  `top_n=5`** (no re-scoring). So we keep the retriever's top 5.
- **The contrast arm `cross_encoder`** would instead re-score each (question, chunk) *pair together*
  with a cross-encoder model and re-order — usually surfacing better chunks. **In the results matrix,
  turning this on raised adherence ~0.2** (see `EXPERIMENTS.md`): better chunks → the model has the
  right material → less hallucination.

### Stage 6 — Repacker  (`src/repacking/reverse_repacker.py`)

`repacker.pack(chunks) -> reordered chunks`.

- **`reverse`** puts the **most-relevant chunk LAST**. Why: LLMs suffer "lost in the middle" — they
  attend best to the start and **end** of the prompt. Putting the best chunk last means it's freshest
  when the model starts writing. Pure reordering; no chunk added or dropped.

### Stage 7 — PromptBuilder  (`src/prompting/grounded_prompt_builder.py`)

`prompt_builder.build(query, chunks) -> str` — assembles the final text sent to the LLM.

- **`grounded`** wraps the chunks + question with an instruction like *"answer using ONLY the
  context; if it's not there, say you can't answer."* This is the **main adherence / anti-
  hallucination lever** and also drives the refusal behavior RGB cares about.
- **The contrast arm `minimal`** omits that instruction — bare context + question. Comparing the two
  measures how much the grounding instruction actually helps.

### Stage 8 — Generator  (`src/generation/ollama_generator.py`)

`generator.generate(prompt) -> str` — POSTs the prompt to the local Ollama server (`llama3.2:3b`)
and returns the answer text.

- **Out (real):** *"To make a word in a cell as a hyperlink, you need to use the HYPERLINK function.
  For example, if the cell contents are 'Please click here...' and you want only the word 'here'
  to be..."*

`answer()` returns `{answer, sources (the 5 chunks), context (their joined text)}` and **stops
there** — scoring is a separate job (clean separation; the judge is pluggable).

---

## EVALUATE — scoring the answer

### Stage 9 — OutputSegmenter  (`src/segmentation/base.py`)

`segmenter.segment(doc_texts, answer) -> {documents_sentences, response_sentences}`.

This is the **bridge** to the evaluator. It splits our context + answer into **keyed sentences**,
byte-for-byte matching RAGBench's reference schema, so the same judge + math work on our output.

- **In:** the 5 retrieved chunk texts + the generated answer.
- **Out (real):**
  - `documents_sentences[0][0] = ['0a', 'Here we will show you how to create a hyperlink...']`
  - `response_sentences[0]     = ['a',  'To make a word in a cell as a hyperlink, you need...']`
  - Keys: `{doc}{letter}` for context (`0a, 0b, 1a...`), plain letters for the answer (`a, b...`).

### Stage 10 — Judge  (`src/judge/ollama_judge.py`)

`judge.label(question, keyed) -> dict` — the LLM examiner (`llama3.1:8b`), using the **exact
RAGBench Appendix-7.4 prompt**. It reads the keyed sentences and returns JSON marking:
- `all_relevant_sentence_keys` (**R**) — which doc sentences are relevant to the question
- `all_utilized_sentence_keys` (**U**) — which the answer actually used
- `overall_supported` (bool) — is the whole answer grounded?

(OSS models emit messy JSON; the judge has a salvage parser + a sampling retry, and a failure on
one example is skipped-and-counted, never fatal.)

### Stage 11 — TRACe math  (`src/evaluator/trace.py`, via `scores_from_label`)

Pure arithmetic on R, U, and the total sentence count `T`. **This is validated to reproduce
RAGBench's reference scores to RMSE = 0** — so any error here is the *judge's* labeling, not the math.

```
  relevance    = len(R) / T            # fraction of context that's relevant   → retriever
  utilization  = len(U) / T            # fraction of context the answer used   → generator/prompt
  completeness = len(R ∩ U) / len(R)   # of relevant info, how much was used   (1.0 if R empty)
  adherence    = overall_supported     # all answer sentences grounded? (bool) → faithfulness
```

**Worked mini-example** (illustrative): if the judge marks 3 of 18 context sentences relevant,
`relevance = 3/18 = 0.167`. That single number is one example's score; a matrix cell is the **mean**
over N examples.

---

## From one question to the results matrix

`run_experiment(cfg, examples, segmenter, judge)` (in `src/runner.py`) runs Stages 1–11 for **each**
of the N examples and **averages** the four scores → one matrix row. `run_named_matrix` loops that
over every (config, domain) → the full strategy × domain matrix (`results/per_example/ragbench_matrix.csv`).

```
  ONE matrix cell =
    for each of N questions:  index → answer → segment → judge → 4 scores
    then: mean of the N × 4 scores
```

So when you read `grounded_rerank / GenKnowledge / adh=0.80`, it means: *"running the grounded-prompt
+ cross-encoder strategy on 10 GenKnowledge questions, 8/10 answers were judged fully grounded."*

See **`EXPERIMENTS.md`** for what the matrix numbers tell us, per lever, with reasoning.

---

## Where each stage lives (quick index)

| Stage | File | Swappable types |
|---|---|---|
| Chunker | `src/chunking/` | `fixed`, `none` |
| Embedder | `src/embeddings/` | `minilm` / `sentence_transformer` |
| Index | `src/indexing/` | `faiss` (per_example / pooled) |
| Retriever | `src/retrieval/` | `dense` |
| Reranker | `src/reranking/` | `cross_encoder`, `none` |
| Repacker | `src/repacking/` | `forward`, `reverse`, `sides` |
| PromptBuilder | `src/prompting/` | `grounded`, `minimal` |
| Generator | `src/generation/` | `ollama`, `hf`, `echo` |
| Segmenter | `src/segmentation/` | splitter: `regex`, `nltk` |
| Judge | `src/judge/` | `ollama`, `hf`, `fake` |
| TRACe math | `src/evaluator/trace.py` | (not swappable — pure math) |
| Orchestration | `src/pipeline.py`, `src/runner.py` | — |
