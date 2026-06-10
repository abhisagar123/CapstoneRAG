# Low-Level Design (LLD) — CapstoneRAG

**Companion to:** [`HLD.md`](./HLD.md) (system-level *what/why*). This document is the *how*.
**Status:** living document · last updated 2026-06-09.

> ⚠️ **Design intent vs as-built.** Sections marked **AS-BUILT** reflect real, shipped code (verified
> against the source). Unmarked sketches (e.g. the optional QueryTransform/Summarizer, the `expand_grid`
> generator, Drive persistence) are *planned* shape and may change. When a sketch and the code disagree,
> the code wins — file an update here.

---

## 1. Module map

```
src/
├── data_loader.py        # load_domain(); domain → config map; sampling
├── chunking/             # one file per strategy (CONVENTION for multi-impl components)
│   ├── base.py           #   Chunk dataclass + Chunker interface + helpers
│   ├── fixed_chunker.py  #   FixedChunker (baseline)  [type "fixed"]
│   ├── noop_chunker.py   #   NoOpChunker (no-chunking) [type "none"]
│   └── __init__.py       #   re-exports + imports each strategy so it registers
├── embeddings/           # Embedder interface + one file per model impl
│   ├── base.py           #   Embedder interface
│   ├── sentence_transformer_embedder.py  # any sentence-transformers model [types "sentence_transformer"/"minilm"]
│   └── __init__.py       #   load_embedders() lazily registers (keeps torch out of `import src`)
├── indexing/             # ✅ Index backends (package, per convention)
│   ├── base.py           #   Index interface + RetrievedChunk result type
│   ├── faiss_index.py    #   FaissIndex (exact IP); per-example | pooled mode [type "faiss"]
│   └── __init__.py       #   re-exports + registers (faiss imported lazily → light)
├── retrieval/            # ✅ Retriever strategies (package, per convention)
│   ├── base.py           #   Retriever interface
│   ├── dense_retriever.py #  DenseRetriever (text→nearest chunks) [type "dense"]
│   └── __init__.py       #   re-exports + registers; sparse/hybrid = sibling files later
├── reranking/            # ✅ Reranker strategies (package)
│   ├── base.py           #   Reranker interface
│   ├── noop_reranker.py  #   NoOpReranker [type "none"] — light control arm
│   ├── cross_encoder_reranker.py  # CrossEncoderReranker [type "cross_encoder"] — heavy (torch)
│   └── __init__.py       #   registers noop on import; load_rerankers() for the cross-encoder
├── repacking/            # ✅ Repacker strategies (package, one file per strategy)
│   ├── base.py           #   Repacker interface
│   ├── forward_repacker.py #  ForwardRepacker [type "forward"]
│   ├── reverse_repacker.py #  ReverseRepacker [type "reverse"]
│   ├── sides_repacker.py  #  SidesRepacker   [type "sides"]
│   └── __init__.py       #   re-exports + registers (pure Python, light)
├── prompting/            # ✅ PromptBuilder strategies (package)
│   ├── base.py           #   PromptBuilder interface + format_chunks() helper
│   ├── grounded_prompt_builder.py  # GroundedPromptBuilder [type "grounded"] — baseline
│   ├── minimal_prompt_builder.py   # MinimalPromptBuilder  [type "minimal"]  — contrast arm
│   └── __init__.py       #   re-exports + registers (pure Python, light)
├── generation/           # ✅ Generator strategies (package)
│   ├── base.py           #   Generator interface
│   ├── echo_generator.py #   EchoGenerator [type "echo"] — light, no model (tests/wiring)
│   ├── hf_generator.py   #   HuggingFaceGenerator [type "hf"] — real LLM, heavy, COLAB
│   └── __init__.py       #   registers echo on import; load_generators() for hf
├── segmentation/         # ✅ pipeline→evaluator bridge (package)
│   ├── base.py           #   SentenceSplitter interface + OutputSegmenter (RAGBench keyer)
│   ├── regex_splitter.py #   RegexSplitter [type "regex"] — baseline, no deps
│   ├── nltk_splitter.py  #   NltkSplitter  [type "nltk"]  — punkt; via load_nltk_splitter()
│   └── __init__.py       #   registers regex on import; load_nltk_splitter() for nltk
├── query.py              # (optional) QueryTransform: HyDE / decomposition
├── summarizer.py         # (optional) Summarizer: context compression
├── registry.py           # ✅ type-string → class registry + @register decorator + build()
├── config.py             # ✅ StageConfig/PipelineConfig + load_config(yaml) + validate
│                         #    (schema lives here, in one file; grid generator deferred)
├── pipeline.py           # ✅ Pipeline: build_pipeline(cfg) assembles; index_documents() + answer()
├── runner.py             # ✅ ExperimentRunner: run_matrix() loops configs×domains → resumable CSV
├── judge/                # ✅ TRACe judge — produces R/U labels (the evaluator's hard half)
│   ├── base.py           #   Judge interface + EXACT Appendix-7.4 prompt + JSON parse + scores_from_label adapter
│   ├── fake_judge.py     #   FakeJudge [type "fake"] — deterministic, no model (tests/wiring)
│   ├── hf_judge.py       #   HuggingFaceJudge [type "hf"] — real OSS model, heavy, COLAB
│   └── __init__.py       #   registers fake on import; load_judges() for hf
└── evaluator/
    ├── trace.py          # ✅ 4 TRACe metrics from labels (BUILT)
    ├── validate.py       # ✅ validate the MATH half vs reference scores (BUILT)
    ├── judge_validate.py # ✅ validate the JUDGE half vs reference scores (§9.4)
    └── rgb.py            # RGB 4-ability metrics (Phase 3)
```

> **Convention — one file per strategy for multi-implementation components.** Any component expected to
> grow several swappable strategies (chunker, embedder, index, retriever, reranker) is a *package* **from the
> start — even when only one implementation exists yet** (don't defer it to "when the 2nd strategy lands"): a
> `base.py` holding the shared interface + data structures, one file per concrete strategy, and an `__init__.py`
> that re-exports the contract and imports each strategy file so its `@register` decorator runs. Every such
> component MUST define its interface in `base.py` (e.g. `Index`, `Retriever`) — an implementation without a
> declared interface is a convention violation. Benefits:
> teammates add a strategy by dropping in a new file + one import line (no merge conflicts on a shared file),
> and each strategy is self-contained and readable. **File naming:** descriptive snake_case carrying the
> component noun — `fixed_chunker.py`, `noop_chunker.py`, later `pgc_chunker.py` (PEP 8: modules lowercase;
> NOT `FixedChunker.py`). Single-implementation components (e.g. `data_loader.py`, `repacker.py`) stay flat
> files until they need to grow. ⚠️ The `__init__.py` imports are **required** — a
> strategy whose file is never imported never registers (decorators only run on import).

## 2. Core data structures

```python
@dataclass(frozen=True)
class Chunk:
    text: str
    doc_id: str            # which source document
    chunk_id: str          # unique within the index
    meta: dict             # domain, char span, parent id (for small2big), etc.

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int

@dataclass(frozen=True)
class PipelineOutput:
    answer: str
    sources: list[RetrievedChunk]   # what was retrieved (post-rerank/repack)
    context: str                    # exact context string given to the generator

# Bridge to the evaluator — mirrors RAGBench's reference schema (see §6).
@dataclass(frozen=True)
class KeyedExample:
    question: str
    documents_sentences: list[list[tuple[str, str]]]  # [doc][sent] -> ("0a", text)
    response_sentences: list[tuple[str, str]]         # [("a", text), ...]

@dataclass(frozen=True)
class TraceScores:
    relevance: float
    utilization: float
    completeness: float
    adherence: bool
```

## 3. Component interfaces (design intent)

Each is a small class registered under a `type` string (see §7). One baseline implementation first;
variants added for ablations.

```python
class Chunker(Protocol):
    def chunk(self, documents: list[str], *, doc_ids: list[str]) -> list[Chunk]: ...
    # FixedChunker(size=512, overlap=50). Later: PGC, DFC, SentenceGroup, Semantic.

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> "np.ndarray": ...   # (n, d) float32
    @property
    def dim(self) -> int: ...

class Index(Protocol):                             # AS-BUILT (src/indexing/base.py)
    # corpus_mode set at construction: "per_example" or "pooled" (per-domain corpus)
    def add(self, chunks: list[Chunk], vectors: "np.ndarray") -> None: ...
    def search(self, query_vector: "np.ndarray", k: int = 5) -> list[RetrievedChunk]: ...
    def __len__(self) -> int: ...
    # persist()/load() to Drive are PLANNED (not built yet) — re-embedding is cheap at current scale.

class Retriever(Protocol):
    # wraps an Index; "dense" | "sparse"(BM25) | "hybrid"(RRF over both)
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...

class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...
    # NoOpReranker for the "rerank: off" ablation arm.

class Repacker(Protocol):
    def pack(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...
    # forward | reverse (most-relevant last; mitigates lost-in-the-middle) | sides

class PromptBuilder(Protocol):
    def build(self, query: str, chunks: list[RetrievedChunk]) -> str: ...
    # grounding instruction + context formatting. The biggest Adherence lever — kept swappable.

class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...
    # OSS LLM (4-bit, Colab). StubGenerator(echo/templated) lets the belt run locally with no GPU.

class OutputSegmenter:                             # AS-BUILT (src/segmentation/base.py)
    def __init__(self, splitter: SentenceSplitter): ...   # swappable regex/nltk splitter
    def segment(self, doc_texts: list[str], answer: str) -> dict: ...
    # returns {"documents_sentences": [[["0a",text],...],...], "response_sentences": [["a",text],...]}
    # i.e. the RAGBench-keyed schema as plain lists (not the KeyedExample dataclass §2 sketched).
```

## 4. Pipeline assembly & flow

```python
class Pipeline:                                    # AS-BUILT (src/pipeline.py)
    def __init__(self, config: PipelineConfig): ...      # built via build_pipeline(config)
    def index_documents(self, documents: list[str], doc_ids=None) -> int: ...  # offline: chunk→embed→index
    def reset_index(self) -> None: ...                   # fresh empty index (per_example mode: clear between examples)
    def answer(self, query: str) -> dict: ...            # online → {"answer", "sources", "context"}
```

> Note: `index_documents(docs)` takes a document list (the runner feeds each example's own docs for
> `per_example` corpus mode), not a `(domain, split)` pair — domain loading lives in `data_loader.py`.

```mermaid
sequenceDiagram
    participant U as Caller
    participant P as Pipeline
    participant QT as QueryTransform?
    participant R as Retriever
    participant RR as Reranker
    participant RP as Repacker
    participant PB as PromptBuilder
    participant G as Generator
    U->>P: answer(query)
    opt optional
        P->>QT: transform(query)
    end
    P->>R: retrieve(query, k)
    R-->>P: candidates
    P->>RR: rerank(query, candidates, top_n)
    RR-->>P: top_n
    P->>RP: pack(top_n)
    RP-->>P: ordered
    P->>PB: build(query, ordered)
    PB-->>P: prompt
    P->>G: generate(prompt)
    G-->>P: answer
    P-->>U: {answer, sources, context}
```

## 5. TRACe evaluator — BUILT (real code, validated)

```python
# src/evaluator/trace.py
def total_doc_sentences(documents_sentences) -> int          # T
def relevance(relevant_keys, total_sentences) -> float       # |R| / T   (LIST length, no dedup)
def utilization(utilized_keys, total_sentences) -> float     # |U| / T
def completeness(relevant_keys, utilized_keys) -> float      # |R∩U| / |R|;  |R|==0 -> 1.0
def adherence(unsupported_response_sentence_keys) -> bool    # == (list is empty)
def score_from_reference_labels(example) -> dict             # all 4, from gold labels

# src/evaluator/validate.py
def rmse(ours, theirs) -> float
def validate_config(config, n=300) -> dict                   # RMSE/exact for fractions, acc for adherence
def validate_all(n=300) -> list[dict]                        # all 12 configs across 5 domains
```

**Critical implementation rules (learned from EDA — see HLD §8):**
- **List semantics, not set:** count R and U by raw `len(list)` to match how RAGBench computed the
  reference scores; compute `R ∩ U` as sets. Deduping R/U introduces a small systematic error.
- **`|R| == 0` → completeness = 1.0** (frequent in the legal corpus; matches reference).
- **Adherence = `len(unsupported_response_sentence_keys) == 0`.** Do *not* derive it from per-sentence
  `fully_supported` flags — those are `None`/blank in most rows (matched reference only ~32%).

**Validation result:** reproduces reference scores exactly — RMSE = 0 on relevance/utilization/
completeness and 100% adherence accuracy across all 12 configs (3,165 examples sampled).

## 6. The judge bridge ✅ (built — brick 10, `src/judge/`)

The judge produces, for *our* pipeline's output, the same labels RAGBench ships, so the validated math
in §5 can score it unchanged.

```python
# src/judge/base.py
class Judge(Protocol):
    def label(self, question: str, keyed: dict) -> dict: ...   # RAGBench label JSON
# scores_from_label(keyed, label) -> {relevance, utilization, completeness, adherence}
#   (adherence = label["overall_supported"]; fractions via trace.py)
```

- Uses the **EXACT RAGBench Appendix-7.4 labeling prompt** (verbatim, in `base.py`). A strong **open-source**
  model on Colab (`hf` judge, temperature 0, deterministic); an OpenAI judge can drop in via the same interface.
  Robust JSON parsing tolerates ```json fences / surrounding prose. `fake` judge for offline tests.
- **Cache (TODO at scale):** `hash(keyed_example) → JSON` on disk to avoid recompute.
- **Validate before trusting** (`evaluator/judge_validate.py`): run on labelled RAGBench examples, compare judge
  scores to the shipped ones (RMSE on fractions, accuracy on adherence) until agreement is acceptable.

```mermaid
flowchart LR
    PO["PipelineOutput<br/>{answer, context}"] --> SEG["OutputSegmenter"]
    SEG --> KE["KeyedExample"]
    KE --> J["Judge.label()<br/>(cached)"]
    J --> LBL["R / U / support"]
    LBL --> M["trace.py math ✅"]
    M --> TS["TraceScores"]
```

## 7. Registry & factory

```python
# src/registry.py
REGISTRY: dict[str, dict[str, type]] = {}    # REGISTRY["chunker"]["fixed"] = FixedChunker

def register(kind: str, name: str):
    def deco(cls): REGISTRY.setdefault(kind, {})[name] = cls; return cls
    return deco

# src/pipeline.py
def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    chunker  = REGISTRY["chunker"][cfg.chunker.type](**cfg.chunker.params)
    # ... one lookup per stage; None config → NoOp/skip for optional stages ...
    return Pipeline(...)
```

A `None` value for an optional stage (reranker, summarizer, query transform) wires in a no-op, so the
same assembly code handles "ablate this stage OFF".

## 8. Configuration — YAML source of truth + validation + grid generator

**Decision (see HLD §9 / AI_CONTEXT §11.2):** YAML files are the source of truth (declarative,
git-diffable, double as the demo's per-domain configs). A typed schema validates them on load
(fail-fast on typos / wrong types), and a generator expands sweeps so we author *2 files, not 50*.

> **AS-BUILT (brick 9, 7 Jun):** implemented in a SINGLE `src/config.py` (the schema dataclasses live
> there, not a separate `config_schema.py`), with `load_config()` + `from_dict()` + `validate()`. Validation
> checks each stage's `type` against the registry (catches typos/unknown/missing). The **grid generator
> (`expand_grid`) is DEFERRED** — the user chose explicit per-experiment YAML files in `configs/` for now;
> the generator can be added later (the runner consumes config objects either way). The code sketch below is
> the original design intent; treat `config_schema.py`/`expand_grid` as aspirational, not current.

```python
# src/config_schema.py
@dataclass(frozen=True)
class StageConfig:
    type: str
    params: dict = field(default_factory=dict)

@dataclass(frozen=True)
class PipelineConfig:
    domain: str
    chunker: StageConfig
    embedder: StageConfig
    index: StageConfig                  # params include corpus: per_example | pooled
    retriever: StageConfig
    reranker: StageConfig | None
    repacker: StageConfig
    summarizer: StageConfig | None
    prompt: StageConfig
    generator: StageConfig
    seed: int = 42

# src/config.py
def load_config(path: str) -> PipelineConfig: ...      # parse + VALIDATE (raise on bad key/type)
def expand_grid(base: str, sweep: str) -> list[PipelineConfig]: ...  # cartesian product
```

```mermaid
flowchart LR
    BASE["base.yaml +<br/>sweep.yaml"] --> GEN["expand_grid()"]
    GEN --> CFGS["[PipelineConfig, ...]"]
    HAND["hand-written<br/>experiment.yaml"] --> VAL["load_config() + validate"]
    VAL --> CFGS
    CFGS --> FACT["build_pipeline()"]
```

Example experiment (illustrative — shows the *shape*; some types below are aspirational). **For configs
that actually validate against the current registry, see the real files in `configs/`** (e.g.
`grounded_rerank.yaml`) and `configs/README.md`. Registered types today: chunker `fixed`/`none`;
embedder `minilm`/`sentence_transformer`; index `faiss`; retriever `dense`; reranker `cross_encoder`/`none`;
repacker `forward`/`reverse`/`sides`; prompt `grounded`/`minimal`; generator `ollama`/`hf`/`echo`; splitter `regex`/`nltk`.

```yaml
domain: Legal
chunker:    { type: fixed,  size: 512, overlap: 50 }
embedder:   { type: minilm }
index:      { type: faiss, corpus_mode: per_example }   # per_example | pooled
retriever:  { type: dense, k: 20 }
reranker:   { type: cross_encoder, top_n: 5 }           # null to ablate OFF
repacker:   { type: reverse }                           # forward | reverse | sides
prompt:     { type: grounded }
generator:  { type: ollama, model: llama3.1:8b }     # local (Mac); or type: hf for Colab. NON-Chinese only.
splitter:   { type: regex }                             # bridge to the judge
```

> Note: the as-built `PipelineConfig` (`src/config.py`) has no `summarizer` field yet (optional stage,
> deferred) and *adds* a required `splitter` (the segmenter's strategy). The §8 dataclass sketch above
> predates that — `src/config.py` is the source of truth.

## 9. Experiment runner (produces the deliverable matrix)

```python
# src/runner.py — AS-BUILT
def run_experiment(cfg, examples, *, segmenter: OutputSegmenter, judge=None) -> dict:
    # for each example: reset_index → index_documents(its docs) → answer → segment → [judge → score]
    # returns one row: {config_id, domain, stage types, n, n_scored, mean TRACe scores, scoring}
    # judge=None → answers only (scoring="pending"); a Judge object → real scores.
    # A judge that emits unparseable JSON on one example is SKIPPED + counted (n - n_scored), not fatal.

def run_matrix(configs, examples_for, out_csv: str, *, segmenter=None, judge=None) -> None:
    # loop configs; examples_for(cfg) supplies that config's domain examples.
    # writes one row per (config_id, domain) AS IT FINISHES (f.flush) and SKIPS already-done
    # pairs → RESUMABLE across a Colab disconnect.
```

Each row records the config's stage types + its mean scores, so the results matrix is reproducible and
diffable. One component changes at a time across rows — that is what makes the matrix interpretable.
(Notebook 04 adds a `config_name` column keyed on the YAML filename, since `config_id` is built from
stage *types* and would collide for two configs differing only in a param.)

## 10. RGB evaluator (Task 2 — Phase 3)

```python
# src/evaluator/rgb.py
def noise_robustness(pred, gold) -> bool            # accuracy: gold answer string present in pred
def information_integration(pred, gold) -> bool     # accuracy (multi-part)
def negative_rejection(pred) -> bool                # emitted the instructed refusal string
def error_detection(pred) -> bool                   # flagged "factual errors present"
def error_correction(pred, gold) -> bool            # flagged AND gave the corrected answer
```

Inputs are constructed per RGB's protocol (relevant + noisy docs, etc.) with instruction prompts.
Evaluated over 4–5 open-source LLMs (English split). Metrics are deterministic string checks — no judge.

## 11. Error handling & testing

- **Validation-first:** `load_config` raises on malformed config; the runner catches per-experiment
  failures, logs them, and continues so one bad run does not abort a long grid.
- **Determinism:** seeds fixed; generator temperature 0 for evaluation runs.
- **Unit tests:** the TRACe math is tested by reproducing reference scores (already passing);
  components get small interface tests; the judge gets a label-vs-reference validation harness.
- **Local-first:** `StubGenerator` + small embedders let the whole pipeline run on CPU before any
  Colab/GPU work, so wiring bugs are caught cheaply.

---

*Companion document: [`HLD.md`](./HLD.md) — goals, constraints, system context, deployment.*
