"""Config — turn a YAML experiment file into a validated PipelineConfig.

An "experiment = a config" (AI_CONTEXT §11). YAML is the source of truth; this
module loads it and VALIDATES it on the way in, so a typo (`chunkr:`) or wrong
type (`size: "512"`) fails immediately with a clear error — not silently 80
experiments into a results sweep.

Two records + a loader:
  StageConfig     one stage: {type, params}      e.g. type="fixed", params={size:512}
  PipelineConfig  the whole experiment: domain + one StageConfig per stage
  load_config()   read YAML -> PipelineConfig, validating against the registry
"""

from dataclasses import dataclass, field

from .registry import REGISTRY

# The stages a pipeline config must/can name, mapped to their registry "kind".
# (key in the YAML)            (registry kind)   (required?)
REQUIRED_STAGES = {
    "chunker": "chunker",
    "embedder": "embedder",
    "index": "index",
    "retriever": "retriever",
    "prompt": "prompt",
    "generator": "generator",
    "splitter": "splitter",
}
OPTIONAL_STAGES = {            # may be null/absent → that stage is skipped/needs care
    "reranker": "reranker",
    "repacker": "repacker",
    "summarizer": "summarizer",   # context compression (trim chunks to query-relevant sentences); None → skip
}


@dataclass(frozen=True)
class StageConfig:
    type: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    domain: str
    chunker: StageConfig
    embedder: StageConfig
    index: StageConfig
    retriever: StageConfig
    prompt: StageConfig
    generator: StageConfig
    splitter: StageConfig
    reranker: StageConfig | None = None
    repacker: StageConfig | None = None
    summarizer: StageConfig | None = None
    seed: int = 42


def _parse_stage(name: str, raw) -> StageConfig:
    """Turn one YAML stage entry into a StageConfig, validating its shape."""
    if not isinstance(raw, dict):
        raise ValueError(f"stage {name!r} must be a mapping with a 'type', got {raw!r}")
    if "type" not in raw:
        raise ValueError(f"stage {name!r} is missing required key 'type'")
    params = {k: v for k, v in raw.items() if k != "type"}
    return StageConfig(type=str(raw["type"]), params=params)


def _check_type_registered(stage_key: str, kind: str, cfg: StageConfig) -> None:
    """Fail fast if the named type isn't registered (catches typos + unregistered)."""
    bucket = REGISTRY.get(kind, {})
    if cfg.type not in bucket:
        available = sorted(bucket) or ["(none registered — did you import/load it?)"]
        raise ValueError(
            f"stage {stage_key!r}: unknown {kind} type {cfg.type!r}. Available: {available}"
        )


def validate(cfg: PipelineConfig) -> None:
    """Validate every named stage's type against the registry. Raises on problems.

    NOTE: heavy strategies (embedder, hf generator, nltk splitter) only register
    after their load_*() is called — so run that before validating a config that
    uses them (e.g. on Colab). Validation error messages say so.
    """
    if not cfg.domain:
        raise ValueError("config is missing 'domain'")
    for key, kind in REQUIRED_STAGES.items():
        _check_type_registered(key, kind, getattr(cfg, key))
    for key, kind in OPTIONAL_STAGES.items():
        stage = getattr(cfg, key)
        if stage is not None:
            _check_type_registered(key, kind, stage)


def from_dict(d: dict, *, do_validate: bool = True) -> PipelineConfig:
    """Build a PipelineConfig from a plain dict (e.g. parsed YAML)."""
    if "domain" not in d:
        raise ValueError("config is missing required key 'domain'")
    missing = [s for s in REQUIRED_STAGES if s not in d]
    if missing:
        raise ValueError(f"config is missing required stage(s): {missing}")

    kwargs = {"domain": d["domain"], "seed": d.get("seed", 42)}
    for key in REQUIRED_STAGES:
        kwargs[key] = _parse_stage(key, d[key])
    for key in OPTIONAL_STAGES:
        kwargs[key] = _parse_stage(key, d[key]) if d.get(key) is not None else None

    cfg = PipelineConfig(**kwargs)
    if do_validate:
        validate(cfg)
    return cfg


def load_config(path: str, *, do_validate: bool = True) -> PipelineConfig:
    """Read a YAML experiment file and return a validated PipelineConfig."""
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"config file {path!r} did not parse to a mapping")
    return from_dict(d, do_validate=do_validate)
