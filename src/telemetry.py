"""Per-stage telemetry — write one JSONL record per (config, domain, example).

The matrix CSV (run_experiment) gives one AGGREGATE row per (config × domain) — fine
for the final table but blind to per-example behaviour. This module adds a parallel
per-example trace so we can retrospect on failures without re-running:

    results/telemetry/legal/<config_name>__<domain>.jsonl

Each line is one example, JSON, with the same shape so jq / pandas can read it:

    {"config_name": ..., "domain": ..., "example_index": 0,
     "question": "<truncated>", "answer": "<truncated>",
     "n_sources": 5, "context_chars": 2480,
     "scores": {"relevance": 0.45, "utilization": 0.29,
                "completeness": 0.64, "adherence": true},
     "cost": {"total_usd": 0.5421, "gen_calls": 240, "judge_calls": 240},
     "ts": "2026-06-21T..."}

Design notes:
  - Append-only: one open()→write→close per record so a crash/SIGKILL between
    examples leaves the file consistent (no half-written lines).
  - Failures in the logger are SWALLOWED (try/except in the runner caller).
    A broken trace must never crash the experiment — the matrix CSV is the
    contractually-required artifact; this file is for diagnostics.
  - Question/answer fields are truncated to 500 chars — full CUAD answers can
    be long, and the trace is for sniff-tests not for re-running. The matrix
    CSV doesn't lose anything (it never had Q/A text).
  - Cost snapshot is read from cost_tracker (the running file totals) at write
    time, NOT recomputed — single source of truth for $.

Public API: log_example(config_name, domain, example_index, ex, out, scores)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TELEMETRY_DIR = "results/telemetry/legal"
MAX_TEXT_CHARS = 500          # truncate Q/A so per-example records stay grep-friendly


def _trace_dir() -> Path:
    return Path(os.getenv("CAPSTONERAG_TELEMETRY_DIR", DEFAULT_TELEMETRY_DIR))


def _trunc(s: Any, n: int = MAX_TEXT_CHARS) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n} chars)"


def _cost_snapshot() -> dict:
    """Read the cost-tracker totals without coupling to its internals.

    Returns the four numbers we want per row: total $, gen calls, judge calls,
    gen+judge tokens. If cost_tracker isn't initialised yet (or its file is
    missing — e.g. on the local-Ollama path where nothing has been recorded),
    return zeros silently.
    """
    try:
        from . import cost_tracker
        path = cost_tracker._state_path()
        if not path.exists():
            return {"total_usd": 0.0, "gen_calls": 0, "judge_calls": 0,
                    "gen_in_tokens": 0, "gen_out_tokens": 0,
                    "judge_in_tokens": 0, "judge_out_tokens": 0}
        with path.open() as f:
            s = json.load(f)
        return {
            "total_usd": s.get("total_usd", 0.0),
            "gen_calls": s.get("gen", {}).get("calls", 0),
            "judge_calls": s.get("judge", {}).get("calls", 0),
            "gen_in_tokens": s.get("gen", {}).get("in_tokens", 0),
            "gen_out_tokens": s.get("gen", {}).get("out_tokens", 0),
            "judge_in_tokens": s.get("judge", {}).get("in_tokens", 0),
            "judge_out_tokens": s.get("judge", {}).get("out_tokens", 0),
        }
    except Exception:
        return {}


def _safe_filename(s: str) -> str:
    """Strip path separators so a config name like 'foo/bar' can't escape the dir."""
    return s.replace("/", "_").replace("\\", "_")


def log_example(*, config_name: str, domain: str, example_index: int,
                ex: dict, out: dict, scores: dict | None) -> None:
    """Append one JSONL record for this example. SWALLOWS errors — see module docstring.

    `out` is the pipeline.answer() return dict {answer, sources, context,
    needs_retrieval}. `scores` is the 4-key TRACe dict, or None if scoring
    was skipped/failed.
    """
    try:
        out_dir = _trace_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{_safe_filename(config_name)}__{_safe_filename(domain)}.jsonl"
        sources = out.get("sources") or []
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_name": config_name,
            "domain": domain,
            "example_index": example_index,
            "example_id": ex.get("id") or f"{ex.get('_config', '')}_{example_index}",
            "question": _trunc(ex.get("question")),
            "answer": _trunc(out.get("answer")),
            "needs_retrieval": out.get("needs_retrieval", True),
            "n_sources": len(sources),
            "context_chars": len(out.get("context") or ""),
            "scores": scores,                       # None when judge skipped/failed
            "cost": _cost_snapshot(),
        }
        # Append one line, flushed + closed before returning — atomic per record.
        with path.open("a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        # Telemetry MUST NOT break experiments. Silent swallow is intentional;
        # if the file is broken/missing the matrix CSV still has the row.
        return
