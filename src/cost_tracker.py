"""Groq cost accounting + hard-stop tripwire.

Why this exists: the Legal-domain sweep (see docs/LEGAL_EXPERIMENT_PLAN.md) runs
two paid Groq legs per example — the RAG generator AND the TRACe scoring judge.
The user-set cap is $5; we want to know in REAL TIME how much we've spent and to
HARD-ABORT before we blow past it (Groq doesn't refund overshoots).

Mechanism:
  - groq_generator.py and groq_judge.py call `record(kind, usage, model)` after
    every successful response. `usage` is the OpenAI-style `{prompt_tokens,
    completion_tokens}` block that Groq returns on the chat-completions endpoint.
  - We translate tokens → $ using PRICES (a SINGLE source of truth — update here
    when Groq changes rates), append to a running totals file, and raise
    BudgetExceeded if cumulative cost > HARD_CAP.
  - The runner catches BudgetExceeded and stops the sweep CLEANLY (partial CSV
    rows are already flushed by run_named_matrix's open->write->close-per-row).

Failure mode: if recording fails (disk full, bad JSON, …), the hook is wrapped
in try/except in the callers so a broken tracker DOES NOT crash the experiment.
The cap-enforcement raise is the one path that DOES propagate.

State file (results/telemetry/legal/_cost_running.json):
    {"gen":   {"calls": N, "in_tokens": ..., "out_tokens": ..., "cost_usd": ...},
     "judge": {"calls": N, "in_tokens": ..., "out_tokens": ..., "cost_usd": ...},
     "total_usd": ..., "hard_cap_usd": 4.50,
     "since": "<ISO timestamp of first record>",
     "last_update": "<ISO timestamp>"}

Pricing source: https://console.groq.com/settings/billing — verified on launch.
The user is responsible for re-checking before each multi-phase run.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


# ── Pricing table (USD per 1M tokens). UPDATE HERE if Groq changes rates. ──────
# Source: https://console.groq.com/settings/billing — re-verify before launch.
PRICES = {
    # llama-3.3-70b-versatile  (default for our 70B leg)
    "llama-3.3-70b-versatile":  {"in": 0.59, "out": 0.79},
    # llama-3.1-70b-versatile  (fallback if Groq retires 3.3)
    "llama-3.1-70b-versatile":  {"in": 0.59, "out": 0.79},
    # llama-3.1-8b-instant     (cheaper option; included for completeness)
    "llama-3.1-8b-instant":     {"in": 0.05, "out": 0.08},
}

# Hard ceiling — RAISES BudgetExceeded once cumulative cost crosses this number.
# Set well below the user-stated $5 cap so the worst-case overshoot (one extra
# call after the threshold) cannot bust the contract.
HARD_CAP_USD = 4.50

# Default state-file location. Override with CAPSTONERAG_COST_FILE env var.
DEFAULT_STATE_FILE = "results/telemetry/legal/_cost_running.json"

_LOCK = threading.Lock()


class BudgetExceeded(RuntimeError):
    """Raised when cumulative Groq cost has crossed HARD_CAP_USD.

    Caught by the runner; uncaught it propagates up and ends the process.
    The state file is ALWAYS written before the raise, so the breach amount
    is visible for forensics.
    """


def _state_path() -> Path:
    return Path(os.getenv("CAPSTONERAG_COST_FILE", DEFAULT_STATE_FILE))


def _empty_state() -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "gen":   {"calls": 0, "in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0},
        "judge": {"calls": 0, "in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0},
        "total_usd": 0.0,
        "hard_cap_usd": HARD_CAP_USD,
        "since": now,
        "last_update": now,
    }


def _load() -> dict:
    p = _state_path()
    if not p.exists():
        return _empty_state()
    try:
        with p.open() as f:
            data = json.load(f)
        # Heal a state file that's missing newer keys (forward-compatible).
        base = _empty_state()
        for k, v in base.items():
            data.setdefault(k, v)
        for leg in ("gen", "judge"):
            for k, v in base[leg].items():
                data[leg].setdefault(k, v)
        return data
    except (json.JSONDecodeError, OSError):
        # Corrupt — start fresh; the old file gets overwritten on next write.
        return _empty_state()


def _save(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename — atomic on POSIX, so a SIGKILL
    # mid-write can never corrupt the running totals.
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(p)


def _price_for(model: str) -> dict:
    """Return {in, out} $/1M-token rates for `model`. Falls back to the most
    expensive entry in PRICES so an unknown model can't UNDERSTATE cost.
    """
    if model in PRICES:
        return PRICES[model]
    # Conservative fallback: use the max rate across the table so an unknown
    # model always bills HIGHER than reality. Better safe than over-budget.
    in_max = max(v["in"] for v in PRICES.values())
    out_max = max(v["out"] for v in PRICES.values())
    return {"in": in_max, "out": out_max}


def record(kind: Literal["gen", "judge"], usage: dict | None, model: str) -> dict:
    """Append one Groq call's tokens + cost to the running tally and persist.

    `usage` is the response['usage'] block — Groq returns {prompt_tokens,
    completion_tokens, total_tokens} on the OpenAI-compatible endpoint. A
    missing/None usage is logged with 0 tokens (the call still counts toward
    the `calls` total so we can see how often Groq didn't report usage).

    Returns the updated state dict so callers can log a one-liner if desired.

    Raises BudgetExceeded if cumulative cost has crossed HARD_CAP_USD AFTER
    this call. The state file is written FIRST, so the breach is forensically
    visible even when the run is killed by the exception.
    """
    if kind not in ("gen", "judge"):
        raise ValueError(f"record: kind must be 'gen' or 'judge', got {kind!r}")

    in_tok = int((usage or {}).get("prompt_tokens", 0) or 0)
    out_tok = int((usage or {}).get("completion_tokens", 0) or 0)
    p = _price_for(model)
    cost = in_tok * p["in"] / 1_000_000 + out_tok * p["out"] / 1_000_000

    with _LOCK:
        state = _load()
        leg = state[kind]
        leg["calls"] += 1
        leg["in_tokens"] += in_tok
        leg["out_tokens"] += out_tok
        leg["cost_usd"] = round(leg["cost_usd"] + cost, 6)
        state["total_usd"] = round(state["gen"]["cost_usd"] + state["judge"]["cost_usd"], 6)
        state["last_update"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(state)

    if state["total_usd"] > HARD_CAP_USD:
        raise BudgetExceeded(
            f"Groq cost {state['total_usd']:.4f} USD exceeded hard cap "
            f"{HARD_CAP_USD:.2f} USD (gen={state['gen']['cost_usd']:.4f}, "
            f"judge={state['judge']['cost_usd']:.4f}, calls "
            f"gen/judge={state['gen']['calls']}/{state['judge']['calls']}). "
            f"State file: {_state_path()}"
        )
    return state


def current_total() -> float:
    """Return cumulative Groq cost in USD (cheap; just reads the file)."""
    return _load()["total_usd"]


def format_one_liner() -> str:
    """Human-readable summary of where we are vs the cap. Useful in logs."""
    s = _load()
    pct = (s["total_usd"] / HARD_CAP_USD * 100) if HARD_CAP_USD else 0.0
    return (f"[cost] ${s['total_usd']:.4f} / ${HARD_CAP_USD:.2f} ({pct:.1f}%)  "
            f"gen={s['gen']['cost_usd']:.4f} ({s['gen']['calls']} calls)  "
            f"judge={s['judge']['cost_usd']:.4f} ({s['judge']['calls']} calls)")


def reset() -> None:
    """Wipe the running totals. Use only between phases / for tests."""
    with _LOCK:
        _save(_empty_state())
