"""Cost accounting for the verification pipeline.

Reuses scripts/harness/cost.py (loaded by file path; the harness is NOT
modified): same COST_LOG_PATH (data/cache/cost_log.jsonl), same record keys,
same per-token price table extended at runtime with:

  - sonar:                  $1.00 / $1.00 per 1M input/output tokens
                            (https://docs.perplexity.ai/getting-started/pricing)
                            PLUS $5 per 1000 search requests ($0.005/request,
                            "low" search context tier) folded into cost_usd.
  - exa:                    $0 token price; the actual request cost (search +
                            text contents) is taken from the API response's
                            costDollars.total (fallback: $7/1k searches +
                            $1/1k text pages) and folded into cost_usd via
                            request_fee_usd.
  - text-embedding-3-small: $0.02 / 1M input tokens.

Records are written with the exact harness schema (ts, model, known_price,
stage, company_id, condition, seed, prompt_tokens, completion_tokens,
cost_usd, latency_s, pid). condition is left "" for verification-stage calls.
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parent
REPO_ROOT = VERIFY_DIR.parents[1]
_HARNESS_COST_PATH = REPO_ROOT / "scripts" / "harness" / "cost.py"

_spec = importlib.util.spec_from_file_location("harness_cost", _HARNESS_COST_PATH)
harness_cost = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness_cost)

# Extend the harness price table in-memory (file untouched).
harness_cost.PRICES.setdefault("sonar", (1.00, 1.00))
harness_cost.PRICES.setdefault("exa", (0.0, 0.0))  # request-fee priced
harness_cost.PRICES.setdefault("text-embedding-3-small", (0.02, 0.0))

SONAR_SEARCH_FEE_USD = 0.005  # $5 per 1k requests, low search context tier

COST_LOG_PATH = harness_cost.COST_LOG_PATH

_lock = threading.Lock()
_session = {
    "cost_usd": 0.0,
    "n_calls": 0,
    "by_stage": defaultdict(lambda: {"calls": 0, "cost_usd": 0.0,
                                     "prompt_tokens": 0, "completion_tokens": 0}),
}
_company_id = ""


def set_company(company_id: str) -> None:
    global _company_id
    _company_id = company_id


def log_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    stage: str,
    latency_s: float | None = None,
    request_fee_usd: float = 0.0,
) -> float:
    """Append one harness-format record; returns the call's USD cost
    (token cost + any flat request fee, e.g. Sonar search pricing)."""
    cost = harness_cost.compute_cost(model, prompt_tokens, completion_tokens)
    cost += request_fee_usd
    record = {
        "ts": time.time(),
        "model": model,
        "known_price": model in harness_cost.PRICES,
        "stage": stage,
        "company_id": _company_id,
        "condition": "",
        "seed": None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 8),
        "latency_s": round(latency_s, 3) if latency_s is not None else None,
        "pid": os.getpid(),
    }
    COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    fd = os.open(COST_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)

    with _lock:
        _session["cost_usd"] += cost
        _session["n_calls"] += 1
        s = _session["by_stage"][stage]
        s["calls"] += 1
        s["cost_usd"] += cost
        s["prompt_tokens"] += prompt_tokens
        s["completion_tokens"] += completion_tokens
    return cost


def log_openai_response(response, stage: str, model: str | None = None,
                        latency_s: float | None = None,
                        request_fee_usd: float = 0.0) -> float:
    """Convenience wrapper for openai-client responses (chat or embeddings)."""
    usage = response.usage
    pt = getattr(usage, "prompt_tokens", None)
    if pt is None:  # embeddings usage has only prompt_tokens/total_tokens
        pt = getattr(usage, "total_tokens", 0)
    ct = getattr(usage, "completion_tokens", 0) or 0
    return log_call(model or response.model, pt, ct, stage,
                    latency_s=latency_s, request_fee_usd=request_fee_usd)


def session_summary() -> dict:
    with _lock:
        return {
            "cost_usd": round(_session["cost_usd"], 6),
            "n_calls": _session["n_calls"],
            "by_stage": {k: {**v, "cost_usd": round(v["cost_usd"], 6)}
                         for k, v in _session["by_stage"].items()},
        }


def reset_session() -> None:
    with _lock:
        _session["cost_usd"] = 0.0
        _session["n_calls"] = 0
        _session["by_stage"].clear()
