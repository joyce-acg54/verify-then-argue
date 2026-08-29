"""Token/cost accounting for the debate harness.

Every LLM call is appended as one JSON line to data/cache/cost_log.jsonl.
Single-line O_APPEND writes are atomic enough for concurrent processes.

Prices verified 2026-06-11:
- gpt-4o-mini: $0.15 / $0.60 per 1M input/output tokens (platform.openai.com/pricing)
- gpt-4o:      $2.50 / $10.00 per 1M input/output tokens (platform.openai.com/pricing)
- meta-llama/Llama-3.3-70B-Instruct-Turbo on Together: $0.88 per 1M tokens,
  flat for input AND output (https://www.together.ai/pricing; cross-checked at
  https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COST_LOG_PATH = REPO_ROOT / "data" / "cache" / "cost_log.jsonl"

# (input $/1M tokens, output $/1M tokens)
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
}


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost in USD for a single call. Unknown models cost 0 (and are flagged)."""
    in_price, out_price = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


def log_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    stage: str = "",
    company_id: str = "",
    condition: str = "",
    seed: int | None = None,
    latency_s: float | None = None,
) -> float:
    """Append one call record to the cost log; returns the call's USD cost."""
    cost = compute_cost(model, prompt_tokens, completion_tokens)
    record = {
        "ts": time.time(),
        "model": model,
        "known_price": model in PRICES,
        "stage": stage,
        "company_id": company_id,
        "condition": condition,
        "seed": seed,
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
    return cost
