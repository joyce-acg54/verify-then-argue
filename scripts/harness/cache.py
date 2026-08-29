"""Per-(company_id, condition, run_seed) JSON result cache.

One file per key under data/cache/harness/. Writes are atomic
(tempfile in the same directory + os.replace), so concurrent processes
can share the cache safely; readers never see partial files.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Cache namespace is env-overridable so a different-model run gets a clean cache
# instead of colliding with the published-config results (key omits the model).
CACHE_DIR = REPO_ROOT / "data" / "cache" / os.environ.get("HARNESS_CACHE_NS", "harness")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def cache_path(company_id: str, condition: str, seed: int) -> Path:
    return CACHE_DIR / f"{_safe(company_id)}__{_safe(condition)}__seed{int(seed)}.json"


def get(company_id: str, condition: str, seed: int) -> dict | None:
    """Return cached result dict, or None on miss/corrupt file."""
    path = cache_path(company_id, condition, seed)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def put(company_id: str, condition: str, seed: int, result: dict) -> Path:
    """Atomically write result dict for the key; returns the cache path."""
    path = cache_path(company_id, condition, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)  # atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
