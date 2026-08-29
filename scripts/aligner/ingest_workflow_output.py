#!/usr/bin/env python3
"""Write aligner results from a workflow output file to data/aligner/raw/.

Usage:
  python scripts/aligner/ingest_workflow_output.py <workflow_output.json> [...]

Each workflow output file is the JSON the harness writes for a completed
background workflow: {"result": [{"tid": ..., "result": {...}}, ...]}.
Existing result files are overwritten (last run wins).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def main() -> int:
    n = 0
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        doc = json.loads(Path(arg).read_text(encoding="utf-8"))
        results = doc.get("result") or []
        for r in results:
            if not r or "tid" not in r or not r.get("result"):
                continue
            out = common.result_path(r["tid"])
            out.write_text(json.dumps(r["result"], indent=1,
                                      ensure_ascii=False), encoding="utf-8")
            n += 1
        print(f"{arg}: {len(results)} results")
    print(f"wrote {n} result files -> {common.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
