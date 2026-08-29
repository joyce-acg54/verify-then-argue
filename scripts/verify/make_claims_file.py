#!/usr/bin/env python3
"""Merge extraction + verification into the harness claims file.

Usage:
  python scripts/verify/make_claims_file.py --account <id>

Writes data/claims/<account_id>.json in the schema the harness expects
(scripts/harness/run_batch.py + debate.build_evidence_block):

  [
    {"claim": str,
     "verdict": "belief"|"disbelief"|"ignorance"|"no_evidence"|null,
     "source_reliability": float|null,
     "consistency": float|null,            # 1 - normalized verdict entropy
     "routing": "verifiable"|"unverifiable"|"inference"|"normative"},
    ...
  ]

Claims routed unverifiable/inference/normative get verdict null (plus the
routing label); the harness reads only the keys it knows. Duplicate claims
are excluded. Claims whose verification failed entirely (api_error) also get
verdict null and are flagged with "error": "api_error".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

HARNESS_VERDICTS = {"belief", "disbelief", "ignorance", "no_evidence"}


def make_claims(account_id: str) -> Path:
    raw_path = common.claims_raw_path(account_id)
    if not raw_path.exists():
        raise FileNotFoundError(f"no extraction output at {raw_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    verify_path = common.verify_jsonl_path(account_id)
    verified: dict[str, dict] = {}
    if verify_path.exists():
        with open(verify_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    verified[r["claim_hash"]] = r  # last write wins on resume
                except (json.JSONDecodeError, KeyError):
                    continue

    out = []
    missing = 0
    for c in raw["claims"]:
        if c.get("is_duplicate"):
            continue
        routing = c.get("routing")
        row: dict = {
            "claim": c["claim_text"],
            "verdict": None,
            "source_reliability": None,
            "consistency": None,
            "routing": routing,
        }
        if routing == "verifiable":
            v = verified.get(common.claim_hash(c["claim_text"]))
            if v is None:
                missing += 1
            elif v["final_label"] in HARNESS_VERDICTS:
                row["verdict"] = v["final_label"]
                row["source_reliability"] = round(v["source_score"], 2)
                row["consistency"] = round(v["consistency"], 2)
            else:  # api_error
                row["error"] = v["final_label"]
        out.append(row)

    if missing:
        print(f"WARNING: {missing} verifiable claims have no verification "
              f"record yet — run verify_claims.py to completion first")

    common.CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = common.CLAIMS_DIR / f"{account_id}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    n_v = sum(1 for r in out if r["verdict"] is not None)
    print(f"[{account_id}] wrote {out_path}: {len(out)} claims "
          f"({n_v} with verdicts, {len(out) - n_v} null-verdict)")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--account", required=True)
    args = ap.parse_args()
    make_claims(args.account)
    return 0


if __name__ == "__main__":
    sys.exit(main())
