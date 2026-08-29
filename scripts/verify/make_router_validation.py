#!/usr/bin/env python3
"""Build the blinded human-validation sample for the claim ROUTER.

Usage (from the repo root):
  python scripts/verify/make_router_validation.py \
      [--per-major 60] [--per-minor 15] [--seed 42]

The 24.9% corpus verifiable-fraction is a gpt-4o-mini routing judgment with
no human error bars (and the router prompt deliberately leans verifiable).
This script samples claims STRATIFIED BY ROUTER LABEL — default 60 routed
verifiable + 60 unverifiable + 15 inference + 15 normative = 150 — shuffles
them, and writes an annotator sheet WITHOUT the router's label (blinded).
The hidden key + per-stratum population counts let score_router_validation.py
recover precision AND population-weighted recall for the verifiable class.

The annotator judges the claim text alone — the same input the router saw.

Writes
  data/router_validation/REVIEW_router.csv   annotator sheet (LABEL column:
      v = checkable against public web sources (news/gov/filings/press)
      u = requires private internal company data
      i = an inference/conclusion, not a checkable fact
      n = opinion or value judgment)
  data/router_validation/router_key.csv      hidden key (do not open while
                                             annotating)
  data/router_validation/strata_counts.json  population counts for weighting
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "verify"
OUT_DIR = REPO_ROOT / "data" / "router_validation"

LABELS = ("verifiable", "unverifiable", "inference", "normative")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-major", type=int, default=60,
                    help="sample size for verifiable and unverifiable strata")
    ap.add_argument("--per-minor", type=int, default=15,
                    help="sample size for inference and normative strata")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-canary-accounts", action="store_true",
                    default=True,
                    help="(default on) only sample non-canary accounts so the "
                         "sheet never contains injected text")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    canary_ids = {p.stem for p in
                  (REPO_ROOT / "data" / "canaries" / "raw").glob("*.json")}

    pool: dict[str, list[dict]] = {l: [] for l in LABELS}
    for path in sorted(CACHE_DIR.glob("*_claims_raw.json")):
        account = path.name.replace("_claims_raw.json", "")
        if args.exclude_canary_accounts and account in canary_ids:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for c in raw.get("claims", []):
            if c.get("is_duplicate"):
                continue
            r = c.get("routing")
            if r in pool:
                pool[r].append({
                    "account_id": account,
                    "claim_id": c.get("claim_id", ""),
                    "claim_text": c["claim_text"],
                    "router_label": r,
                })

    counts = {l: len(pool[l]) for l in LABELS}
    print("population (unique routed claims):", counts,
          "| total:", sum(counts.values()))

    want = {"verifiable": args.per_major, "unverifiable": args.per_major,
            "inference": args.per_minor, "normative": args.per_minor}
    sample: list[dict] = []
    for l in LABELS:
        n = min(want[l], len(pool[l]))
        if n < want[l]:
            print(f"WARN: stratum {l} has only {len(pool[l])} claims",
                  file=sys.stderr)
        sample.extend(rng.sample(pool[l], n))
    rng.shuffle(sample)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet = OUT_DIR / "REVIEW_router.csv"
    key = OUT_DIR / "router_key.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "claim_text",
                    "LABEL (v=public-web checkable / u=private-internal / "
                    "i=inference / n=opinion)"])
        for i, s in enumerate(sample):
            w.writerow([f"R{i:04d}", s["claim_text"], ""])
    with open(key, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "account_id", "claim_id", "router_label"])
        for i, s in enumerate(sample):
            w.writerow([f"R{i:04d}", s["account_id"], s["claim_id"],
                        s["router_label"]])
    (OUT_DIR / "strata_counts.json").write_text(
        json.dumps({"population": counts,
                    "sampled": dict(Counter(s["router_label"]
                                            for s in sample)),
                    "seed": args.seed}, indent=1), encoding="utf-8")

    print(f"sampled {len(sample)} claims -> {sheet}")
    print(f"hidden key -> {key}  (do not open while annotating)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
