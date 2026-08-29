#!/usr/bin/env python3
"""Build the blinded human-validation sample for the aligner (150-200 pairs).

Usage (from the repo root):
  python scripts/aligner/make_validation_sample.py [--n 180] [--seed 42]

Samples (premise, candidate) pairs from data/aligner/alignments.jsonl:
  ~50% positives  — pairs the aligner linked (asserts / hedges, claim or canary)
  ~50% negatives  — same-task pairs the aligner did NOT link (hard negatives:
                    candidate from the same deck, so topical overlap is common)

Writes
  data/aligner/REVIEW_alignments.csv  the annotator sheet (pair_id, premise,
      candidate text, blank LABEL column) — shuffled, no condition info, no
      model output
  data/aligner/validation_key.csv     the hidden key (pair_id -> task_id,
      candidate ref, model relation) used to compute precision/recall after
      annotation. DO NOT open while annotating.

Annotator instruction (also embedded in the sheet header row):
  LABEL = y      the premise asserts the candidate's factual content
        = hedge  it conveys the content but with an uncertainty marker
        = n      it does not reuse the candidate's factual content
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

POSITIVE_RELATIONS = {"asserts", "hedges", "asserts_falsified",
                      "asserts_true", "hedged"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    align_path = common.ALIGNER_DIR / "alignments.jsonl"
    if not align_path.is_file():
        print("ERROR: run collect.py first", file=sys.stderr)
        return 1

    positives: list[dict] = []   # linked pairs
    negatives: list[dict] = []   # same-task unlinked pairs

    with open(align_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            task = common.load_json(common.task_path(rec["task_id"]))
            claims = {c["idx"]: c["text"] for c in task["claims"]}
            canaries = {c["id"]: c for c in task["canaries"]}
            for a in rec["result"]["arguments"]:
                for p in a["premises"]:
                    linked_claims = {l["claim_idx"]: l["relation"]
                                     for l in p.get("claim_links", [])}
                    linked_canaries = {l["canary_id"]: l["relation"]
                                       for l in p.get("canary_links", [])}
                    for ci, rel in linked_claims.items():
                        if rel in POSITIVE_RELATIONS and ci in claims:
                            positives.append({
                                "task_id": rec["task_id"],
                                "premise": p["premise"],
                                "candidate": claims[ci],
                                "ref": f"claim:{ci}",
                                "model_relation": rel,
                            })
                    for cid, rel in linked_canaries.items():
                        if rel in POSITIVE_RELATIONS and cid in canaries:
                            c = canaries[cid]
                            positives.append({
                                "task_id": rec["task_id"],
                                "premise": p["premise"],
                                "candidate": (f"{c['fact_type']}: "
                                              f"'{c['true_span']}' OR "
                                              f"'{c['falsified_span']}'"),
                                "ref": f"canary:{cid}",
                                "model_relation": rel,
                            })
                    unlinked = [ci for ci in claims if ci not in linked_claims]
                    rng.shuffle(unlinked)
                    for ci in unlinked[:2]:
                        negatives.append({
                            "task_id": rec["task_id"],
                            "premise": p["premise"],
                            "candidate": claims[ci],
                            "ref": f"claim:{ci}",
                            "model_relation": "none",
                        })

    n_pos = min(len(positives), args.n // 2)
    n_neg = min(len(negatives), args.n - n_pos)
    sample = rng.sample(positives, n_pos) + rng.sample(negatives, n_neg)
    rng.shuffle(sample)

    common.ALIGNER_DIR.mkdir(parents=True, exist_ok=True)
    sheet = common.ALIGNER_DIR / "REVIEW_alignments.csv"
    key = common.ALIGNER_DIR / "validation_key.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "premise", "candidate",
                    "LABEL (y / hedge / n)"])
        for i, s in enumerate(sample):
            w.writerow([f"P{i:04d}", s["premise"], s["candidate"], ""])
    with open(key, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "task_id", "ref", "model_relation"])
        for i, s in enumerate(sample):
            w.writerow([f"P{i:04d}", s["task_id"], s["ref"],
                        s["model_relation"]])

    print(f"sampled {n_pos} positives + {n_neg} negatives "
          f"(pool: {len(positives)}/{len(negatives)})")
    print(f"annotator sheet -> {sheet}")
    print(f"hidden key      -> {key}  (do not open while annotating)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
