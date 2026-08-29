#!/usr/bin/env python3
"""Score the aligner against the operator's labels -> precision/recall for the paper.

Usage (from the repo root, after the LABEL column is filled):
  python scripts/aligner/score_validation.py

Reads data/aligner/{REVIEW_alignments.csv, validation_key.csv}; writes
data/aligner/aligner_validation_report.md.

Sample design (make_validation_sample.py): ~50% positives (pairs the aligner
linked with an asserting/hedging relation) and ~50% hard negatives (same-task
pairs the aligner did NOT link; topical near-misses). Human labels y/hedge/n;
y and hedge both count as "human says linked" (hedge is a sub-type of link).

Metrics
  precision  = P(human linked | aligner linked)        [from positive rows]
  miss rate  = P(human linked | aligner did not link)  [from negative rows;
               estimates false-negative rate on same-deck near-miss pairs,
               the hardest negative population]
  Wilson 95% CIs on both.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIR = REPO_ROOT / "data" / "aligner"

POSITIVE = {"asserts", "hedges", "asserts_falsified", "asserts_true", "hedged"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    key = {r["pair_id"]: r for r in
           csv.DictReader(open(DIR / "validation_key.csv", encoding="utf-8"))}
    pos_n = pos_y = neg_n = neg_y = skipped = 0
    disagreements: list[str] = []
    with open(DIR / "REVIEW_alignments.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pid, label = row[0], (row[3] if len(row) > 3 else "").strip().lower()
            if label not in ("y", "hedge", "n"):
                if label:
                    print(f"SKIP {pid}: unrecognized label {label!r}",
                          file=sys.stderr)
                skipped += 1
                continue
            human_link = label in ("y", "hedge")
            model_link = key[pid]["model_relation"] in POSITIVE
            if model_link and human_link:
                pos_y += 1
            elif model_link:
                pos_n += 1
                disagreements.append(f"FP {pid} (model {key[pid]['model_relation']})")
            elif human_link:
                neg_y += 1
                disagreements.append(f"MISS {pid}")
            else:
                neg_n += 1

    n_pos, n_neg = pos_y + pos_n, neg_y + neg_n
    prec = pos_y / n_pos if n_pos else float("nan")
    miss = neg_y / n_neg if n_neg else float("nan")
    plo, phi = wilson(pos_y, n_pos)
    mlo, mhi = wilson(neg_y, n_neg)

    lines = [
        "# Aligner validation report (prompt aligner-v1, Claude Haiku 4.5)",
        "",
        f"Labeled: {n_pos + n_neg} pairs ({n_pos} aligner-linked, "
        f"{n_neg} hard negatives); skipped {skipped}",
        "",
        f"- **Precision** P(human links | aligner links): "
        f"**{prec:.1%}** ({pos_y}/{n_pos}), Wilson 95% CI [{plo:.1%}, {phi:.1%}]",
        f"- **Miss rate on same-deck near-miss negatives**: "
        f"**{miss:.1%}** ({neg_y}/{n_neg}), Wilson 95% CI [{mlo:.1%}, {mhi:.1%}]",
        f"- Negative predictive value on this hard-negative population: "
        f"{1 - miss:.1%} (not recall, which this 50/50 design cannot recover)",
        "",
        "Disagreements (for audit): " + (", ".join(disagreements) or "none"),
        "",
        "Use: report alongside the E1/E2 endpoints as a bound on their "
        "interpretation. The endpoint CIs are deck-level bootstrap intervals "
        "and do not themselves incorporate the aligner's error rate.",
    ]
    out = DIR / "aligner_validation_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
