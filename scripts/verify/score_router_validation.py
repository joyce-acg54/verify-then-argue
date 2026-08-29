#!/usr/bin/env python3
"""Score the router against the human labels (annotated by one co-author); emit calibrated verifiable-fraction.

Usage (from the repo root, AFTER the LABEL column is filled in):
  python scripts/verify/score_router_validation.py

Reads data/router_validation/{REVIEW_router.csv, router_key.csv,
strata_counts.json}; writes data/router_validation/router_validation_report.md.

What it computes
  - 4x4 confusion matrix (router label vs human label) + per-class agreement.
  - Binary precision for the verifiable class:
        P(human says v | router said verifiable)            [from the v stratum]
  - Population-weighted binary recall for the verifiable class. Sampling is
    stratified, so raw recall would be biased; instead:
        recall = W_v * P(h=v | r=v) / sum_s W_s * P(h=v | r=s)
    where W_s is stratum s's share of the routed population.
  - The CALIBRATED verifiable fraction with a 95% CI:
        true_frac ~= sum_s W_s * P(h=v | r=s)
    (i.e. what fraction of ALL claims a human would call verifiable),
    with a normal-approximation CI propagated from per-stratum binomials.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "router_validation"

HUMAN_TO_LABEL = {"v": "verifiable", "u": "unverifiable",
                  "i": "inference", "n": "normative"}
LABELS = ("verifiable", "unverifiable", "inference", "normative")


def main() -> int:
    sheet = OUT_DIR / "REVIEW_router.csv"
    key = OUT_DIR / "router_key.csv"
    strata = json.loads((OUT_DIR / "strata_counts.json").read_text())

    with open(key, newline="", encoding="utf-8") as f:
        router = {r["pair_id"]: r["router_label"] for r in csv.DictReader(f)}

    human: dict[str, str] = {}
    unlabeled = 0
    with open(sheet, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pid, _, label = row[0], row[1], (row[2] if len(row) > 2 else "")
            label = label.strip().lower()
            if label in HUMAN_TO_LABEL:
                human[pid] = HUMAN_TO_LABEL[label]
            elif label:
                print(f"WARN {pid}: unrecognized label {label!r} — skipped",
                      file=sys.stderr)
                unlabeled += 1
            else:
                unlabeled += 1
    if not human:
        print("ERROR: no labels found — fill the LABEL column first",
              file=sys.stderr)
        return 1
    if unlabeled:
        print(f"NOTE: {unlabeled} rows unlabeled/skipped", file=sys.stderr)

    # Confusion matrix and per-stratum P(human=v | router=s)
    conf: dict[str, Counter] = defaultdict(Counter)
    for pid, h in human.items():
        conf[router[pid]][h] += 1

    pop = strata["population"]
    w = {s: pop[s] / sum(pop.values()) for s in LABELS}

    p_v_given = {}   # P(human=v | router=s) with sample sizes
    for s in LABELS:
        n_s = sum(conf[s].values())
        k_s = conf[s].get("verifiable", 0)
        p_v_given[s] = (k_s / n_s if n_s else 0.0, k_s, n_s)

    precision = p_v_given["verifiable"][0]
    contrib = {s: w[s] * p_v_given[s][0] for s in LABELS}
    true_frac = sum(contrib.values())
    recall = contrib["verifiable"] / true_frac if true_frac else float("nan")

    # 95% CI on true_frac via independent per-stratum binomial variances
    var = 0.0
    for s in LABELS:
        p, _, n_s = p_v_given[s]
        if n_s:
            var += (w[s] ** 2) * p * (1 - p) / n_s
    ci = 1.96 * math.sqrt(var)

    # Cohen's kappa (4-class) on the labeled sample (stratified — report as-is
    # with the caveat that it is per-stratum agreement, not population kappa)
    n = len(human)
    po = sum(conf[s].get(s, 0) for s in LABELS) / n
    pe = sum((sum(conf[s].values()) / n) *
             (sum(conf[r].get(s, 0) for r in LABELS) / n) for s in LABELS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    lines = [
        "# Router validation report",
        "",
        f"Labeled pairs: {n} (unlabeled/skipped: {unlabeled})",
        "",
        "## Confusion (rows = router, cols = human)",
        "",
        "| router \\ human | " + " | ".join(LABELS) + " | n |",
        "|---|" + "---|" * (len(LABELS) + 1),
    ]
    for s in LABELS:
        n_s = sum(conf[s].values())
        lines.append(f"| {s} | " +
                     " | ".join(str(conf[s].get(h, 0)) for h in LABELS) +
                     f" | {n_s} |")
    lines += [
        "",
        "## Verifiable-class metrics (population-weighted)",
        "",
        f"- Precision  P(human=v | router=v): **{precision:.1%}**  "
        f"({p_v_given['verifiable'][1]}/{p_v_given['verifiable'][2]})",
        f"- Recall (weighted): **{recall:.1%}**",
        f"- Router (raw) verifiable fraction: "
        f"**{w['verifiable']:.1%}** of routed population",
        f"- **Calibrated human verifiable fraction: "
        f"{true_frac:.1%} ± {ci:.1%} (95% CI)**",
        "",
        f"- 4-class observed agreement: {po:.1%}; Cohen's kappa: {kappa:.2f} "
        f"(stratified sample — interpret as conditional agreement)",
        "",
        "Per-stratum P(human=v | router=s): " +
        ", ".join(f"{s}={p_v_given[s][0]:.1%} ({p_v_given[s][1]}/{p_v_given[s][2]})"
                  for s in LABELS),
    ]
    report = OUT_DIR / "router_validation_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
