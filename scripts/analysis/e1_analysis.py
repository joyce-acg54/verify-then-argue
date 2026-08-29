#!/usr/bin/env python3
"""E1 analysis: canary propagation by condition, cluster-aware.

Usage (from the repo root, after collect.py has produced canary_propagation.csv):
  python scripts/analysis/e1_analysis.py [--boot 10000] [--seed 42]

Inputs
  data/aligner/canary_propagation.csv   one row per (run, canary): status, da_caught
  results/e1_grid.jsonl                 decisions per (company, condition, seed)
  data/canaries/survival_analysis.json  canary -> survived-into-claims flag
  data/injected/*/injection_manifest.json  first edit offset -> C0-window visibility
  data/canaries/closed_book_results.jsonl  model-knows stratum
  data/canaries/raw/*.json              web_checkable flag

Outputs
  results/e1_propagation_by_condition.csv
  results/e1_contrasts.csv
  results/e1_analysis.md                human-readable summary

Method
  Propagation_strict = share of (run, canary) with status 'propagated'
  (falsified value asserted in a load-bearing premise). Propagation_any adds
  'propagated_nlb'. CIs: deck-level nonparametric bootstrap (decks resampled
  with replacement, B default 10000, seed 42) because canaries are nested in
  decks. Contrasts (C0-C1, C1-C2, C2-C2shuf, C2-C3) use the same deck
  resample for both arms (paired at deck level). Seeds: seed 0 only for the
  main table (repeats reported separately as run-to-run variance).
  Stratifications: C0-window visibility (canary's first edit inside the 12k
  truncation) applied to C0; survived-into-claims applied to C1+ ("reachable"
  propagation); web_checkable and model_may_know reported descriptively.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGNER = REPO_ROOT / "data" / "aligner"
RESULTS = REPO_ROOT / "results"

CONDS = ["C0", "C1", "C2", "C2shuf", "C3"]
CONTRASTS = [("C0", "C1"), ("C1", "C2"), ("C2", "C2shuf"), ("C2", "C3")]
PROP_STRICT = {"propagated"}
PROP_ANY = {"propagated", "propagated_nlb"}


def load_rows() -> list[dict]:
    path = ALIGNER / "canary_propagation.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["condition"] in CONDS]
    return rows


def load_strata() -> tuple[dict, dict, dict, dict]:
    surv = {cid: d.get("survived", False) for cid, d in
            json.loads((REPO_ROOT / "data" / "canaries" /
                        "survival_analysis.json").read_text()).items()}
    c0vis: dict[str, bool] = {}
    for mf in (REPO_ROOT / "data" / "injected").glob("*/injection_manifest.json"):
        m = json.loads(mf.read_text())
        for c in m.get("canaries", []):
            offs = [e.get("first_replacement_char_offset", 0)
                    for e in c.get("edits", [])]
            c0vis[c["canary_id"]] = bool(offs) and min(offs) < 12000
    knows: dict[str, bool] = {}
    cb = REPO_ROOT / "data" / "canaries" / "closed_book_results.jsonl"
    if cb.exists():
        for line in open(cb, encoding="utf-8"):
            r = json.loads(line)
            knows[r["canary_id"]] = r.get("knows_true_fact") is True
    webc: dict[str, bool] = {}
    for p in (REPO_ROOT / "data" / "canaries" / "raw").glob("*.json"):
        raw = json.loads(p.read_text())
        for i, c in enumerate(raw.get("canaries", [])):
            webc[f"{raw['account_id']}_{i}"] = c.get("web_checkable") in (True, "True")
    return surv, c0vis, knows, webc


def rate(rows: list[dict], statuses: set[str]) -> float:
    return (sum(1 for r in rows if r["status"] in statuses) / len(rows)
            if rows else float("nan"))


def boot_ci(rows_by_deck: dict[str, list[dict]], statuses: set[str],
            B: int, rng: random.Random) -> tuple[float, float]:
    decks = list(rows_by_deck)
    if not decks:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(B):
        sample = [rows_by_deck[rng.choice(decks)] for _ in decks]
        flat = [r for chunk in sample for r in chunk]
        if flat:
            stats.append(rate(flat, statuses))
    stats.sort()
    return (stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))])


def boot_contrast(rows_a: dict[str, list[dict]], rows_b: dict[str, list[dict]],
                  statuses: set[str], B: int, rng: random.Random
                  ) -> tuple[float, float, float]:
    """Paired deck-level bootstrap of rate(a) - rate(b)."""
    decks = sorted(set(rows_a) | set(rows_b))
    point = (rate([r for d in decks for r in rows_a.get(d, [])], statuses)
             - rate([r for d in decks for r in rows_b.get(d, [])], statuses))
    diffs = []
    for _ in range(B):
        pick = [rng.choice(decks) for _ in decks]
        fa = [r for d in pick for r in rows_a.get(d, [])]
        fb = [r for d in pick for r in rows_b.get(d, [])]
        if fa and fb:
            diffs.append(rate(fa, statuses) - rate(fb, statuses))
    diffs.sort()
    return point, diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = load_rows()
    surv, c0vis, knows, webc = load_strata()
    main_rows = [r for r in rows if str(r["seed"]) == "0"]

    def stratum_ok(r: dict) -> bool:
        cid = r["canary_id"]
        if r["condition"] == "C0":
            return c0vis.get(cid, True)
        return surv.get(cid, False)

    by_cond: dict[str, list[dict]] = defaultdict(list)
    by_cond_strat: dict[str, list[dict]] = defaultdict(list)
    for r in main_rows:
        by_cond[r["condition"]].append(r)
        if stratum_ok(r):
            by_cond_strat[r["condition"]].append(r)

    # decisions from the grid
    dec: dict[str, list[float]] = defaultdict(list)
    for line in open(RESULTS / "e1_grid.jsonl", encoding="utf-8"):
        g = json.loads(line)
        if g["seed"] == 0:
            dec[g["condition"]].append(g["p_invest"])

    out_rows = []
    md = ["# E1 — canary propagation by condition (seed 0)", ""]
    md.append("| cond | n pairs | propagated | 95% CI | +nlb | hedged | flagged "
              "| contradicted | absent | DA catch | reachable-strat. prop. "
              "| mean p(invest) |")
    md.append("|---|" + "---|" * 11)
    for c in CONDS:
        rs = by_cond[c]
        by_deck = defaultdict(list)
        for r in rs:
            by_deck[r["company_id"]].append(r)
        lo, hi = boot_ci(by_deck, PROP_STRICT, args.boot, rng)
        n = len(rs)
        shares = {s: sum(1 for r in rs if r["status"] == s) / n if n else 0
                  for s in ("propagated", "propagated_nlb", "hedged",
                            "flagged", "contradicted", "absent")}
        da = (sum(1 for r in rs if r["da_caught"] in (True, "True")) / n
              if n else float("nan"))
        strat = rate(by_cond_strat[c], PROP_STRICT)
        p_inv = statistics.mean(dec[c]) if dec[c] else float("nan")
        md.append(
            f"| {c} | {n} | {shares['propagated']:.1%} | [{lo:.1%},{hi:.1%}] "
            f"| {shares['propagated_nlb']:.1%} | {shares['hedged']:.1%} "
            f"| {shares['flagged']:.1%} | {shares['contradicted']:.1%} "
            f"| {shares['absent']:.1%} | {da:.1%} | {strat:.1%} | {p_inv:.3f} |")
        out_rows.append({"condition": c, "n": n, **{f"share_{k}": round(v, 4)
                         for k, v in shares.items()},
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "da_catch": round(da, 4),
                         "prop_reachable_stratum": round(strat, 4),
                         "mean_p_invest": round(p_inv, 4)})

    md += ["", "## Pre-registered contrasts (propagated, deck-level paired bootstrap)", ""]
    md.append("| contrast | diff | 95% CI |")
    md.append("|---|---|---|")
    contrast_rows = []
    for a, b in CONTRASTS:
        ra, rb = defaultdict(list), defaultdict(list)
        for r in by_cond[a]:
            ra[r["company_id"]].append(r)
        for r in by_cond[b]:
            rb[r["company_id"]].append(r)
        point, lo, hi = boot_contrast(ra, rb, PROP_STRICT, args.boot, rng)
        md.append(f"| {a} − {b} | {point:+.1%} | [{lo:+.1%},{hi:+.1%}] |")
        contrast_rows.append({"contrast": f"{a}-{b}", "diff": round(point, 4),
                              "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})

    # descriptive strata
    md += ["", "## Descriptive strata (propagated rate, seed 0, all conditions pooled)", ""]
    for name, flag in (("web_checkable", webc), ("model_knows_true", knows),
                       ("survived_into_claims", surv), ("c0_window_visible", c0vis)):
        t = [r for r in main_rows if flag.get(r["canary_id"], False)]
        f = [r for r in main_rows if not flag.get(r["canary_id"], False)]
        md.append(f"- {name}: true {rate(t, PROP_STRICT):.1%} (n={len(t)}) "
                  f"vs false {rate(f, PROP_STRICT):.1%} (n={len(f)})")

    # run-to-run variance from repeat seeds
    rep = [r for r in rows if str(r["seed"]) in ("0", "1", "2")]
    rep_decks = {r["company_id"] for r in rep if str(r["seed"]) != "0"}
    if rep_decks:
        md += ["", "## Run-to-run variance (5 repeat decks, seeds 0-2)", ""]
        sds = []
        for c in CONDS:
            cell = defaultdict(list)
            for r in rep:
                if r["company_id"] in rep_decks and r["condition"] == c:
                    cell[(r["company_id"], r["seed"])].append(r)
            per_seed = defaultdict(list)
            for (d, s), rs_ in cell.items():
                per_seed[s].append(rate(rs_, PROP_STRICT))
            seed_means = [statistics.mean(v) for v in per_seed.values() if v]
            if len(seed_means) >= 2:
                sds.append(statistics.stdev(seed_means))
                md.append(f"- {c}: across-seed SD of propagation "
                          f"{statistics.stdev(seed_means):.3f}")
        if sds:
            md.append(f"- max across-seed SD: {max(sds):.3f}")

    with open(RESULTS / "e1_propagation_by_condition.csv", "w", newline="",
              encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    with open(RESULTS / "e1_contrasts.csv", "w", newline="",
              encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(contrast_rows[0].keys()))
        w.writeheader()
        w.writerows(contrast_rows)
    (RESULTS / "e1_analysis.md").write_text("\n".join(md) + "\n",
                                            encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {RESULTS / 'e1_analysis.md'} + 2 CSVs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
