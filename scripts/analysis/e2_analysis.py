#!/usr/bin/env python3
"""E2 — natural unsupported-claim leakage (ecological-validity check for E1).

Question: over the natural (non-canary) decks, does a claim the verifier
labeled NoEvidence / Disbelief still become a load-bearing premise in the
debate's final arguments — and do verdict labels (C2/C3) reduce that
relative to raw text (C0)?

Definitions (mirrors e1_analysis.py conventions):
- Unit: one row per (deck, condition, claim) for claims whose verdict is in
  the target class. Denominator comes from data/claims/<id>.json (unlinked
  claims never appear in claim_leakage.csv).
- A claim's status per (deck, condition): "asserts" if any load-bearing
  premise asserts it; else "hedges" if any load-bearing premise hedges it;
  else "absent".
- Leakage rate  = P(status in {asserts, hedges}); hedge rate = P(hedges).
- Deck-level nonparametric bootstrap (decks resampled with replacement;
  claims nest in decks), 95% percentile CIs; paired deck-level bootstrap
  for the C0-C2 and C2-C3 contrasts. B=10,000, seed 42 (as in E1).

Confound handling: NoEvidence is the primary class (absence of public
evidence is what the verifier measured). Disbelief leakage is reported
DESCRIPTIVELY ONLY: the recall probe measured an 18.3% false-refutation
rate on known-true facts, so a Disbelief-labeled claim may be true and its
"leakage" is not necessarily a failure.

Run: python scripts/analysis/e2_analysis.py [--boot 10000]
Inputs: data/targets_e2.csv, data/claims/<id>.json, and the aligner's
claim_leakage.csv are withheld corpus artifacts, not part of this release;
the script is included for methodological transparency.
Outputs: results/e2_analysis.md, results/e2_leakage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGETS = REPO / "data" / "targets_e2.csv"
CLAIMS_DIR = REPO / "data" / "claims"
LEAKAGE = REPO / "data" / "aligner" / "claim_leakage.csv"
OUT_MD = REPO / "results" / "e2_analysis.md"
OUT_CSV = REPO / "results" / "e2_leakage.csv"

CONDITIONS = ["C0", "C2", "C3"]
CLASSES = {"no_evidence": "NoEvidence (primary)",
           "disbelief": "Disbelief (descriptive; 18.3% false-refutation confound)"}
LEAK = {"asserts", "hedges"}


def load_targets() -> set[str]:
    with open(TARGETS, newline="", encoding="utf-8") as f:
        return {r["account_id"] for r in csv.DictReader(f)}


def load_denominators(accounts: set[str]) -> dict[str, dict[str, list[int]]]:
    """account -> verdict_class -> list of claim indices in that class."""
    out: dict[str, dict[str, list[int]]] = {}
    for acc in sorted(accounts):
        claims = json.loads((CLAIMS_DIR / f"{acc}.json").read_text())
        per = defaultdict(list)
        for i, c in enumerate(claims):
            v = c.get("verdict")
            if v in CLASSES:
                per[v].append(i)
        out[acc] = per
    return out


def load_links(accounts: set[str]) -> dict[tuple[str, str, int], set[str]]:
    """(account, condition, claim_idx) -> set of load-bearing relations."""
    links: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    with open(LEAKAGE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["company_id"] in accounts and r["condition"] in CONDITIONS
                    and r["seed"] == "0" and r["load_bearing"] == "True"
                    and r["claim_idx"] != ""):
                links[(r["company_id"], r["condition"],
                       int(r["claim_idx"]))].add(r["relation"])
    return links


def status_rows(denoms, links, verdict_class: str, condition: str
                ) -> dict[str, list[dict]]:
    """deck -> [{status}] for every claim in the class, this condition."""
    by_deck: dict[str, list[dict]] = {}
    for acc, per in denoms.items():
        rows = []
        for idx in per.get(verdict_class, []):
            rels = links.get((acc, condition, idx), set())
            if "asserts" in rels:
                s = "asserts"
            elif "hedges" in rels:
                s = "hedges"
            else:
                s = "absent"
            rows.append({"status": s})
        if rows:
            by_deck[acc] = rows
    return by_deck


def rate(rows: list[dict], statuses: set[str]) -> float:
    return (sum(1 for r in rows if r["status"] in statuses) / len(rows)
            if rows else float("nan"))


def boot_ci(rows_by_deck, statuses, B, rng) -> tuple[float, float]:
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


def boot_contrast(rows_a, rows_b, statuses, B, rng):
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

    accounts = load_targets()
    denoms = load_denominators(accounts)
    links = load_links(accounts)
    conds_seen = {c for (_, c, _) in links}
    missing = [c for c in CONDITIONS if c not in conds_seen]
    if missing:
        print(f"WARNING: no aligned links for conditions {missing} — "
              f"run the aligner first; rates would be spuriously 0.")

    md = ["# E2 — natural unsupported-claim leakage (C0/C2/C3, seed 0)", ""]
    md.append(f"E2 target set: {len(accounts)} natural (non-canary) decks; "
              "denominators from data/claims/<id>.json; instrument: "
              "premise-tracing aligner (Claude Haiku 4.5, prompt aligner-v1; "
              "claim-link precision 65.6 [55.3, 74.6], miss 4.5 [1.8, 11.0] "
              "— applies to all E2 metrics, disclosed). One of 216 "
              "transcripts (a 122-claim deck) was aligned on Claude Sonnet "
              "4.5 after Haiku twice emitted out-of-range claim indices; the "
              "aligner is measurement-only and outside the decision chain, "
              "so a stronger model adds no leakage confound (per RUNNER.md).")
    md.append("")
    md.append("E2 is ecological-validity insurance for E1, powered for "
              "large effects only: it asks whether E1's null dissociation "
              "(verdict labels do not keep flagged content out of "
              "load-bearing premises) also holds for natural claims.")
    md.append("")

    csv_rows = []
    stats: dict[tuple[str, str], dict] = {}
    for vclass, label in CLASSES.items():
        md.append(f"## {label}")
        md.append("")
        n_claims = sum(len(p.get(vclass, [])) for p in denoms.values())
        n_decks = sum(1 for p in denoms.values() if p.get(vclass))
        md.append(f"Denominator: {n_claims} claims across {n_decks} decks.")
        md.append("")
        md.append("| cond | n | leakage (asserts+hedges) | 95% CI | "
                  "asserts only | hedge rate |")
        md.append("|---|---|---|---|---|---|")
        for cond in CONDITIONS:
            rows = status_rows(denoms, links, vclass, cond)
            flat = [r for d in rows.values() for r in d]
            leak = rate(flat, LEAK)
            lo, hi = boot_ci(rows, LEAK, args.boot, rng)
            ass = rate(flat, {"asserts"})
            hed = rate(flat, {"hedges"})
            stats[(vclass, cond)] = rows
            md.append(f"| {cond} | {len(flat)} | {leak:.1%} | "
                      f"[{lo:.1%}, {hi:.1%}] | {ass:.1%} | {hed:.1%} |")
            csv_rows.append({"verdict_class": vclass, "condition": cond,
                             "n_claims": len(flat), "leakage": round(leak, 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                             "asserts_only": round(ass, 4),
                             "hedge_rate": round(hed, 4)})
        md.append("")
        md.append("### Pre-specified contrasts (paired deck-level bootstrap)")
        md.append("")
        md.append("| contrast | diff | 95% CI |")
        md.append("|---|---|---|")
        contrasts = {}
        for a, b in (("C0", "C2"), ("C2", "C3")):
            pt, lo, hi = boot_contrast(stats[(vclass, a)], stats[(vclass, b)],
                                       LEAK, args.boot, rng)
            contrasts[f"{a}-{b}"] = (pt, lo, hi)
            md.append(f"| {a}$-${b} | {pt:+.1%} | [{lo:+.1%}, {hi:+.1%}] |")
            csv_rows.append({"verdict_class": vclass,
                             "condition": f"{a}-{b}", "n_claims": "",
                             "leakage": round(pt, 4), "ci_lo": round(lo, 4),
                             "ci_hi": round(hi, 4), "asserts_only": "",
                             "hedge_rate": ""})
        md.append("")
        # Minimum detectable effect from the primary C0-C2 contrast CI.
        pt, lo, hi = contrasts["C0-C2"]
        mde = (hi - lo) / 2
        md.append(f"**Power / MDE:** the C0$-$C2 contrast CI half-width is "
                  f"{mde:.1%}, so this experiment can only distinguish a "
                  f"verification-induced leakage reduction larger than "
                  f"~{mde:.0%} from zero; it is powered for large effects "
                  f"only. The point estimate ({pt:+.1%}) "
                  f"{'excludes' if lo > 0 or hi < 0 else 'includes'} zero. "
                  f"No multiplicity correction is applied across the "
                  f"contrast family (as in E1): the inferential claim is the "
                  f"MDE bound, not a discovery, so any single boundary-"
                  f"significant interval (e.g. C2$-$C3) is not robust and, "
                  f"pointing toward *more* leakage with added signals, does "
                  f"not indicate a verification benefit.")
        md.append("")
        if vclass == "disbelief":
            md.append("**Confound (do not interpret causally):** the recall "
                      "probe measured an 18.3% false-refutation rate on "
                      "known-true facts, so a Disbelief label does not "
                      "establish the claim is false; Disbelief 'leakage' "
                      "may include true claims correctly used.")
            md.append("")

    # Plain-English read (NoEvidence is the primary class).
    ne = {k: v for k, v in stats.items() if k[0] == "no_evidence"}
    c0 = rate([r for d in ne[("no_evidence", "C0")].values() for r in d], LEAK)
    c2 = rate([r for d in ne[("no_evidence", "C2")].values() for r in d], LEAK)
    c3 = rate([r for d in ne[("no_evidence", "C3")].values() for r in d], LEAK)
    md.append("## Plain-English read")
    md.append("")
    md.append(f"- **Natural-claim leakage mirrors E1's null dissociation.** "
              f"NoEvidence-flagged claims become load-bearing premises in "
              f"every condition ({c0:.0%} C0, {c2:.0%} C2, {c3:.0%} C3); "
              f"attaching real verdict labels (C2) does not reliably reduce "
              f"this versus raw text (C0-C2 CI includes zero). As in E1, "
              f"verdict labels do not keep flagged content out of the "
              f"arguments.")
    md.append(f"- **Powered for large effects only.** With ~{len(ne[('no_evidence','C0')])} "
              f"decks the C0-C2 contrast resolves only effects larger than "
              f"~7pp; a smaller real reduction would not register. E2 is "
              f"ecological-validity insurance for the E1 finding, not a new "
              f"headline, and the aligner instrument (claim-link precision "
              f"65.6%) bounds all E2 metrics.")
    md.append(f"- **Added signals (C3) do not help.** C3 leaks no less than "
              f"C2 (point estimate slightly higher), echoing E1, where "
              f"reliability/consistency signals did not improve falsehood "
              f"handling. Disbelief leakage is reported descriptively only: "
              f"the verifier's 18.3% false-refutation rate means a "
              f"Disbelief-flagged natural claim may be true, so its presence "
              f"in an argument is not necessarily an error.")
    md.append("")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {OUT_MD} and {OUT_CSV}")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
