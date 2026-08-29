#!/usr/bin/env python3
"""Validate aligner results, join blinding manifest, compute E1/E2 tables.

Usage (from the repo root):
  python scripts/aligner/collect.py            # validate + write tables
  python scripts/aligner/collect.py --strict   # nonzero exit on any invalid/missing

Reads  data/aligner/tasks/<task_id>.json   (what the aligner saw)
       data/aligner/raw/<task_id>.json     (what the aligner returned)
       data/aligner/tasks_manifest.csv     (blinding join)
       data/claims/<company_id>.json       (verdicts, joined HERE, post-hoc)

Writes data/aligner/alignments.jsonl       one line per task, unblinded
       data/aligner/canary_propagation.csv E1 table: one row per (run, canary)
       data/aligner/claim_leakage.csv      E2 table: one row per (run, premise-claim link)
       prints a validation report; invalid/missing tasks are listed for re-run.

Propagation status per (run, canary), priority order:
  propagated     some load-bearing premise asserts_falsified
  propagated_nlb a non-load-bearing premise asserts_falsified
  hedged         best link is hedged
  flagged        best link is flagged
  contradicted   best link is asserts_true (model/evidence pushed back)
  absent         no premise links to the canary
Devil's-advocate catches are reported in their own column (da_caught).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

STATUS_PRIORITY = ["propagated", "propagated_nlb", "hedged", "flagged",
                   "contradicted", "absent"]


def canary_status(links: list[tuple[str, bool]]) -> str:
    """links: [(relation, load_bearing), ...] for one canary across premises."""
    have = set()
    for rel, lb in links:
        if rel == "asserts_falsified":
            have.add("propagated" if lb else "propagated_nlb")
        elif rel == "hedged":
            have.add("hedged")
        elif rel == "flagged":
            have.add("flagged")
        elif rel == "asserts_true":
            have.add("contradicted")
    for s in STATUS_PRIORITY:
        if s in have:
            return s
    return "absent"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    manifest = common.load_manifest()
    if not manifest:
        print("ERROR: empty manifest — run prepare_tasks.py first",
              file=sys.stderr)
        return 1

    missing, invalid, ok = [], [], []
    align_path = common.ALIGNER_DIR / "alignments.jsonl"
    prop_path = common.ALIGNER_DIR / "canary_propagation.csv"
    leak_path = common.ALIGNER_DIR / "claim_leakage.csv"

    verdict_cache: dict[str, list[dict]] = {}

    def verdicts_for(company_id: str) -> list[dict]:
        if company_id not in verdict_cache:
            p = common.CLAIMS_DIR / f"{company_id}.json"
            verdict_cache[company_id] = (
                json.loads(p.read_text(encoding="utf-8")) if p.is_file() else [])
        return verdict_cache[company_id]

    prop_rows: list[dict] = []
    leak_rows: list[dict] = []

    with open(align_path, "w", encoding="utf-8") as out:
        for tid, meta in sorted(manifest.items()):
            tpath, rpath = common.task_path(tid), common.result_path(tid)
            if not tpath.is_file():
                missing.append((tid, "task file missing"))
                continue
            if not rpath.is_file():
                missing.append((tid, "result missing"))
                continue
            task = common.load_json(tpath)
            try:
                res = common.load_json(rpath)
            except json.JSONDecodeError as e:
                invalid.append((tid, f"result not JSON: {e}"))
                continue
            errs = common.validate_result(res, task)
            if errs:
                invalid.append((tid, "; ".join(errs[:4])))
                continue
            ok.append(tid)

            company = meta["company_id"]
            rec = {
                "task_id": tid,
                "company_id": company,
                "condition": meta["condition"],
                "seed": meta["seed"],
                "prompt_version": common.PROMPT_VERSION,
                "result": res,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # --- E1: canary propagation ---
            da = {c["canary_id"]: c for c in
                  res.get("canary_critique_catches", [])}
            links_by_canary: dict[str, list[tuple[str, bool]]] = {
                c["id"]: [] for c in task["canaries"]}
            for a in res["arguments"]:
                for p in a["premises"]:
                    for link in p.get("canary_links", []):
                        links_by_canary.setdefault(
                            link["canary_id"], []).append(
                            (link["relation"], bool(p.get("load_bearing"))))
            cmeta = {c["id"]: c for c in task["canaries"]}
            for cid, links in links_by_canary.items():
                prop_rows.append({
                    "company_id": company,
                    "condition": meta["condition"],
                    "seed": meta["seed"],
                    "canary_id": cid,
                    "fact_type": cmeta.get(cid, {}).get("fact_type", ""),
                    "status": canary_status(links),
                    "n_links": len(links),
                    "da_caught": bool(da.get(cid, {}).get("caught")),
                })

            # --- E2: premise-claim links joined with verdicts ---
            verdicts = verdicts_for(company)
            for a in res["arguments"]:
                for j, p in enumerate(a["premises"]):
                    for link in p.get("claim_links", []):
                        ci = link["claim_idx"]
                        v = verdicts[ci] if ci < len(verdicts) else {}
                        leak_rows.append({
                            "company_id": company,
                            "condition": meta["condition"],
                            "seed": meta["seed"],
                            "arg_idx": a["idx"],
                            "premise_idx": j,
                            "load_bearing": bool(p.get("load_bearing")),
                            "relation": link["relation"],
                            "claim_idx": ci,
                            "verdict": v.get("verdict"),
                            "routing": v.get("routing"),
                            "source_reliability": v.get("source_reliability"),
                            "consistency": v.get("consistency"),
                        })

    for path, rows in ((prop_path, prop_rows), (leak_path, leak_rows)):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print(f"valid: {len(ok)} | missing: {len(missing)} | invalid: {len(invalid)}")
    for tid, why in missing + invalid:
        print(f"  RERUN {tid}: {why}")
    print(f"wrote {align_path} ({len(ok)} tasks), "
          f"{prop_path} ({len(prop_rows)} rows), "
          f"{leak_path} ({len(leak_rows)} rows)")
    if args.strict and (missing or invalid):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
