#!/usr/bin/env python3
"""Build blinded aligner task files from debate results.

Usage (from the repo root):
  python scripts/aligner/prepare_tasks.py --results results/smoke_test.jsonl
  python scripts/aligner/prepare_tasks.py --results results/debate_runs.jsonl [--limit N] [--force]

For every debate record (one JSONL line = one (company, condition, seed) run)
this writes data/aligner/tasks/<task_id>.json containing the final arguments,
flattened critique items, the company's extracted claim TEXTS (verdict-blind),
and its canary spans (when the deck is a canary deck). task_id is an opaque
hash; the task_id -> (company, condition, seed) join lives only in
data/aligner/tasks_manifest.csv (never shown to the aligner model or to human
validators).

Canary candidates come from data/injected/<id>/injection_manifest.json when
present (exactly the canaries that were injected), else fall back to all
non-dropped canaries in data/canaries/raw/<id>.json.

Existing task files are skipped unless --force. The manifest is merge-updated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def load_claims(company_id: str) -> list[dict]:
    path = common.CLAIMS_DIR / f"{company_id}.json"
    if not path.is_file():
        print(f"  WARN {company_id}: no claims file at {path} -> claims=[]",
              file=sys.stderr)
        return []
    claims = json.loads(path.read_text(encoding="utf-8"))
    # Verdict-blind: claim texts only.
    return [{"idx": i, "text": c["claim"]} for i, c in enumerate(claims)]


def load_canaries(company_id: str) -> list[dict]:
    raw_path = common.CANARY_RAW_DIR / f"{company_id}.json"
    if not raw_path.is_file():
        return []
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    selected_ids: set[str] | None = None
    mani_path = common.INJECTED_DIR / company_id / "injection_manifest.json"
    if mani_path.is_file():
        mani = json.loads(mani_path.read_text(encoding="utf-8"))
        selected_ids = {c["canary_id"] for c in mani.get("canaries", [])
                        if "canary_id" in c}
    out = []
    for i, c in enumerate(raw.get("canaries", [])):
        cid = f"{company_id}_{i}"
        if c.get("qc_status") == "dropped":
            continue
        if selected_ids is not None and cid not in selected_ids:
            continue
        out.append({
            "id": cid,
            "fact_type": c.get("fact_type", ""),
            "true_span": c.get("original_span", ""),
            "falsified_span": c.get("falsified_span", ""),
        })
    return out


def flatten_critiques(rec: dict) -> list[dict]:
    items = []
    for crit in rec.get("critiques", []):
        for it in crit.get("items", []):
            items.append({
                "side": it.get("type", ""),
                "argument_excerpt": (it.get("argument") or "")[:200],
                "text": it.get("critique", ""),
            })
    return items


def build_task(rec: dict, results_file: str) -> tuple[str, dict, dict]:
    company_id = rec["company_id"]
    tid = common.task_id_for(company_id, rec["condition"], rec.get("seed"))
    task = {
        "task_id": tid,
        "arguments": [
            {"idx": i, "side": a.get("type", ""), "text": a["text"]}
            for i, a in enumerate(rec.get("arguments_final", []))
        ],
        "critiques": flatten_critiques(rec),
        "claims": load_claims(company_id),
        "canaries": load_canaries(company_id),
    }
    mani_row = {
        "task_id": tid,
        "company_id": company_id,
        "condition": rec["condition"],
        "seed": str(rec.get("seed", "")),
        "results_file": results_file,
    }
    return tid, task, mani_row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="debate results JSONL (one record per run)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="rewrite task files that already exist")
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"ERROR: no such file: {results_path}", file=sys.stderr)
        return 1

    common.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = common.load_manifest()

    n_new = n_skip = n_noargs = 0
    with open(results_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]

    for rec in records:
        if not rec.get("arguments_final"):
            n_noargs += 1
            continue
        tid, task, mani_row = build_task(rec, str(results_path))
        path = common.task_path(tid)
        if path.exists() and not args.force:
            n_skip += 1
            manifest[tid] = mani_row
            continue
        path.write_text(json.dumps(task, indent=1, ensure_ascii=False),
                        encoding="utf-8")
        manifest[tid] = mani_row
        n_new += 1
        print(f"  {tid}  args={len(task['arguments'])} "
              f"claims={len(task['claims'])} canaries={len(task['canaries'])}")

    common.write_manifest(manifest)

    # Rebuild the task index (used by the workflow runner to generate
    # per-task structured-output schemas with exact cardinalities).
    index = {}
    for tp in sorted(common.TASKS_DIR.glob("*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        index[t["task_id"]] = {
            "n_args": len(t["arguments"]),
            "n_claims": len(t["claims"]),
            "canary_ids": [c["id"] for c in t["canaries"]],
        }
    index_path = common.ALIGNER_DIR / "tasks_index.json"
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")

    print(f"tasks: {n_new} written, {n_skip} existing, {n_noargs} skipped "
          f"(no arguments) | manifest: {len(manifest)} rows -> "
          f"{common.MANIFEST_PATH} | index: {len(index)} -> {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
