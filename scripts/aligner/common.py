"""Shared paths + result validation for the premise-tracing aligner.

The aligner is a MEASUREMENT instrument (outside the decision chain): it
decomposes debate arguments into atomic premises and aligns each premise to
(a) extracted deck claims and (b) planted canaries. It is executed by Claude
Haiku 4.5 against the versioned prompt `aligner-v1` (see RUNNER.md) — these
modules only prepare tasks, validate results, and compute metrics. No API
calls here.

Blinding: task_id is an opaque hash; the (company, condition, seed) mapping
lives ONLY in tasks_manifest.csv, which is never shown to the aligner model
or to human validators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Output namespace is env-overridable so a different-model arm (e.g. the gpt-4o
# robustness run) gets its own tasks/raw/manifest instead of colliding with the
# published-config aligner data (task_id omits the model).
ALIGNER_DIR = REPO_ROOT / "data" / os.environ.get("ALIGNER_NS", "aligner")
TASKS_DIR = ALIGNER_DIR / "tasks"
RESULTS_DIR = ALIGNER_DIR / "raw"
MANIFEST_PATH = ALIGNER_DIR / "tasks_manifest.csv"
CLAIMS_DIR = REPO_ROOT / "data" / "claims"
CANARY_RAW_DIR = REPO_ROOT / "data" / "canaries" / "raw"
INJECTED_DIR = REPO_ROOT / "data" / "injected"

CLAIM_RELATIONS = {"asserts", "hedges", "contradicts"}
CANARY_RELATIONS = {"asserts_falsified", "asserts_true", "hedged", "flagged"}

PROMPT_VERSION = "aligner-v1"


def task_id_for(company_id: str, condition: str, seed) -> str:
    return hashlib.sha1(
        f"{company_id}|{condition}|{seed}".encode("utf-8")).hexdigest()[:12]


def task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def result_path(task_id: str) -> Path:
    return RESULTS_DIR / f"{task_id}.json"


def load_manifest() -> dict[str, dict]:
    """task_id -> {company_id, condition, seed, results_file}."""
    if not MANIFEST_PATH.is_file():
        return {}
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return {r["task_id"]: r for r in csv.DictReader(f)}


def write_manifest(rows: dict[str, dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cols = ["task_id", "company_id", "condition", "seed", "results_file"]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for tid in sorted(rows):
            w.writerow({c: rows[tid].get(c, "") for c in cols})


def validate_result(res: dict, task: dict) -> list[str]:
    """Returns a list of problems; empty list = valid."""
    errs: list[str] = []
    if res.get("task_id") != task["task_id"]:
        errs.append(f"task_id mismatch: {res.get('task_id')!r}")
    n_args = len(task["arguments"])
    claim_idxs = {c["idx"] for c in task["claims"]}
    canary_ids = {c["id"] for c in task["canaries"]}

    args_out = res.get("arguments")
    if not isinstance(args_out, list) or len(args_out) != n_args:
        errs.append(f"arguments: expected list of {n_args}, "
                    f"got {type(args_out).__name__}"
                    f"[{len(args_out) if isinstance(args_out, list) else '?'}]")
        return errs
    for a in args_out:
        idx = a.get("idx")
        if not isinstance(idx, int) or not 0 <= idx < n_args:
            errs.append(f"bad argument idx {idx!r}")
            continue
        prems = a.get("premises")
        if not isinstance(prems, list) or not prems:
            errs.append(f"arg {idx}: premises missing/empty")
            continue
        for j, p in enumerate(prems):
            if not isinstance(p.get("premise"), str) or not p["premise"].strip():
                errs.append(f"arg {idx} premise {j}: empty text")
            if not isinstance(p.get("load_bearing"), bool):
                errs.append(f"arg {idx} premise {j}: load_bearing not bool")
            for link in p.get("claim_links", []):
                if link.get("claim_idx") not in claim_idxs:
                    errs.append(f"arg {idx} premise {j}: unknown claim_idx "
                                f"{link.get('claim_idx')!r}")
                if link.get("relation") not in CLAIM_RELATIONS:
                    errs.append(f"arg {idx} premise {j}: bad claim relation "
                                f"{link.get('relation')!r}")
            for link in p.get("canary_links", []):
                if link.get("canary_id") not in canary_ids:
                    errs.append(f"arg {idx} premise {j}: unknown canary_id "
                                f"{link.get('canary_id')!r}")
                if link.get("relation") not in CANARY_RELATIONS:
                    errs.append(f"arg {idx} premise {j}: bad canary relation "
                                f"{link.get('relation')!r}")
    catches = res.get("canary_critique_catches", [])
    if not isinstance(catches, list):
        errs.append("canary_critique_catches not a list")
    else:
        for c in catches:
            if c.get("canary_id") not in canary_ids:
                errs.append(f"critique catch: unknown canary_id "
                            f"{c.get('canary_id')!r}")
            if not isinstance(c.get("caught"), bool):
                errs.append(f"critique catch {c.get('canary_id')}: "
                            f"caught not bool")
    if task["canaries"] and not isinstance(
            res.get("canary_critique_catches"), list):
        errs.append("canary deck but canary_critique_catches missing")
    return errs


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
