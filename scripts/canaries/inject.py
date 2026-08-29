#!/usr/bin/env python
"""Inject falsified canary spans into parsed deck texts.

Usage (from the repo root):
  python scripts/canaries/inject.py --accounts all
  python scripts/canaries/inject.py \
      --accounts ACCOUNT_ID_1,ACCOUNT_ID_2
  python scripts/canaries/inject.py --accounts all \
      --approvals data/canaries/approvals.csv

Inputs (READ-ONLY, never modified):
  data/canaries/raw/<account_id>.json   canary definitions
  <deck_txt_path>                       parsed deck text referenced therein

Outputs per account, under data/injected/<account_id>/:
  deck_original.txt        unmodified copy of the parsed deck text — the
                           C0-control twin (experiments run on the injected
                           deck; the original is kept for diffing)
  deck_injected.txt        deck text with all selected canary edits applied
  injection_manifest.json  per-canary edit accounting + sha256 of both texts

Selection: qc_status != "dropped" AND, if an approvals file is given,
approve == "y" for the canary's canary_id. canary_id convention matches
data/canaries/review_sheet.csv: "<account_id>_<i>" where i is the 0-based
position in the raw file's "canaries" list (dropped canaries keep their
index). Approval column values:
  y / yes     inject
  n / no      exclude
  edit:...    exclude with a LOUD warning — resolve the requested edit in
              data/canaries/raw/ first, then flip the row to "y"
  (missing)   exclude with a warning

Without --approvals the script runs in SMOKE MODE: every non-dropped canary
is treated as approved and a warning is printed. If the canonical approvals
file (data/canaries/approvals.csv) already exists, you must either pass
--approvals or opt out explicitly with --ignore-approvals.

Hard failure: if any edit's "find" string does not occur at least once in
the current deck text, the script prints a detailed error and exits
non-zero WITHOUT writing any output for that account. No silent skips.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RAW_DIR = REPO_ROOT / "data" / "canaries" / "raw"
OUT_ROOT = REPO_ROOT / "data" / "injected"
CANONICAL_APPROVALS = REPO_ROOT / "data" / "canaries" / "approvals.csv"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def parse_accounts(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return sorted(p.stem for p in RAW_DIR.glob("*.json"))
    ids = [a.strip() for a in spec.split(",") if a.strip()]
    missing = [a for a in ids if not (RAW_DIR / f"{a}.json").is_file()]
    if missing:
        die(f"no raw canary file under {RAW_DIR} for account(s): {missing}")
    return ids


def load_approvals(path: Path) -> dict[str, str]:
    """canary_id -> normalized approve value (lowercased, stripped)."""
    if not path.is_file():
        die(f"approvals file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not {"canary_id", "approve"} <= cols:
            die(f"approvals file {path} must have columns canary_id,approve "
                f"(found: {sorted(cols)})")
        approvals: dict[str, str] = {}
        for row in reader:
            cid = (row.get("canary_id") or "").strip()
            if not cid:
                continue
            if cid in approvals:
                warn(f"approvals: duplicate canary_id {cid!r}; keeping last row")
            approvals[cid] = (row.get("approve") or "").strip().lower()
    return approvals


def select_canaries(
    account_id: str,
    canaries: list[dict],
    approvals: dict[str, str] | None,
) -> tuple[list[tuple[str, int, dict]], list[dict]]:
    """Returns (selected [(canary_id, index, canary)], excluded [{...}])."""
    selected: list[tuple[str, int, dict]] = []
    excluded: list[dict] = []
    for i, canary in enumerate(canaries):
        cid = f"{account_id}_{i}"
        status = canary.get("qc_status", "")
        if status == "dropped":
            excluded.append({"canary_id": cid, "reason": "qc_status=dropped"})
            continue
        if approvals is not None:
            val = approvals.get(cid)
            if val is None:
                warn(f"{cid}: not present in approvals file -> EXCLUDED")
                excluded.append({"canary_id": cid, "reason": "missing_from_approvals"})
                continue
            if val.startswith("edit:"):
                warn(f"{cid}: approvals says {val!r} -> EXCLUDED. Resolve the "
                     f"edit in data/canaries/raw/{account_id}.json, then mark "
                     f"the row 'y' and re-run.")
                excluded.append({"canary_id": cid, "reason": f"approve={val}"})
                continue
            if val in ("y", "yes"):
                selected.append((cid, i, canary))
                continue
            if val not in ("n", "no"):
                warn(f"{cid}: unrecognized approve value {val!r} -> EXCLUDED "
                     f"(treated as not approved)")
            excluded.append({"canary_id": cid, "reason": f"approve={val or 'BLANK'}"})
            continue
        selected.append((cid, i, canary))
    return selected, excluded


def apply_canary_edits(
    text: str, cid: str, canary: dict, deck_path: str
) -> tuple[str, list[dict], int]:
    """Apply one canary's edits sequentially; replace ALL occurrences of each
    find. Hard-fails if a find string is absent. Returns
    (new_text, per_edit_records, n_replacements_total)."""
    records: list[dict] = []
    n_total = 0
    edits = canary.get("edits") or []
    if not edits:
        die(f"{cid}: selected canary has no edits")
    for j, edit in enumerate(edits):
        find, replace = edit.get("find"), edit.get("replace")
        if not find or replace is None:
            die(f"{cid} edit[{j}]: malformed edit (find={find!r})")
        if find == replace:
            die(f"{cid} edit[{j}]: find == replace ({find!r})")
        n = text.count(find)
        if n == 0:
            die(
                f"{cid} edit[{j}]: find string NOT FOUND in current deck text.\n"
                f"  deck: {deck_path}\n"
                f"  find ({len(find)} chars): {find!r}\n"
                f"  Possible causes: deck re-parsed since canary generation, or "
                f"an earlier edit in this run consumed the span. "
                f"No output was written for this account."
            )
        char_off = text.index(find)
        byte_off = len(text[:char_off].encode("utf-8"))
        text = text.replace(find, replace)
        n_total += n
        records.append({
            "edit_index": j,
            "find": find,
            "replace": replace,
            "n_replacements": n,
            "first_replacement_char_offset": char_off,
            "first_replacement_byte_offset": byte_off,
        })
    return text, records, n_total


def cross_edit_warning(all_selected: list[tuple[str, int, dict]]) -> None:
    """Warn if a later canary's find string appears inside an earlier
    canary's replace string (possible double-application hazard)."""
    seen: list[tuple[str, str]] = []  # (cid, replace)
    for cid, _i, canary in all_selected:
        for edit in canary.get("edits") or []:
            for prev_cid, prev_replace in seen:
                if edit["find"] in prev_replace:
                    warn(f"{cid}: find {edit['find']!r} occurs inside the "
                         f"replace text of earlier canary {prev_cid}; check "
                         f"the manifest replacement counts carefully")
            seen.append((cid, edit.get("replace", "")))


def inject_account(account_id: str, approvals: dict[str, str] | None,
                   approvals_path: Path | None) -> dict | None:
    raw_path = RAW_DIR / f"{account_id}.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    company = raw.get("company", "")
    deck_path = Path(raw["deck_txt_path"])
    if not deck_path.is_file():
        die(f"{account_id}: deck text not found: {deck_path}")
    # Mirror the project read convention (scripts/verify/common.py).
    original = deck_path.read_text(encoding="utf-8", errors="replace")

    selected, excluded = select_canaries(account_id, raw.get("canaries") or [],
                                         approvals)
    if not selected:
        warn(f"{account_id} ({company}): no canaries selected "
             f"({len(excluded)} excluded) -> nothing written")
        return None

    cross_edit_warning(selected)

    text = original
    canary_records: list[dict] = []
    total_repl = 0
    for cid, _i, canary in selected:
        text, edit_records, n_repl = apply_canary_edits(
            text, cid, canary, str(deck_path))
        total_repl += n_repl
        canary_records.append({
            "canary_id": cid,
            "fact_type": canary.get("fact_type"),
            "qc_status": canary.get("qc_status"),
            "n_edits": len(edit_records),
            "n_replacements_made": n_repl,
            "edits": edit_records,
        })

    if text == original:
        die(f"{account_id}: injected text identical to original despite "
            f"{total_repl} replacement(s) — should be impossible")

    out_dir = OUT_ROOT / account_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "account_id": account_id,
        "company": company,
        "deck_txt_path": str(deck_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approvals_file": str(approvals_path) if approvals_path else None,
        "approvals_mode": ("approvals_csv" if approvals_path
                           else "SMOKE: all non-dropped treated as approved"),
        "sha256_original": sha256_text(original),
        "sha256_injected": sha256_text(text),
        "original_chars": len(original),
        "injected_chars": len(text),
        "n_canaries_in_raw": len(raw.get("canaries") or []),
        "n_selected": len(selected),
        "n_replacements_total": total_repl,
        "excluded": excluded,
        "canaries": canary_records,
        "offset_note": ("Edits are applied sequentially in raw-file order; "
                        "offsets are UTF-8 byte / unicode char offsets into "
                        "the text state immediately before that edit."),
    }
    (out_dir / "deck_original.txt").write_text(original, encoding="utf-8")
    (out_dir / "deck_injected.txt").write_text(text, encoding="utf-8")
    (out_dir / "injection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # Read-back self-check.
    rb = (out_dir / "deck_injected.txt").read_text(encoding="utf-8")
    if sha256_text(rb) != manifest["sha256_injected"]:
        die(f"{account_id}: read-back sha mismatch on deck_injected.txt")

    print(f"[{account_id}] {company}: {len(selected)} canaries, "
          f"{sum(c['n_edits'] for c in canary_records)} edits, "
          f"{total_repl} replacements | sha256 orig={manifest['sha256_original'][:12]} "
          f"inj={manifest['sha256_injected'][:12]} -> {out_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accounts", required=True,
                    help="comma-separated account ids, or 'all'")
    ap.add_argument("--approvals", type=Path, default=None,
                    help="approvals CSV (canary_id,approve). Omit for smoke mode.")
    ap.add_argument("--ignore-approvals", action="store_true",
                    help="run in smoke mode even though the canonical approvals "
                         "file exists")
    args = ap.parse_args()

    approvals: dict[str, str] | None = None
    if args.approvals is not None:
        approvals = load_approvals(args.approvals)
        print(f"Approvals: {len(approvals)} rows from {args.approvals}")
    else:
        if CANONICAL_APPROVALS.is_file() and not args.ignore_approvals:
            die(f"{CANONICAL_APPROVALS} exists but --approvals was not given. "
                f"Pass --approvals {CANONICAL_APPROVALS.relative_to(REPO_ROOT)} "
                f"or use --ignore-approvals to force smoke mode.")
        warn("no approvals file given — SMOKE MODE: treating ALL non-dropped "
             "canaries as approved")

    accounts = parse_accounts(args.accounts)
    print(f"Injecting {len(accounts)} account(s)...")
    n_written = 0
    for account_id in accounts:
        if inject_account(account_id, approvals, args.approvals) is not None:
            n_written += 1
    print(f"Done: wrote injected decks for {n_written}/{len(accounts)} account(s) "
          f"under {OUT_ROOT}")


if __name__ == "__main__":
    main()
