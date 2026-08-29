#!/usr/bin/env python3
"""Full-corpus verification driver: extract -> verify -> claims file, per account.

Usage (from the repo root):
  python scripts/verify/run_all.py                  # non-canary usable decks
  python scripts/verify/run_all.py --only-canary    # canary accounts (post-injection)
  python scripts/verify/run_all.py --accounts <id>,<id>  [--parallel 3]

Account universe: every dir under data/documents/ with a usable parsed deck
(common.pick_deck). Accounts with a canary file under data/canaries/raw/ are
EXCLUDED by default: their claims must be extracted from the injected deck
text (data/injected/<id>/deck_injected.txt) so planted falsehoods enter the
claim stream — run them with --only-canary after inject.py has produced the
injected decks.

Per account, stages run sequentially (each is itself resumable / cached):
  1. extract_claims.py    skipped if data/cache/verify/<id>_claims_raw.json exists
  2. verify_claims.py     --search-before-date <doc_received_date MM/DD/YYYY>
                          (per-deck retrieval cutoff, E3 leakage control);
                          appends to <id>_verify.jsonl, skips done claim hashes
  3. make_claims_file.py  rebuilds data/claims/<id>.json

doc_received_date is looked up in targets_scale.csv, then targets_pilot.csv,
then targets_full.csv, then targets_synthetic.csv (the internal target CSVs
are not shipped); a missing date aborts that account (no silent uncapped
verification).

Accounts run --parallel at a time (default 3) as subprocesses of the tested
CLIs. A per-account failure is logged and does not stop the batch. Progress
goes to stdout and data/verify_run_all.log.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402

PY = sys.executable
LOG_PATH = REPO_ROOT / "data" / "verify_run_all.log"
CANARY_RAW = REPO_ROOT / "data" / "canaries" / "raw"
TARGET_FILES = ("targets_scale.csv", "targets_pilot.csv", "targets_full.csv",
                "targets_synthetic.csv")


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_dates() -> dict[str, str]:
    """account_id -> doc_received_date (MM/DD/YYYY), first file wins."""
    dates: dict[str, str] = {}
    for name in TARGET_FILES:
        path = REPO_ROOT / "data" / name
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                acc = row.get("account_id", "").strip()
                raw = (row.get("doc_received_date") or "").strip()
                if not acc or acc in dates or not raw:
                    continue
                try:
                    dates[acc] = datetime.strptime(raw, "%Y-%m-%d").strftime("%m/%d/%Y")
                except ValueError:
                    pass
    return dates


def usable_accounts() -> list[str]:
    docs = REPO_ROOT / "data" / "documents"
    return sorted(d.name for d in docs.iterdir()
                  if d.is_dir() and common.pick_deck(d.name))


# verify_claims.py exit code meaning "API pool looks dead — stop the batch".
VERIFY_ABORT_CODE = 2


class CircuitBreakerTripped(Exception):
    """A stage hit the dead-pool circuit breaker; abort the whole batch."""


def run_stage(account: str, label: str, cmd: list[str]) -> None:
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-8:])
        if label == "verify" and proc.returncode == VERIFY_ABORT_CODE:
            raise CircuitBreakerTripped(
                f"{account}: verify circuit breaker tripped ({dt:.0f}s):\n{tail}")
        raise RuntimeError(f"{label} failed (exit {proc.returncode}, {dt:.0f}s):\n{tail}")
    log(f"[{account}] {label} ok ({dt:.0f}s)")


def process_account(account: str, cutoff: str, runs: int, workers: int,
                    injected: bool = False) -> str:
    raw_path = common.claims_raw_path(account)
    if raw_path.exists():
        if injected:
            src = json.loads(raw_path.read_text(encoding="utf-8")).get("source_file")
            if src != "deck_injected.txt":
                raise RuntimeError(
                    f"claims_raw exists but was extracted from {src!r}, not the "
                    f"injected deck. Delete {raw_path} (and {account}_verify.jsonl) "
                    f"to re-extract from data/injected/{account}/deck_injected.txt.")
        log(f"[{account}] extract: cached")
    else:
        cmd = [PY, "scripts/verify/extract_claims.py", "--account", account]
        if injected:
            cmd += ["--deck-path",
                    str(REPO_ROOT / "data" / "injected" / account / "deck_injected.txt")]
        run_stage(account, "extract", cmd)
    run_stage(account, "verify",
              [PY, "scripts/verify/verify_claims.py", "--account", account,
               "--search-before-date", cutoff,
               "--runs", str(runs), "--workers", str(workers)])
    run_stage(account, "claims-file",
              [PY, "scripts/verify/make_claims_file.py", "--account", account])
    return account


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accounts", default=None,
                    help="comma-separated account ids (default: auto universe)")
    ap.add_argument("--only-canary", action="store_true",
                    help="run ONLY canary accounts (requires injected decks)")
    ap.add_argument("--parallel", type=int, default=3,
                    help="accounts processed concurrently")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the account plan, run nothing")
    args = ap.parse_args()

    canary_ids = {p.stem for p in CANARY_RAW.glob("*.json")}
    if args.accounts:
        accounts = [a.strip() for a in args.accounts.split(",") if a.strip()]
    else:
        universe = usable_accounts()
        if args.only_canary:
            accounts = [a for a in universe if a in canary_ids]
            missing = [a for a in accounts
                       if not (REPO_ROOT / "data" / "injected" / a /
                               "deck_injected.txt").is_file()]
            if missing:
                print(f"ERROR: --only-canary but {len(missing)} account(s) have no "
                      f"injected deck yet (run inject.py first): {missing[:5]}",
                      file=sys.stderr)
                return 1
        else:
            accounts = [a for a in universe if a not in canary_ids]
    if args.limit:
        accounts = accounts[: args.limit]

    dates = load_dates()
    undated = [a for a in accounts if a not in dates]
    if undated:
        print(f"ERROR: no doc_received_date for {len(undated)} account(s): "
              f"{undated[:10]}", file=sys.stderr)
        return 1

    log(f"=== run_all: {len(accounts)} account(s), parallel={args.parallel}, "
        f"runs={args.runs}, workers={args.workers}, "
        f"only_canary={args.only_canary} ===")
    if args.dry_run:
        for a in accounts:
            print(f"  {a}  cutoff={dates[a]}")
        return 0

    t0 = time.time()
    ok, failed = [], []
    aborted = False
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(process_account, a, dates[a], args.runs, args.workers,
                      args.only_canary): a
            for a in accounts
        }
        for fut in as_completed(futures):
            a = futures[fut]
            try:
                fut.result()
                ok.append(a)
            except CircuitBreakerTripped as e:
                # A whole API pool looks dead. Don't churn every remaining deck
                # against it — cancel pending accounts and stop the batch.
                failed.append(a)
                aborted = True
                log(f"[{a}] CIRCUIT BREAKER: {e}")
                log("=== ABORTING BATCH: API pool appears dead/empty. "
                    "Top up credits and re-run — completed decks are cached "
                    "and skipped, partial verifications resume per-claim. ===")
                for f in futures:
                    f.cancel()
                break
            except Exception as e:
                failed.append(a)
                log(f"[{a}] FAILED: {e}")
            log(f"=== progress: {len(ok)} ok, {len(failed)} failed, "
                f"{len(accounts) - len(ok) - len(failed)} pending, "
                f"{(time.time() - t0) / 60:.0f} min elapsed ===")

    log(f"=== {'ABORTED' if aborted else 'DONE'}: {len(ok)} ok, "
        f"{len(failed)} failed in {(time.time() - t0) / 60:.0f} min ===")
    if failed:
        log(f"failed accounts: {failed}")
    if aborted:
        return VERIFY_ABORT_CODE
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
