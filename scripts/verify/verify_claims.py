#!/usr/bin/env python3
"""Claim verification CLI: Sonar search + gpt-4o verdict + scoring + adjudication.

Usage:
  python scripts/verify/verify_claims.py --account <id> \
      [--search-before-date MM/DD/YYYY] [--runs 5] [--workers 4] [--limit N]

For every unique claim routed "verifiable" in
data/cache/verify/<account_id>_claims_raw.json:
  - N rotated-angle Perplexity Sonar searches (model `sonar`, PPLX key),
    each followed by a gpt-4o temperature-0 verdict over the evidence
  - tier-based Beta source scoring
  - verdict-cluster entropy (consistency = 1 - normalized entropy)
  - deterministic 4-way adjudication (belief/disbelief/ignorance/no_evidence)

Crash-safe: appends one JSON line per claim to
data/cache/verify/<account_id>_verify.jsonl and skips already-done claim
hashes on resume. --search-before-date is optional; omitted = uncapped.
Costs go to data/cache/cost_log.jsonl (harness format).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402  (loads .env)
import costlog  # noqa: E402
from vendored import adjudicator, verification_sonar  # noqa: E402
from vendored.models import (  # noqa: E402
    AtomicClaim, ClaimCategory, ClaimScope, ClaimType, RoutedClaim,
    VerifiabilityLabel,
)

_write_lock = threading.Lock()

# Distinct exit code so run_all.py can tell "this deck had a few claim errors"
# (exit 1) apart from "the API pool looks dead, stop the whole batch" (this).
ABORT_EXIT_CODE = 2


def _routed_from_record(rec: dict) -> RoutedClaim:
    claim = AtomicClaim(
        claim_id=rec["claim_id"],
        claim_text=rec["claim_text"],
        source_page=rec.get("source_page", 0),
        source_file=rec.get("source_file", ""),
        startup_id=rec.get("startup_id", ""),
        category=ClaimCategory(rec.get("category", "other")),
        speaker=rec.get("speaker", "company"),
        scope=ClaimScope(rec.get("scope", "mixed")),
        claim_type=ClaimType(rec.get("claim_type", "factual")),
        support_confidence=rec.get("support_confidence", 0.0),
    )
    return RoutedClaim(claim=claim,
                       verifiability=VerifiabilityLabel.VERIFIABLE)


def _done_hashes(path: Path) -> set[str]:
    """Claim hashes that are completed AND outage-free.

    A claim counts as done only if its most-recent record contains NO
    api_error run. A run fails with verdict 'api_error' on any exception,
    including out-of-credits / rate-limit errors (see verification_sonar.py).
    A partially-failed claim (e.g. credits ran out after 2 of 5 runs) thus
    re-enters the pending set and is re-verified on the next run with a full
    run set; make_claims_file.py takes last-write-wins, so the fresh record
    supersedes the degraded one. This guarantees every persisted verdict is
    computed from a complete, outage-free 5-run protocol — no verdict is
    silently shaped by a credit interruption."""
    last: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    last[r["claim_hash"]] = r  # last record wins
                except (json.JSONDecodeError, KeyError):
                    continue
    done = set()
    for h, r in last.items():
        runs = r.get("runs") or []
        if r.get("final_label") == "api_error":
            continue
        if any(run.get("verdict") == "api_error" for run in runs):
            continue
        done.add(h)
    return done


def _append_jsonl(path: Path, record: dict) -> None:
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    with _write_lock:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)


def verify_one(rec: dict, account_id: str, description: str,
               n_runs: int, search_before_date: str | None) -> dict:
    routed = _routed_from_record(rec)
    t0 = time.time()
    runs = verification_sonar.verify_claim(
        routed, n_runs=n_runs,
        startup_description=description,
        search_before_date=search_before_date,
    )
    scored = adjudicator.adjudicate(routed, runs, n_runs_expected=n_runs)

    return {
        "account_id": account_id,
        "claim_id": rec["claim_id"],
        "claim_hash": common.claim_hash(rec["claim_text"]),
        "claim": rec["claim_text"],
        "routing": "verifiable",
        "search_before_date": search_before_date,
        "n_runs": n_runs,
        "runs": [
            {
                "run_index": r.run_index,
                "verdict": r.verdict,
                "reasoning": r.reasoning,
                "evidence_text": r.evidence_text,
                "source_url": r.source_url,
                "source_domain": r.source_domain,
                "source_tier": r.source_tier,
                "raw_response": r.raw_response,
            }
            for r in runs
        ],
        "source_score": round(scored.source_score, 4),
        "beta_alpha": round(scored.beta_alpha, 2),
        "beta_beta": round(scored.beta_beta, 2),
        "entropy": round(scored.semantic_entropy, 4),
        "consistency": round(1.0 - scored.semantic_entropy, 4),
        "aleatoric": round(scored.aleatoric_uncertainty, 4),
        "epistemic": round(scored.epistemic_uncertainty, 4),
        "prediction_set": [l.value for l in scored.prediction_set],
        "final_label": scored.final_label.value,
        "confidence": round(scored.confidence, 4),
        "explanation": scored.explanation,
        "wall_s": round(time.time() - t0, 1),
        "ts": time.time(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--account", required=True)
    ap.add_argument("--search-before-date", default=None,
                    help="MM/DD/YYYY; only sources published before this date "
                         "(Perplexity search_before_date_filter). Omit = uncapped.")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4,
                    help="claims verified concurrently (runs within a claim are sequential)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of claims (testing)")
    ap.add_argument("--abort-after", type=int, default=8,
                    help="circuit breaker: abort the run (exit %d) after this "
                         "many CONSECUTIVE fully-failed claims (all runs "
                         "api_error) — a dead/empty API pool, not a flaky "
                         "claim. Completed claims are saved; resume continues. "
                         "0 disables." % ABORT_EXIT_CODE)
    args = ap.parse_args()

    if args.search_before_date:
        try:
            datetime.strptime(args.search_before_date, "%m/%d/%Y")
        except ValueError:
            ap.error("--search-before-date must be MM/DD/YYYY")

    raw_path = common.claims_raw_path(args.account)
    if not raw_path.exists():
        ap.error(f"no extraction output at {raw_path} — run extract_claims.py first")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    # short company context for search framing (first pseudo-page is too long;
    # use deck filename + account id, which is what the source pipeline had)
    description = raw.get("source_file", "")

    todo = [c for c in raw["claims"]
            if c.get("routing") == "verifiable" and not c.get("is_duplicate")]
    if args.limit:
        todo = todo[: args.limit]

    out_path = common.verify_jsonl_path(args.account)
    common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    done = _done_hashes(out_path)
    pending = [c for c in todo if common.claim_hash(c["claim_text"]) not in done]
    print(f"[{args.account}] {len(todo)} verifiable claims, "
          f"{len(done)} already done, {len(pending)} to verify "
          f"(runs={args.runs}, cutoff={args.search_before_date or 'uncapped'})")

    costlog.set_company(args.account)
    t0 = time.time()
    n_ok = n_err = 0
    consecutive_fail = 0   # streak of fully-failed completions (api_error)
    aborted = False
    verdict_counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(verify_one, c, args.account, description,
                      args.runs, args.search_before_date): c
            for c in pending
        }
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                record = fut.result()
            except Exception as e:
                n_err += 1
                consecutive_fail += 1
                print(f"[ERR ] {c['claim_text'][:60]}...: {e}")
            else:
                _append_jsonl(out_path, record)
                label = record["final_label"]
                if label == "api_error":
                    # All runs failed — counts toward the dead-pool streak but
                    # is NOT counted as a successful verification.
                    n_err += 1
                    consecutive_fail += 1
                    print(f"[ERR ] all runs api_error | {record['claim'][:60]}")
                else:
                    n_ok += 1
                    consecutive_fail = 0   # the API answered → pool is alive
                    verdict_counts[label] = verdict_counts.get(label, 0) + 1
                    print(f"[{n_ok}/{len(pending)}] {label:<11} "
                          f"src={record['source_score']:.2f} "
                          f"cons={record['consistency']:.2f} "
                          f"| {record['claim'][:70]}")

            if args.abort_after and consecutive_fail >= args.abort_after:
                aborted = True
                print(f"\n!! CIRCUIT BREAKER: {consecutive_fail} consecutive "
                      f"fully-failed claims — API pool likely dead/empty. "
                      f"Aborting (exit {ABORT_EXIT_CODE}). {n_ok} claims saved; "
                      f"top up credits and re-run to resume.", file=sys.stderr)
                # Cancel not-yet-started futures; in-flight ones finish and are
                # ignored. Their partial records (if any) are re-queued next run.
                for f in futures:
                    f.cancel()
                break

    s = costlog.session_summary()
    print(
        f"done: {n_ok} verified, {n_err} errors in {time.time() - t0:.0f}s | "
        f"verdicts: {verdict_counts} | session cost ${s['cost_usd']:.4f} "
        f"({s['n_calls']} calls) -> {out_path}"
    )
    if aborted:
        return ABORT_EXIT_CODE
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
