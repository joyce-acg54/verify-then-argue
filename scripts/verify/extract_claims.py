#!/usr/bin/env python3
"""Claim extraction CLI: page gate -> atomizer -> audit -> dedup -> router.

Usage:
  python scripts/verify/extract_claims.py --account ACCOUNT_ID
  python scripts/verify/extract_claims.py --targets data/targets_synthetic.csv [--limit N]

Reads the best deck text from data/documents/<account_id>/parsed/ (largest
parsed .txt that is not an email-export wrapper; accounts with no
file >= 4000 chars are skipped). All LLM calls on gpt-4o-mini.

Writes data/cache/verify/<account_id>_claims_raw.json and appends per-call
cost records to data/cache/cost_log.jsonl (harness format).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402  (loads .env)
import costlog  # noqa: E402
from vendored import claim_router, deduplication, l1_extraction  # noqa: E402
from vendored.models import Page  # noqa: E402


def extract_account(account_id: str, workers: int = 4,
                    deck_path_override: str | None = None) -> dict | None:
    if deck_path_override:
        p = Path(deck_path_override)
        if not p.is_file():
            print(f"[skip] {account_id}: deck override not found: {p}")
            return None
        text = p.read_text(encoding="utf-8")
        if len(text) < common.MIN_DECK_CHARS:
            print(f"[skip] {account_id}: deck override < {common.MIN_DECK_CHARS} chars")
            return None
        deck = (p, text)
    else:
        deck = common.pick_deck(account_id)
    if deck is None:
        print(f"[skip] {account_id}: no parsed deck >= {common.MIN_DECK_CHARS} chars")
        return None
    deck_path, text = deck
    costlog.set_company(account_id)
    t0 = time.time()

    page_texts = common.segment_pages(text)
    pages = [
        Page(startup_id=account_id, source_file=deck_path.name,
             page_number=i + 1, page_text=pt)
        for i, pt in enumerate(page_texts)
    ]
    print(f"[{account_id}] {deck_path.name}: {len(text)} chars -> "
          f"{len(pages)} pseudo-pages")

    # L1A + L1B + L1C per page, in parallel
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(l1_extraction.process_page, pages))

    gated_pages = [p for p, _, _ in results if p.should_extract]
    valid, audited_out = [], []
    for _, v, d in results:
        valid.extend(v)
        audited_out.extend(d)
    n_extracted = len(valid) + len(audited_out)
    print(f"[{account_id}] gate: {len(gated_pages)}/{len(pages)} pages | "
          f"extracted {n_extracted} claims, audited out {len(audited_out)}")

    # Dedup (marks duplicates, keeps canonical)
    deduped = deduplication.deduplicate_claims(valid)
    unique = deduplication.filter_unique(deduped)
    n_dup = len(deduped) - len(unique)

    # Router on unique claims
    to_verify, flagged, borderline = claim_router.route_claims(unique)
    routing_by_id = {}
    for rc in to_verify + flagged + borderline:
        routing_by_id[rc.claim.claim_id] = rc.verifiability.value
    routing_split = {}
    for label in ("verifiable", "unverifiable", "inference", "normative"):
        routing_split[label] = sum(1 for v in routing_by_id.values() if v == label)

    print(f"[{account_id}] dedup: {n_dup} duplicates, {len(unique)} unique | "
          f"routing: {routing_split} (borderline: {len(borderline)})")

    def claim_record(c) -> dict:
        d = asdict(c)
        d["category"] = c.category.value
        d["scope"] = c.scope.value
        d["claim_type"] = c.claim_type.value
        d["routing"] = routing_by_id.get(c.claim_id)  # None for duplicates
        return d

    out = {
        "account_id": account_id,
        "source_file": deck_path.name,
        "n_chars": len(text),
        "n_pages": len(pages),
        "n_pages_gated": len(gated_pages),
        "counts": {
            "extracted": n_extracted,
            "audited_out": len(audited_out),
            "after_audit": len(valid),
            "duplicates": n_dup,
            "unique": len(unique),
            "routing": routing_split,
            "borderline": len(borderline),
        },
        "claims": [claim_record(c) for c in deduped],
        "audited_out_claims": [
            {"claim_text": c.claim_text, "source_page": c.source_page,
             "audit_reason": c.audit_reason}
            for c in audited_out
        ],
        "config": {
            "extract_model": "gpt-4o-mini",
            "dedup_threshold": deduplication.config.DEDUP_SIMILARITY_THRESHOLD,
        },
        "wall_s": round(time.time() - t0, 1),
        "ts": time.time(),
    }

    common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = common.claims_raw_path(account_id)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[{account_id}] wrote {out_path} in {out['wall_s']}s")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--account", action="append",
                   help="account id (repeatable / comma-separated)")
    g.add_argument("--targets", help="CSV with an account_id column")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deck-path", default=None,
                    help="explicit deck text file to extract from (e.g. "
                         "data/injected/<id>/deck_injected.txt); requires "
                         "exactly one --account")
    args = ap.parse_args()

    if args.account:
        ids = [a for chunk in args.account for a in chunk.split(",") if a]
    else:
        with open(args.targets, newline="", encoding="utf-8") as f:
            ids = [r["account_id"] for r in csv.DictReader(f) if r.get("account_id")]
    if args.limit:
        ids = ids[: args.limit]
    if args.deck_path and len(ids) != 1:
        ap.error("--deck-path requires exactly one --account")

    n_ok = 0
    for aid in ids:
        try:
            if extract_account(aid, workers=args.workers,
                               deck_path_override=args.deck_path) is not None:
                n_ok += 1
        except Exception as e:
            print(f"[ERR ] {aid}: {e}")
    s = costlog.session_summary()
    print(f"done: {n_ok}/{len(ids)} accounts | session cost ${s['cost_usd']:.4f} "
          f"over {s['n_calls']} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
