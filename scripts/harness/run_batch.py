"""Batch runner for the debate harness.

Usage:
  python scripts/harness/run_batch.py \
      --targets data/targets_synthetic.csv --conditions C0,C2 \
      --concurrency 8 --runs 1 [--out results/debate_runs.jsonl] \
      [--T 2 --K 5,4] [--limit N] [--only ACCOUNT_ID,...]

Iterates companies x conditions x runs; resumable via the per-key JSON cache
(data/cache/harness/). Writes one JSONL line per (company, condition, run).

Evidence sources:
  C0                 largest .txt under data/documents/<account_id>/parsed/
                     (READ-ONLY; truncated to ~12k chars)
  C1/C2/C2shuf/C3    data/claims/<account_id>.json — a list of
                     {"claim", "verdict", "source_reliability", "consistency"}

C2shuf: the verdicts are shuffled ACROSS claims here in the caller (seeded by
company_id:seed so it is reproducible); debate.py formats C2shuf identically
to C2.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import cache  # noqa: E402
from debate import CONDITIONS, DebateHarness, build_evidence_block  # noqa: E402

DOCS_DIR = REPO_ROOT / "data" / "documents"
CLAIMS_DIR = REPO_ROOT / "data" / "claims"
INJECTED_DIR = REPO_ROOT / "data" / "injected"


def load_targets(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_raw_text(account_id: str, injected: bool = False) -> str:
    """Pick the deck text for C0. With injected=True (E1 canary runs), read
    the canary-injected twin under data/injected/ and hard-fail if absent —
    C0 on a canary deck must NEVER silently fall back to the original."""
    if injected:
        p = INJECTED_DIR / account_id / "deck_injected.txt"
        if not p.is_file():
            raise FileNotFoundError(
                f"--c0-injected: no injected deck for {account_id} at {p} "
                f"(run scripts/canaries/inject.py first)")
        return p.read_text(encoding="utf-8", errors="replace")
    parsed = DOCS_DIR / account_id / "parsed"
    txts = sorted(parsed.glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)
    if not txts:
        raise FileNotFoundError(f"no parsed .txt for {account_id} under {parsed}")
    return txts[0].read_text(encoding="utf-8", errors="replace")


def load_claims(account_id: str) -> list[dict]:
    path = CLAIMS_DIR / f"{account_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no claims file for {account_id} at {path} "
            "(build it with scripts/verify/make_claims_file.py, or use the "
            "included synthetic claims files under data/claims/)"
        )
    claims = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(claims, list) or not claims:
        raise ValueError(f"claims file {path} must be a non-empty JSON list")
    return claims


def shuffle_verdicts(claims: list[dict], rng_key: str) -> list[dict]:
    """C2shuf: permute the verdict assignment across claims (derangement not
    enforced; seeded shuffle for reproducibility)."""
    rng = random.Random(rng_key)
    verdicts = [c.get("verdict") for c in claims]
    rng.shuffle(verdicts)
    return [{**c, "verdict": v} for c, v in zip(claims, verdicts)]


def build_evidence(account_id: str, condition: str, seed: int,
                   c0_injected: bool = False) -> str:
    if condition == "C0":
        return build_evidence_block(
            "C0", raw_text=load_raw_text(account_id, injected=c0_injected))
    claims = load_claims(account_id)
    if condition == "C2shuf":
        claims = shuffle_verdicts(claims, f"{account_id}:{seed}")
    return build_evidence_block(condition, claims=claims)


def existing_out_keys(out_path: Path) -> set[tuple]:
    keys = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["company_id"], r["condition"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", required=True, help="CSV with an account_id column")
    ap.add_argument("--conditions", required=True,
                    help=f"comma-separated subset of {','.join(CONDITIONS)}")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--runs", type=int, default=1, help="runs per cell; seed = run index")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "debate_runs.jsonl"))
    ap.add_argument("--T", type=int, default=2, help="refinement iterations")
    ap.add_argument("--K", default="5,4", help="comma-separated top-K per iteration")
    ap.add_argument("--limit", type=int, default=None, help="cap number of companies")
    ap.add_argument("--only", default=None, help="comma-separated account_ids to run")
    ap.add_argument("--c0-injected", action="store_true",
                    help="E1: C0 debates the canary-injected deck "
                         "(data/injected/<id>/deck_injected.txt); hard-fails "
                         "if an account has no injected deck")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        ap.error(f"unknown conditions: {bad}")
    K = [int(k) for k in args.K.split(",")]

    targets = load_targets(Path(args.targets))
    ids = [t["account_id"] for t in targets if t.get("account_id")]
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        ids = [i for i in ids if i in keep]
    if args.limit:
        ids = ids[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = existing_out_keys(out_path)
    out_lock = threading.Lock()

    harness = DebateHarness(T=args.T, K=K)
    tasks = [
        (cid, cond, seed)
        for cid in ids
        for cond in conditions
        for seed in range(args.runs)
    ]
    print(f"{len(tasks)} tasks ({len(ids)} companies x {conditions} x {args.runs} runs)")

    def to_line(result: dict) -> dict:
        return {
            "company_id": result["company_id"],
            "condition": result["condition"],
            "seed": result["seed"],
            "arguments_final": result["arguments_final"],
            "critiques": result["critiques"],
            "decision": result["decision"],
            "p_invest": result["p_invest"],
            "score_decision": result["score_decision"],
            "pro_avg_score": result["pro_avg_score"],
            "contra_avg_score": result["contra_avg_score"],
            "tokens": result["tokens"],
            "cost_usd": result["cost_usd"],
            "timings": result["timings"],
        }

    def run_one(task: tuple) -> tuple:
        cid, cond, seed = task
        cached = cache.get(cid, cond, seed)
        if cached is not None:
            return task, cached, True, None
        try:
            evidence = build_evidence(cid, cond, seed, args.c0_injected)
            result = harness.run(cid, cond, evidence, seed=seed)
            cache.put(cid, cond, seed, result)
            return task, result, False, None
        except Exception as e:  # keep the batch going
            traceback.print_exc()
            return task, None, False, e

    n_ok = n_cached = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futures):
            task, result, was_cached, err = fut.result()
            cid, cond, seed = task
            if err is not None:
                n_err += 1
                print(f"[ERR ] {cid} {cond} seed={seed}: {err}")
                continue
            n_cached += was_cached
            n_ok += 1
            key = (result["company_id"], result["condition"], result["seed"])
            with out_lock:
                if key not in written:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(to_line(result), ensure_ascii=False) + "\n")
                    written.add(key)
            print(
                f"[{'hit ' if was_cached else 'run '}] {cid} {cond} seed={seed} "
                f"decision={result['decision']} p_invest={result['p_invest']:.4f} "
                f"cost=${result['cost_usd']:.4f} "
                f"wall={result['timings'].get('total', 0):.0f}s"
            )

    print(
        f"done: {n_ok} ok ({n_cached} cache hits), {n_err} errors "
        f"in {time.time() - t0:.0f}s -> {out_path}"
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
