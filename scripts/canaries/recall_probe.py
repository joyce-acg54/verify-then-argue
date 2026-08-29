#!/usr/bin/env python3
"""Verification-recall probe: can the pipeline retrieve facts KNOWN to be public?

Usage (from the repo root):
  python scripts/canaries/recall_probe.py --dry-run        # print claims plan
  python scripts/canaries/recall_probe.py --limit 2        # smoke (~$0.20)
  python scripts/canaries/recall_probe.py                  # all (~$8-10 both modes)
  python scripts/canaries/recall_probe.py --mode uncapped --runs 3
  python scripts/canaries/recall_probe.py --provider exa --limit 2  # 2nd-provider smoke (~$0.30)
  python scripts/canaries/recall_probe.py --provider exa   # full Exa run (~$12-18 both modes)

Purpose (calibrates the headline no_evidence rate): 65% of routed-verifiable
deck claims came back no_evidence. That conflates (a) genuinely undocumented
private-company facts with (b) pipeline retrieval misses. The 82 web-checkable
canaries have TRUE values asserted (at canary construction) to be publicly
documented. This probe renders each TRUE fact as a standalone claim and runs
it through the exact production verification stack (5x rotated Sonar ->
gpt-4o verdict -> adjudication). The belief rate on these known-public facts
is the measured retrieval recall R_v; no_evidence on them is a measured
false-negative.

Modes (default: both):
  uncapped  no date filter — the pipeline's retrieval ceiling
  capped    per-deck doc_received_date cutoff — decision-time realistic
            (some true facts, e.g. funding rounds announced after deck
            receipt, are only documented later; capped recall <= uncapped)

The claim text is rendered by gpt-4o-mini from company identity + the TRUE
span only — the falsified value is never given to any model, and a hard
assert rejects any rendering that contains it. Resume-safe per
(canary_id, mode): records whose latest version contains api_error runs are
re-run. Output: data/canaries/recall_probe_results.jsonl. Costs logged via
the harness cost log (stages: recall_render / sonar_search / verdict_gpt4o).

--provider exa (second-provider robustness run) re-runs the probe with Exa as
the retrieval provider; the gpt-4o verdict stage is shared with the Sonar
path, so the UNCAPPED contrast isolates the retrieval provider. Writes
recall_probe_results_exa.jsonl (stages: exa_query / exa_search /
verdict_gpt4o) and reuses the Sonar run's rendered claim sentences so the
claims are byte-identical across providers. CAPPED-mode caveat: Exa's
API-side date filter empirically leaks undated post-cutoff pages, so capped
mode enforces the cutoff client-side and fail-closed (dated pre-cutoff pages
only); capped Exa recall is a structural lower bound, not directly
comparable to capped Sonar (see
vendored/verification_exa.py docstring and PROVENANCE #13).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify"))

import common as vcommon  # noqa: E402  (loads .env)
import costlog  # noqa: E402
from vendored import (  # noqa: E402
    adjudicator, config, verification_exa, verification_sonar,
)
from vendored.models import (  # noqa: E402
    AtomicClaim, ClaimCategory, ClaimScope, ClaimType, RoutedClaim,
    VerifiabilityLabel,
)

from openai import OpenAI  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "canaries" / "raw"
CRM_PATH = REPO_ROOT / "data" / "canaries" / "crm_reference.json"
OUT_PATH = REPO_ROOT / "data" / "canaries" / "recall_probe_results.jsonl"
TARGET_FILES = ("targets_scale.csv", "targets_pilot.csv", "targets_full.csv",
                "targets_synthetic.csv")

PROVIDERS = {"sonar": verification_sonar, "exa": verification_exa}


def out_path(provider: str) -> Path:
    """Sonar keeps the original path; other providers get a suffixed file."""
    if provider == "sonar":
        return OUT_PATH
    return OUT_PATH.with_name(f"recall_probe_results_{provider}.jsonl")

RENDER_MODEL = "gpt-4o-mini"

RENDER_SYSTEM = (
    "You turn a fact from a company's materials into ONE standalone "
    "declarative sentence that names the company. Keep every number, unit, "
    "currency and proper name EXACTLY as given — do not round, convert, "
    "embellish or add information. Output only the sentence."
)


def load_doc_dates() -> dict[str, str]:
    dates: dict[str, str] = {}
    for name in TARGET_FILES:
        path = REPO_ROOT / "data" / name
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                acc = row.get("account_id", "").strip()
                raw = (row.get("doc_received_date") or "").strip()
                if acc and raw and acc not in dates:
                    try:
                        dates[acc] = datetime.strptime(
                            raw, "%Y-%m-%d").strftime("%m/%d/%Y")
                    except ValueError:
                        pass
    return dates


def load_probes() -> list[dict]:
    """One probe per web-checkable, non-dropped canary."""
    crm = json.loads(CRM_PATH.read_text(encoding="utf-8"))
    probes = []
    for path in sorted(RAW_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        account = raw["account_id"]
        info = crm.get(account, {})
        for i, c in enumerate(raw.get("canaries", [])):
            if c.get("qc_status") == "dropped":
                continue
            if c.get("web_checkable") not in (True, "True", "true"):
                continue
            probes.append({
                "canary_id": f"{account}_{i}",
                "account_id": account,
                "company": raw.get("company", info.get("Name", "")),
                "website": info.get("Website") or "",
                "country": info.get("BillingCountry") or "",
                "fact_type": c.get("fact_type", ""),
                "true_span": c.get("original_span", ""),
                "falsified_span": c.get("falsified_span", ""),
            })
    return probes


def render_claim(client: OpenAI, p: dict) -> str:
    ident = p["company"]
    extra = ", ".join(x for x in (p["website"], p["country"]) if x)
    if extra:
        ident += f" ({extra})"
    user = (f"Company: {ident}\n"
            f"Fact type: {p['fact_type']}\n"
            f"Fact as stated in the company's materials: \"{p['true_span']}\"")
    resp = client.chat.completions.create(
        model=RENDER_MODEL, temperature=0,
        messages=[{"role": "system", "content": RENDER_SYSTEM},
                  {"role": "user", "content": user}],
        max_tokens=120,
    )
    costlog.log_openai_response(resp, stage="recall_render",
                                model=RENDER_MODEL)
    text = (resp.choices[0].message.content or "").strip()
    fs = p["falsified_span"].strip()
    if fs and fs.lower() in text.lower():
        raise RuntimeError(
            f"{p['canary_id']}: rendered claim contains the falsified span "
            f"— refusing (render: {text!r})")
    return text


def load_rendered_claims(path: Path) -> dict[str, str]:
    """canary_id -> rendered claim text from an existing results file (latest
    record wins), so a second-provider run uses byte-identical claims."""
    out: dict[str, str] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("claim"):
                        out[r["canary_id"]] = r["claim"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return out


def done_keys(path: Path) -> set[tuple[str, str]]:
    """(canary_id, mode) pairs whose latest record is outage-free."""
    last: dict[tuple[str, str], dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    last[(r["canary_id"], r["mode"])] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    done = set()
    for k, r in last.items():
        if r.get("final_label") == "api_error":
            continue
        if any(run.get("verdict") == "api_error"
               for run in r.get("runs", [])):
            continue
        done.add(k)
    return done


def verify_probe(p: dict, claim_text: str, mode: str,
                 cutoff: str | None, n_runs: int,
                 provider: str = "sonar") -> dict:
    claim = AtomicClaim(
        claim_id=f"recall_{p['canary_id']}_{mode}",
        claim_text=claim_text,
        source_page=0,
        source_file="recall_probe",
        startup_id=p["account_id"],
        category=ClaimCategory("other"),
        speaker="company",
        scope=ClaimScope("mixed"),
        claim_type=ClaimType("factual"),
        support_confidence=0.0,
    )
    routed = RoutedClaim(claim=claim,
                         verifiability=VerifiabilityLabel.VERIFIABLE)
    desc = f"{p['company']} {p['website']}".strip()
    sbd = cutoff if mode == "capped" else None
    t0 = time.time()
    runs = PROVIDERS[provider].verify_claim(
        routed, n_runs=n_runs, startup_description=desc,
        search_before_date=sbd)
    scored = adjudicator.adjudicate(routed, runs, n_runs_expected=n_runs)
    return {
        "canary_id": p["canary_id"],
        "account_id": p["account_id"],
        "provider": provider,
        "mode": mode,
        "fact_type": p["fact_type"],
        "true_span": p["true_span"],
        "claim": claim_text,
        "search_before_date": sbd,
        "n_runs": n_runs,
        "runs": [{"run_index": r.run_index, "verdict": r.verdict,
                  "source_url": r.source_url, "source_tier": r.source_tier,
                  "evidence_text": r.evidence_text[:500],
                  **_run_query_fields(r)} for r in runs],
        "final_label": scored.final_label.value,
        "source_score": round(scored.source_score, 4),
        "consistency": round(1.0 - scored.semantic_entropy, 4),
        "confidence": round(scored.confidence, 4),
        "wall_s": round(time.time() - t0, 1),
        "ts": time.time(),
    }


def _run_query_fields(r) -> dict:
    """Surface the Exa arm's per-run query (+ fallback flag) from
    raw_response into the record, so query-duplication across angles is
    auditable post hoc (temp-0 query gen can collapse distinct angles to
    identical queries, deflating verdict-cluster entropy)."""
    try:
        raw = json.loads(r.raw_response) if r.raw_response else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    out = {}
    if raw.get("query"):
        out["query"] = raw["query"]
    if raw.get("query_source"):
        out["query_source"] = raw["query_source"]
    return out


def summarize(path: Path) -> None:
    from collections import Counter, defaultdict
    latest: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                latest[(r["canary_id"], r["mode"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
    by_mode: dict[str, Counter] = defaultdict(Counter)
    run_stats: dict[str, list] = defaultdict(lambda: [0, 0])  # supports, total
    query_stats: dict[str, list] = defaultdict(lambda: [0, 0])  # distinct, n
    for (_, mode), r in latest.items():
        by_mode[mode][r["final_label"]] += 1
        runs = r.get("runs", [])
        run_stats[mode][0] += sum(1 for x in runs
                                  if x.get("verdict") == "supports")
        run_stats[mode][1] += len(runs)
        queries = {x["query"] for x in runs if x.get("query")}
        if queries:
            query_stats[mode][0] += len(queries)
            query_stats[mode][1] += 1
    for mode, c in sorted(by_mode.items()):
        n = sum(c.values())
        bel, dis = c.get("belief", 0), c.get("disbelief", 0)
        print(f"[{mode}] n={n} {dict(c)}")
        print(f"  R_v (belief on known-public facts): {bel / n:.1%} | "
              f"retrieved-but-misjudged (disbelief): {dis / n:.1%} | "
              f"measured false-negative (no_evidence+ignorance): "
              f"{(c.get('no_evidence', 0) + c.get('ignorance', 0)) / n:.1%}")
        sup, tot = run_stats[mode]
        if tot:
            print(f"  run-level supports rate (adjudication-free, "
                  f"provider-comparable): {sup / tot:.1%} ({sup}/{tot})")
        dq, nq = query_stats[mode]
        if nq:
            print(f"  mean distinct queries per probe: {dq / nq:.2f}/5 "
                  f"(angle collapse under temp-0 query gen)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("uncapped", "capped", "both"),
                    default="both")
    ap.add_argument("--provider", choices=tuple(PROVIDERS), default="sonar",
                    help="retrieval provider; non-sonar providers write to "
                         "recall_probe_results_<provider>.jsonl and reuse "
                         "the Sonar run's rendered claims")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of canaries (smoke testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the probe plan; no API calls")
    ap.add_argument("--summary-only", action="store_true",
                    help="just recompute the summary from existing results")
    args = ap.parse_args()

    if args.summary_only:
        sp = out_path(args.provider)
        if not sp.exists():
            print(f"no results yet for provider={args.provider}: {sp}",
                  file=sys.stderr)
            return 1
        summarize(sp)
        return 0

    probes = load_probes()
    if args.limit:
        probes = probes[: args.limit]
    modes = ["uncapped", "capped"] if args.mode == "both" else [args.mode]
    dates = load_doc_dates()

    if args.dry_run:
        for p in probes:
            cut = dates.get(p["account_id"], "MISSING")
            print(f"  {p['canary_id']} [{p['fact_type']}] "
                  f"true='{p['true_span'][:50]}' cutoff={cut}")
        print(f"plan: {len(probes)} canaries x {len(modes)} mode(s) "
              f"x {args.runs} runs")
        return 0

    undated = [p["canary_id"] for p in probes
               if "capped" in modes and p["account_id"] not in dates]
    if undated:
        print(f"ERROR: no doc date for {undated[:5]} (capped mode)",
              file=sys.stderr)
        return 1

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    path = out_path(args.provider)
    done = done_keys(path)
    todo = [(p, m) for p in probes for m in modes
            if (p["canary_id"], m) not in done]
    print(f"{len(probes)} probes x {len(modes)} modes: "
          f"{len(done)} done, {len(todo)} to run "
          f"(runs={args.runs}, provider={args.provider})")

    rendered: dict[str, str] = {}
    if args.provider != "sonar":
        rendered.update(load_rendered_claims(path))      # own prior partial run
        rendered.update(load_rendered_claims(OUT_PATH))  # sonar canonical wins
        print(f"reusing {len(rendered)} rendered claims "
              f"(canonical: {OUT_PATH.name})")
    n_ok = n_err = 0
    for p, mode in todo:
        costlog.set_company(p["account_id"])
        try:
            if p["canary_id"] not in rendered:
                if args.provider != "sonar":
                    print(f"  [warn] {p['canary_id']}: no reusable rendered "
                          f"claim — rendering fresh (cross-provider claim "
                          f"text will differ for this canary)")
                rendered[p["canary_id"]] = render_claim(client, p)
            claim_text = rendered[p["canary_id"]]
            # Re-assert the falsified-span guard for REUSED claims too
            # (render_claim only checks fresh renders).
            fs = p["falsified_span"].strip()
            if fs and fs.lower() in claim_text.lower():
                raise RuntimeError(
                    "reused claim contains the falsified span — refusing "
                    f"(claim: {claim_text!r})")
            rec = verify_probe(p, claim_text, mode,
                               dates.get(p["account_id"]), args.runs,
                               provider=args.provider)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            print(f"[{n_ok}/{len(todo)}] {mode:<8} {rec['final_label']:<11} "
                  f"| {rec['claim'][:70]}")
        except Exception as e:
            n_err += 1
            print(f"[ERR ] {p['canary_id']} {mode}: {e}", file=sys.stderr)

    s = costlog.session_summary()
    print(f"done: {n_ok} ok, {n_err} errors | session cost "
          f"${s['cost_usd']:.2f} ({s['n_calls']} calls) -> {path}\n")
    if path.exists():
        summarize(path)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
