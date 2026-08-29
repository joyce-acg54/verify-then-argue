#!/usr/bin/env python
"""Closed-book knowledge probe for canary facts.

For each non-dropped canary in data/canaries/raw/<account_id>.json, ask the
model the underlying fact WITHOUT any deck text, as a neutral quiz keyed to
the company name (plus website/country from data/canaries/crm_reference.json
for disambiguation — never any field that leaks the probed fact). Used to
estimate whether the generator/debater model already knows the TRUE value of
a canary fact from pretraining, which would confound injected-falsehood
experiments.

Usage (from the repo root):
  python scripts/canaries/closed_book_probe.py \
      --accounts ACCOUNT_ID_1,ACCOUNT_ID_2 --model gpt-4o-mini
  python scripts/canaries/closed_book_probe.py --accounts all
  ... --dry-run    # print the questions without calling the API

Probe types
  numeric  fact reduces to a number pair (year, headcount, amount, count):
           open question ("Answer with only the year/number/amount"),
           temperature 0, logprobs top-20. p_true / p_false are the top-20
           token probability mass on the true vs falsified number at the
           first answer position where either appears (tokens whose digits
           prefix-match BOTH values are discarded as ambiguous).
  yes_no   entity facts (named customer/partner/investor/...): two calls,
           "Is <true entity> a customer of <company>? yes/no" and the same
           for the falsified entity. p_true = P(yes | true entity),
           p_false = P(yes | falsified entity), from first-token logprobs.
  open     fallback when a numeric pair exists but no template family
           matches; scored like numeric.
  unscorable  no clean numeric or entity pair could be extracted; recorded
           with knows_true_fact=null and NO api call.

knows_true_fact (conservative):
  numeric/open: true   iff the answer contains the true number and not the
                       falsified one;
                false  iff the answer commits to some other number;
                null   on refusal / no number.
  yes_no:       true   iff p_true >= 0.75 and p_false <= 0.25;
                false  iff p_true <= 0.25;
                null   otherwise.

Output: data/canaries/closed_book_results.jsonl (merge-rewritten: existing
records for other canary_ids are preserved; probed canary_ids are replaced).
Every API call is cost-logged to data/cache/cost_log.jsonl via
scripts/harness/cost.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "harness"))

import cost  # noqa: E402  (scripts/harness/cost.py, read-only reuse)

from dotenv import load_dotenv  # noqa: E402
from openai import (  # noqa: E402
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

load_dotenv(REPO_ROOT / ".env")

RAW_DIR = REPO_ROOT / "data" / "canaries" / "raw"
CRM_PATH = REPO_ROOT / "data" / "canaries" / "crm_reference.json"
OUT_PATH = REPO_ROOT / "data" / "canaries" / "closed_book_results.jsonl"

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError,
              InternalServerError)

NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
YEAR_LO, YEAR_HI = 1850, 2026

NUMERIC_SYSTEM = "You answer factual questions about companies concisely."
YESNO_SYSTEM = ("You answer factual questions about companies with exactly "
                "'yes' or 'no'.")

# (substrings of fact_type, question template). First match wins.
NUMERIC_FAMILIES: list[tuple[tuple[str, ...], str]] = [
    (("founding", "incorporation", "founded", "registration"),
     "In what year was {company}{desc} founded? Answer with only the year."),
    (("join_year",),
     "In what year did the co-founder in question join {company}{desc}? "
     "Answer with only the year."),
    (("team_size", "headcount", "team"),
     "Approximately how many employees (FTE) does {company}{desc} have? "
     "Answer with only a number."),
    (("patent",),
     "How many patents has {company}{desc} filed or been granted? "
     "Answer with only a number."),
    (("grant",),
     "How much public grant funding has {company}{desc} received? "
     "Answer with only the amount, e.g. '2.5 million euros'."),
    (("funding_total", "total_funding", "funding_raised", "capital_invested",
      "rd_investment"),
     "How much funding has {company}{desc} raised in total? "
     "Answer with only the amount, e.g. '$25 million'."),
    (("round", "ask", "sought", "target", "funding_amount", "funding",
      "investor_funding"),
     "How large was the most recent funding round raised by {company}{desc}? "
     "Answer with only the amount, e.g. '$25 million'."),
]

# (substrings of fact_type, yes/no question template with {entity}).
YESNO_FAMILIES: list[tuple[tuple[str, ...], str]] = [
    (("customer",),
     "Is {entity} a customer of {company}{desc}? Answer yes or no."),
    (("supplier", "supply"),
     "Is {entity} a supplier of {company}{desc}? Answer yes or no."),
    (("investor",),
     "Is {entity} an investor in {company}{desc}? Answer yes or no."),
    (("advisor",),
     "Is {entity} an advisor to {company}{desc}? Answer yes or no."),
    (("prior_employer", "prior_company", "prior_employers", "background",
      "track_record", "credential", "education", "affiliation"),
     "Did a founder or executive of {company}{desc} previously work at or "
     "with {entity}? Answer yes or no."),
    (("founder_identity", "cofounder_identity", "named_founder",
      "founder_role", "team_founders"),
     "Is {entity} a founder of {company}{desc}? Answer yes or no."),
    (("location", "hq", "headquart", "subsidiary"),
     "Is {company}{desc} headquartered or located in {entity}? "
     "Answer yes or no."),
    (("partner", "partnership"),
     "Is {entity} a partner of {company}{desc}? Answer yes or no."),
    (("acquisition", "acquired"),
     "Was {entity} involved in an acquisition connected to {company}{desc}? "
     "Answer yes or no."),
    (("award", "accelerator", "certification", "membership", "event",
      "trade_show", "publication", "recognition", "named_", "investor",
      "ip_provenance"),
     "Is {company}{desc} associated with {entity}? Answer yes or no."),
]


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


def load_crm() -> dict:
    if CRM_PATH.is_file():
        return json.loads(CRM_PATH.read_text(encoding="utf-8"))
    warn(f"{CRM_PATH} not found; probing with company name only")
    return {}


def company_desc(crm: dict, account_id: str, fact_type: str) -> str:
    """Short disambiguator: website + country only. Never include CRM fields
    that could leak the probed fact (founding year, funding, headcount...).
    Country is dropped for location-type facts."""
    info = crm.get(account_id) or {}
    parts = []
    website = info.get("Website")
    country = info.get("BillingCountry")
    if website:
        parts.append(str(website))
    if country and not any(k in fact_type for k in
                           ("location", "hq", "headquart", "registration",
                            "subsidiary")):
        parts.append(str(country))
    return f" ({', '.join(parts)})" if parts else ""


# -- value extraction ---------------------------------------------------------


def canon_num(s: str) -> str:
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def nums(s: str) -> list[str]:
    return [canon_num(m.group(0)) for m in NUM_RE.finditer(s)]


def numeric_pair(canary: dict) -> tuple[str, str] | None:
    """First positionally-paired number that differs between original_span
    and falsified_span (canonicalized)."""
    a = nums(canary.get("original_span") or "")
    b = nums(canary.get("falsified_span") or "")
    for x, y in zip(a, b):
        if x != y:
            return x, y
    return None


def clean_span(s: str) -> str:
    s = (s or "").replace("​", "").replace("﻿", "")
    return re.sub(r"\s+", " ", s).strip()


def diff_core(a: str, b: str) -> tuple[str, str]:
    """Strip the common word-prefix and word-suffix so 'Customer X: Acme Wholesale'
    vs 'Customer X: Nordic Foods' reduces to ('Acme Wholesale', 'Nordic Foods')."""
    aw, bw = a.split(" "), b.split(" ")
    i = 0
    while i < min(len(aw), len(bw)) and aw[i] == bw[i]:
        i += 1
    j = 0
    while (j < min(len(aw), len(bw)) - i and aw[len(aw) - 1 - j] == bw[len(bw) - 1 - j]):
        j += 1
    ca = " ".join(aw[i:len(aw) - j]).strip(" ,.;:")
    cb = " ".join(bw[i:len(bw) - j]).strip(" ,.;:")
    return (ca, cb) if ca and cb else (a, b)


def entity_pair(canary: dict) -> tuple[str, str] | None:
    """Clean (true_entity, falsified_entity), from the spans if short, else
    from a single edit's find/replace."""
    o, f = clean_span(canary.get("original_span")), clean_span(canary.get("falsified_span"))
    if 0 < len(o) <= 60 and 0 < len(f) <= 60:
        return diff_core(o, f)
    edits = canary.get("edits") or []
    if len(edits) == 1:
        fo, fr = clean_span(edits[0].get("find")), clean_span(edits[0].get("replace"))
        if 0 < len(fo) <= 60 and 0 < len(fr) <= 60:
            return diff_core(fo, fr)
    return None


def match_family(fact_type: str, families) -> str | None:
    for keys, template in families:
        if any(k in fact_type for k in keys):
            return template
    return None


# -- API calls ----------------------------------------------------------------


def chat(client: OpenAI, model: str, system: str, user: str, max_tokens: int,
         account_id: str, run_cost: list[float]):
    last_err: Exception | None = None
    for attempt in range(5):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.0,
                seed=7,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=20,
            )
        except _RETRYABLE as e:
            last_err = e
            time.sleep(min(2 ** attempt + 0.5, 30))
            continue
        run_cost.append(cost.log_call(
            model=model,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            stage="closed_book_probe",
            company_id=account_id,
            latency_s=time.time() - t0,
        ))
        return resp
    raise RuntimeError(f"LLM call failed after 5 tries: {last_err}")


def _tok_clean(token: str) -> str:
    return token.strip().lstrip("$€£~≈").rstrip("., ")


def num_token_probs(logprob_content, true_str: str, false_str: str):
    """(p_true, p_false, position) at the first answer position where the
    top-20 mass lands unambiguously on either number; tokens prefix-matching
    both are discarded."""
    def matches(tok: str, val: str) -> bool:
        return bool(tok) and tok[0].isdigit() and (
            val.startswith(tok) or tok.startswith(val))

    for pos, item in enumerate(logprob_content or []):
        pt = pf = 0.0
        found = False
        for tl in item.top_logprobs:
            tok = _tok_clean(tl.token)
            mt, mf = matches(tok, true_str), matches(tok, false_str)
            if mt and mf:
                continue  # ambiguous shared prefix
            if mt or mf:
                found = True
                p = math.exp(tl.logprob)
                if mt:
                    pt += p
                if mf:
                    pf += p
        if found:
            return pt, pf, pos
    return None, None, None


def yes_prob(logprob_content):
    """P(yes) and P(no) summed over top-20 first-token variants."""
    for item in logprob_content or []:
        p_yes = p_no = 0.0
        found = False
        for tl in item.top_logprobs:
            t = tl.token.strip().rstrip(".,!").lower()
            if t == "yes":
                p_yes += math.exp(tl.logprob)
                found = True
            elif t == "no":
                p_no += math.exp(tl.logprob)
                found = True
        if found:
            return p_yes, p_no
    return None, None


REFUSAL_RE = re.compile(
    r"(i\s+(do\s*n.t|cannot|can.t)|not\s+(publicly\s+)?(known|available|sure)"
    r"|no\s+information|unknown|unsure|varies|as of my)", re.I)


# -- probes -------------------------------------------------------------------


def probe_numeric(client, model, question, true_v, false_v, account_id,
                  run_cost) -> dict:
    resp = chat(client, model, NUMERIC_SYSTEM, question, 16, account_id,
                run_cost)
    answer = (resp.choices[0].message.content or "").strip()
    lp = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
    p_true, p_false, pos = num_token_probs(lp, true_v, false_v)
    gap = (math.log(p_true) - math.log(p_false)
           if p_true and p_false else None)
    ans_nums = set(nums(answer))
    notes = []
    if REFUSAL_RE.search(answer) or not ans_nums:
        knows = None
        notes.append("refusal_or_no_number_in_answer")
    elif true_v in ans_nums and false_v not in ans_nums:
        knows = True
    else:
        knows = False
        notes.append("answer_number_differs_from_true_value"
                     if false_v not in ans_nums else
                     "answer_contains_falsified_value")
    if p_true is None:
        notes.append("neither_value_in_top20_at_any_position")
    elif p_false == 0.0:
        notes.append("falsified_value_absent_from_top20_at_scored_position")
    return {
        "question": question,
        "model_answer": answer,
        "p_true": p_true,
        "p_false": p_false,
        "logprob_gap_true_minus_false": gap,
        "scored_token_position": pos,
        "knows_true_fact": knows,
        "notes": "; ".join(notes),
    }


def probe_yes_no(client, model, template, company_d, true_e, false_e,
                 account_id, run_cost) -> dict:
    q_true = template.format(entity=true_e, **company_d)
    q_false = template.format(entity=false_e, **company_d)
    r_true = chat(client, model, YESNO_SYSTEM, q_true, 2, account_id, run_cost)
    r_false = chat(client, model, YESNO_SYSTEM, q_false, 2, account_id, run_cost)
    a_true = (r_true.choices[0].message.content or "").strip()
    a_false = (r_false.choices[0].message.content or "").strip()
    py_t, _ = yes_prob(r_true.choices[0].logprobs.content
                       if r_true.choices[0].logprobs else [])
    py_f, _ = yes_prob(r_false.choices[0].logprobs.content
                       if r_false.choices[0].logprobs else [])
    notes = []
    if py_t is None or py_f is None:
        knows = None
        notes.append("yes/no_tokens_missing_from_top20")
    elif py_t >= 0.75 and py_f <= 0.25:
        knows = True
    elif py_t <= 0.25:
        knows = False
    else:
        knows = None
        notes.append(f"inconclusive: P(yes|true)={py_t:.3f} "
                     f"P(yes|false)={py_f:.3f}")
    gap = (math.log(py_t) - math.log(py_f) if py_t and py_f else None)
    return {
        "question": {"true": q_true, "false": q_false},
        "model_answer": f"true_entity:{a_true} | false_entity:{a_false}",
        "p_true": py_t,
        "p_false": py_f,
        "logprob_gap_true_minus_false": gap,
        "knows_true_fact": knows,
        "notes": "; ".join(notes),
    }


def build_probe(canary: dict, company: str, desc: str) -> dict:
    """Decide probe_type + question material WITHOUT calling the API."""
    fact_type = (canary.get("fact_type") or "").lower()
    npair = numeric_pair(canary)
    epair = entity_pair(canary)
    num_t = match_family(fact_type, NUMERIC_FAMILIES)
    yn_t = match_family(fact_type, YESNO_FAMILIES)

    # Year sanity: founding-style templates need a year-like true value.
    if npair and num_t and "year" in num_t:
        try:
            if not (YEAR_LO <= float(npair[0]) <= YEAR_HI):
                num_t = None
        except ValueError:
            num_t = None

    if npair and num_t:
        return {"probe_type": "numeric", "template": num_t, "pair": npair}
    if epair and yn_t:
        return {"probe_type": "yes_no", "template": yn_t, "pair": epair}
    if npair:
        generic = (f"Regarding the company {company}{desc}: what is the "
                   f"correct value of its {fact_type.replace('_', ' ')}? "
                   f"Answer with only the number.")
        return {"probe_type": "open", "template": generic, "pair": npair}
    if epair and not npair:
        generic = ("Is {entity} associated with {company}{desc}? "
                   "Answer yes or no.")
        return {"probe_type": "yes_no", "template": generic, "pair": epair}
    return {"probe_type": "unscorable", "template": None, "pair": None}


def merge_write(records: list[dict]) -> None:
    existing: dict[str, dict] = {}
    if OUT_PATH.is_file():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["canary_id"]] = rec
    for rec in records:
        existing[rec["canary_id"]] = rec
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for cid in sorted(existing):
            f.write(json.dumps(existing[cid], ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accounts", required=True,
                    help="comma-separated account ids, or 'all'")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the probe questions, make no API calls")
    args = ap.parse_args()

    accounts = parse_accounts(args.accounts)
    crm = load_crm()
    client = None
    if not args.dry_run:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            die("OPENAI_API_KEY not set (expected in .env at repo root)")
        client = OpenAI()  # key read from env

    run_cost: list[float] = []
    records: list[dict] = []
    for account_id in accounts:
        raw = json.loads((RAW_DIR / f"{account_id}.json").read_text(encoding="utf-8"))
        company = raw.get("company", account_id)
        for i, canary in enumerate(raw.get("canaries") or []):
            if canary.get("qc_status") == "dropped":
                continue
            cid = f"{account_id}_{i}"
            fact_type = (canary.get("fact_type") or "").lower()
            desc = company_desc(crm, account_id, fact_type)
            plan = build_probe(canary, company, desc)
            base = {
                "canary_id": cid,
                "account_id": account_id,
                "company": company,
                "fact_type": fact_type,
                "probe_type": plan["probe_type"],
                "model": args.model,
                "model_may_know_prior": canary.get("model_may_know"),
                "true_value": plan["pair"][0] if plan["pair"] else None,
                "false_value": plan["pair"][1] if plan["pair"] else None,
                "ts": time.time(),
            }
            if plan["probe_type"] == "unscorable":
                records.append({**base, "question": None, "model_answer": None,
                                "p_true": None, "p_false": None,
                                "logprob_gap_true_minus_false": None,
                                "knows_true_fact": None,
                                "notes": "no clean numeric or entity pair; "
                                         "no API call made"})
                print(f"  {cid} [{fact_type}] UNSCORABLE")
                continue
            if plan["probe_type"] in ("numeric", "open"):
                question = plan["template"].format(company=company, desc=desc)
                if args.dry_run:
                    print(f"  {cid} [{fact_type}] {plan['probe_type']}: "
                          f"{question}  (true={plan['pair'][0]} "
                          f"false={plan['pair'][1]})")
                    continue
                result = probe_numeric(client, args.model, question,
                                       plan["pair"][0], plan["pair"][1],
                                       account_id, run_cost)
            else:  # yes_no
                if args.dry_run:
                    q = plan["template"].format(entity=plan["pair"][0],
                                                company=company, desc=desc)
                    print(f"  {cid} [{fact_type}] yes_no: {q}  "
                          f"(false entity: {plan['pair'][1]})")
                    continue
                result = probe_yes_no(client, args.model, plan["template"],
                                      {"company": company, "desc": desc},
                                      plan["pair"][0], plan["pair"][1],
                                      account_id, run_cost)
            rec = {**base, **result}
            records.append(rec)
            pt = f"{rec['p_true']:.3f}" if rec["p_true"] is not None else "-"
            pf = f"{rec['p_false']:.3f}" if rec["p_false"] is not None else "-"
            print(f"  {cid} [{fact_type}] {plan['probe_type']}: "
                  f"answer={rec['model_answer']!r} knows={rec['knows_true_fact']} "
                  f"p_true={pt} p_false={pf}")

    if args.dry_run:
        print("(dry run: no API calls, nothing written)")
        return
    merge_write(records)
    print(f"\nWrote {len(records)} record(s) -> {OUT_PATH}")
    print(f"Run cost: ${sum(run_cost):.4f} over {len(run_cost)} call(s) "
          f"(logged to data/cache/cost_log.jsonl)")


if __name__ == "__main__":
    main()
