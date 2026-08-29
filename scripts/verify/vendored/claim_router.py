"""
Stage 2b: Claim Router — OpenAI gpt-4o-mini

Routes claims by verifiability in batches of 10 with deterministic credal
intervals per label (the LLM only returns a label).

Adapted from the earlier pipeline's claim_router.py — see PROVENANCE.md
(#3 neutral few-shot examples + typo fix, #6 real OpenAI client).
"""

import json

from openai import OpenAI

import costlog
from . import config
from .models import AtomicClaim, ClaimScope, ClaimType, RoutedClaim, VerifiabilityLabel
from .retry import with_retry

client = OpenAI(api_key=config.OPENAI_API_KEY)

BATCH_SIZE    = 10    # claims per LLM call — keeps response well within token limits
ROUTER_TOKENS = 2000  # enough for 10 claims with labels + reasons

# FIX #3: few-shot examples replaced with neutral, fictional ones (the source
# used real eval-company names and country-specific market facts); typo fixed.
ROUTER_SYSTEM = """You are a claim routing specialist for a fact-checking system.

For each claim, classify its verifiability — can it be checked against public web sources?

Labels (choose exactly one per claim):
- verifiable:   a definite answer exists in public sources (news, gov data, company filings)
- unverifiable: requires private internal company data only the startup has
- inference:    a logical conclusion drawn from other facts, not a checkable fact itself
- normative:    a value judgment or opinion — has no factual true/false answer

Examples:
  "The global cloud accounting market is worth $12B"          → verifiable   (public market data exists)
  "Our MRR grew 40% last quarter"                             → unverifiable (internal metric)
  "Therefore our TAM is large"                                → inference    (conclusion from other claims)
  "We are the best solution on the market"                    → normative    (value judgment)
  "Acme Robotics raised $7M in seed funding"                  → verifiable   (public funding record)
  "Over 3 million industrial robots are in operation worldwide" → verifiable (industry data exists)
  "The EU approved new packaging waste regulations in 2024"   → verifiable   (public record)

When in doubt, lean toward VERIFIABLE — it is better to attempt verification
and find no evidence than to skip a checkable claim. Published forecasts and
cited statistics are verifiable, not inference.

Respond ONLY with a JSON array — no preamble, no markdown fences:
[
  {
    "claim_id": "<exact id from input>",
    "verifiability": "verifiable|unverifiable|inference|normative",
    "reason": "<one sentence>"
  }
]"""

# Deterministic credal intervals per label
CREDAL_INTERVALS: dict[str, tuple[float, float]] = {
    "verifiable":   (0.85, 0.95),
    "unverifiable": (0.05, 0.20),
    "inference":    (0.10, 0.30),
    "normative":    (0.05, 0.15),
    "_fallback":    (0.70, 0.90),  # LLM call failed but defaulting to verifiable
}


def route_claims(
    claims: list[AtomicClaim],
) -> tuple[list[RoutedClaim], list[RoutedClaim], list[RoutedClaim]]:
    """Returns (to_verify, flagged, borderline)."""
    if not claims:
        return [], [], []

    # Rule-based pre-routing
    pre_routed: dict[str, VerifiabilityLabel] = {}
    for c in claims:
        if c.claim_type == ClaimType.NORMATIVE:
            pre_routed[c.claim_id] = VerifiabilityLabel.NORMATIVE
        elif c.scope == ClaimScope.INTERNAL and c.claim_type == ClaimType.NUMERIC:
            pre_routed[c.claim_id] = VerifiabilityLabel.UNVERIFIABLE

    # LLM routing in batches
    needs_llm = [c for c in claims if c.claim_id not in pre_routed]
    llm_labels: dict[str, str] = {}
    for i in range(0, len(needs_llm), BATCH_SIZE):
        batch = needs_llm[i:i + BATCH_SIZE]
        llm_labels.update(_route_batch(batch))

    # Assemble + bucket
    to_verify, flagged, borderline = [], [], []
    for claim in claims:
        if claim.claim_id in pre_routed:
            verif = pre_routed[claim.claim_id]
            p_lower, p_upper = CREDAL_INTERVALS[verif.value]
            flagged.append(RoutedClaim(
                claim=claim, verifiability=verif,
                verifiability_lower=p_lower, verifiability_upper=p_upper,
            ))
            continue

        label_str = llm_labels.get(claim.claim_id)
        if label_str is None:
            verif = VerifiabilityLabel.VERIFIABLE
            p_lower, p_upper = CREDAL_INTERVALS["_fallback"]
        else:
            verif = _safe_verif(label_str)
            p_lower, p_upper = CREDAL_INTERVALS.get(verif.value, CREDAL_INTERVALS["_fallback"])

        routed = RoutedClaim(
            claim=claim, verifiability=verif,
            verifiability_lower=p_lower, verifiability_upper=p_upper,
        )
        if (p_upper - p_lower) > config.CREDAL_AMBIGUITY_THRESHOLD:
            borderline.append(routed)
        elif verif == VerifiabilityLabel.VERIFIABLE:
            to_verify.append(routed)
        else:
            flagged.append(routed)

    return to_verify, flagged, borderline


def _route_batch(batch: list[AtomicClaim]) -> dict[str, str]:
    payload = [
        {
            "claim_id":   c.claim_id,
            "claim_text": c.claim_text,
            "scope":      c.scope.value if hasattr(c.scope, "value") else str(c.scope),
            "claim_type": c.claim_type.value if hasattr(c.claim_type, "value") else str(c.claim_type),
        }
        for c in batch
    ]
    try:
        response = with_retry(lambda: client.chat.completions.create(
            model=config.EXTRACT_MODEL,
            max_tokens=ROUTER_TOKENS,
            temperature=0,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user",   "content": (
                    f"Route these {len(batch)} claims:\n"
                    f"{json.dumps(payload, indent=2)}"
                )}
            ]
        ))
        costlog.log_openai_response(response, stage="router", model=config.EXTRACT_MODEL)
        raw     = response.choices[0].message.content or ""
        results = _safe_json(raw, default=[])
        return {
            r["claim_id"]: r["verifiability"]
            for r in results
            if isinstance(r, dict) and "claim_id" in r and "verifiability" in r
        }
    except Exception as e:
        print(f"  Warning: router batch failed ({e}), defaulting batch to verifiable")
        return {}


def _safe_verif(value: str) -> VerifiabilityLabel:
    try:
        return VerifiabilityLabel(value.lower())
    except ValueError:
        return VerifiabilityLabel.VERIFIABLE


def _safe_json(raw: str, default=None):
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return default if default is not None else {}
