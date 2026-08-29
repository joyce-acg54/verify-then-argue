"""
Stage 1: L1 Extraction Layer — OpenAI gpt-4o-mini
L1A: page gate, L1B: claim extraction + atomization, L1C: merged audit
(support check + quality audit, with programmatic quote-span verification).

Adapted from the earlier pipeline's l1_extraction.py — see PROVENANCE.md
(#1 token limits, #5 stage label, #6 real OpenAI client, #10 dropped claims).
"""

import json
import uuid

from openai import OpenAI

import costlog
from . import config
from .models import AtomicClaim, ClaimCategory, ClaimScope, ClaimType, Page
from .retry import with_retry

client = OpenAI(api_key=config.OPENAI_API_KEY)


# ── L1A: Page Gate ────────────────────────────────────────────────────────────

L1A_SYSTEM = """You are a document analyst for a startup pitch deck verifier.
Decide if a page contains enough factual content to extract verifiable claims from.

Respond ONLY with a JSON object — no preamble, no markdown fences:
{
  "should_extract": true | false,
  "signal": "high" | "medium" | "low",
  "reason": "<one sentence>"
}

Do NOT extract: cover pages, table of contents, pure image pages, legal disclaimers, thank-you slides.
DO extract: traction slides, market size, team credentials, product claims, financial projections."""


def run_l1a(page: Page) -> Page:
    if not page.page_text.strip():
        page.should_extract = False
        page.gate_reason = "Empty page"
        return page

    response = with_retry(lambda: client.chat.completions.create(
        model=config.EXTRACT_MODEL,
        max_tokens=config.MAX_TOKENS_GATE,
        temperature=0,
        messages=[
            {"role": "system", "content": L1A_SYSTEM},
            {"role": "user",   "content": f"Page {page.page_number}:\n\n{page.page_text[:3000]}"}
        ]
    ))
    costlog.log_openai_response(response, stage="l1a_gate", model=config.EXTRACT_MODEL)

    raw  = response.choices[0].message.content or ""
    data = _safe_json(raw)

    page.should_extract = data.get("should_extract", False)
    page.gate_reason    = data.get("reason", "")
    return page


# ── L1B: Claim Extraction (the atomizer) ─────────────────────────────────────

L1B_SYSTEM = """You are a claim extraction specialist for a fact-checking system.

Extract every ATOMIC, VERIFIABLE claim from the page.
An atomic claim = one single fact, self-contained, independently falsifiable.
Split compound sentences: "X and Y" → two claims.
Exclude inferences, opinions, and vague statements.

For each claim:
- category: problem | solution | market | finance | operations | team | other
- speaker: "company" | "other"
- scope: "internal" (needs company data) | "external" (public sources) | "mixed"
- claim_type: numeric | factual | comparative | forecast | normative

Respond ONLY with a JSON array — no preamble, no markdown fences:
[
  {
    "claim_text": "<standalone sentence>",
    "category": "...",
    "speaker": "company|other",
    "scope": "internal|external|mixed",
    "claim_type": "..."
  }
]

Return [] if no verifiable claims exist."""


def run_l1b(page: Page) -> list[AtomicClaim]:
    response = with_retry(lambda: client.chat.completions.create(
        model=config.EXTRACT_MODEL,
        max_tokens=config.MAX_TOKENS_EXTRACT,
        temperature=0,
        messages=[
            {"role": "system", "content": L1B_SYSTEM},
            {"role": "user",   "content": f"Extract atomic claims:\n\n{page.page_text[:4000]}"}
        ]
    ))
    costlog.log_openai_response(response, stage="l1b_extract", model=config.EXTRACT_MODEL)

    raw  = response.choices[0].message.content or ""
    data = _safe_json(raw, default=[])

    claims = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("claim_text", "").strip()
        if not text:
            continue
        claims.append(AtomicClaim(
            claim_id=str(uuid.uuid4()),
            claim_text=text,
            source_page=page.page_number,
            source_file=page.source_file,
            startup_id=page.startup_id,
            category=_safe_enum(ClaimCategory, item.get("category"), ClaimCategory.OTHER),
            speaker=item.get("speaker", "company"),
            scope=_safe_enum(ClaimScope, item.get("scope"), ClaimScope.MIXED),
            claim_type=_safe_enum(ClaimType, item.get("claim_type"), ClaimType.FACTUAL),
        ))
    return claims


# ── L1C: Merged Audit ────────────────────────────────────────────────────────

L1C_SYSTEM = """You are a claim auditor for a fact-checking system.

For each claim, perform a two-part audit in one pass:

PART 1 — Support check (was this actually said on the page?):
- supported: page explicitly states this fact
- contradicted: page says something contradicting this
- not_in_text: cannot be traced to the page

PART 2 — Quality check (is this a valid atomic claim?):
- ok: valid standalone verifiable fact
- questionable: compound, ambiguous, or borderline
- wrong: not verifiable (inference, opinion, empty)

For supported claims, find an exact verbatim quote from the page.
Do NOT invent quotes — only quote text that literally appears in the page.

Respond ONLY with a JSON array — no preamble, no markdown fences:
[
  {
    "claim_id": "...",
    "support_label": "supported|contradicted|not_in_text",
    "support_span": "<exact quote or empty string>",
    "support_confidence": 0.0-1.0,
    "audit_label": "ok|questionable|wrong",
    "audit_reason": "<one sentence>"
  }
]"""


def run_l1c(page: Page, claims: list[AtomicClaim]) -> tuple[list[AtomicClaim], list[AtomicClaim]]:
    """Returns (valid, dropped) — dropped = audit_label 'wrong'."""
    if not claims:
        return [], []

    payload = [{"claim_id": c.claim_id, "claim_text": c.claim_text} for c in claims]

    response = with_retry(lambda: client.chat.completions.create(
        model=config.EXTRACT_MODEL,
        max_tokens=config.MAX_TOKENS_AUDIT,
        temperature=0,
        messages=[
            {"role": "system", "content": L1C_SYSTEM},
            {"role": "user", "content": (
                f"Page text:\n{page.page_text[:3000]}\n\n"
                f"Claims:\n{json.dumps(payload, indent=2)}"
            )}
        ]
    ))
    costlog.log_openai_response(response, stage="l1c_audit", model=config.EXTRACT_MODEL)

    raw = response.choices[0].message.content or ""
    results = _safe_json(raw, default=[])
    audit_map = {r["claim_id"]: r for r in results
                 if isinstance(r, dict) and "claim_id" in r}

    valid, dropped = [], []
    for claim in claims:
        r = audit_map.get(claim.claim_id, {})
        claim.support_label      = r.get("support_label", "not_in_text")
        try:
            claim.support_confidence = float(r.get("support_confidence", 0.0))
        except (TypeError, ValueError):
            claim.support_confidence = 0.0
        claim.audit_label        = r.get("audit_label", "questionable")
        claim.audit_reason       = r.get("audit_reason", "")

        # Programmatic span verification — reject hallucinated quotes
        raw_span = r.get("support_span", "")
        claim.support_span = raw_span if raw_span and raw_span in page.page_text else ""

        if claim.audit_label != "wrong":
            valid.append(claim)
        else:
            dropped.append(claim)

    return valid, dropped


# ── Full L1 pipeline per page ─────────────────────────────────────────────────

def process_page(page: Page) -> tuple[Page, list[AtomicClaim], list[AtomicClaim]]:
    """Returns (gated page, valid claims, audited-out claims)."""
    page = run_l1a(page)
    if not page.should_extract:
        return page, [], []
    claims = run_l1b(page)
    if not claims:
        return page, [], []
    valid, dropped = run_l1c(page, claims)
    return page, valid, dropped


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_json(raw: str, default=None):
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return default if default is not None else {}


def _safe_enum(enum_class, value, default):
    if value is None:
        return default
    try:
        return enum_class(str(value).lower())
    except ValueError:
        return default
