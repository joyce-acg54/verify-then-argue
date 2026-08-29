"""
Stage 3: Web Search Verification — Perplexity Sonar search + gpt-4o verdict.

Adapted from the earlier pipeline's verification_sonar.py — see PROVENANCE.md
(#2 single search_before_date parameter, #7 search/verdict split).

Each of the N runs:
  (a) one Perplexity `sonar` call with one of 5 rotated angle framings —
      returns evidence text + citations (search only, NO verdict);
  (b) one OpenAI gpt-4o call at temperature 0 that reasons over the Sonar
      evidence and emits the verdict JSON (reasoning before verdict).

The date cutoff is one value: `search_before_date` (%m/%d/%Y or None).
It is passed BOTH as Perplexity's `search_before_date_filter` AND interpolated
into the Sonar/verdict prompts, so prompt and filter cannot disagree.
None = uncapped (no filter, no prompt line).
"""

import json
import re
import time
from urllib.parse import urlparse

from openai import OpenAI

import costlog
from . import config
from .models import EvidenceRun, RoutedClaim
from .retry import with_retry

sonar_client = OpenAI(
    api_key=config.PERPLEXITY_API_KEY,
    base_url=config.PERPLEXITY_BASE_URL,
)
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

ANGLE_PROMPTS = [
    "Search for evidence FOR OR AGAINST this claim.",
    "Search for the LATEST DATA about this claim.",
    "Search for OFFICIAL SOURCES about this claim.",
    "FACT-CHECK this claim using web sources.",
    "Search for NEWS REPORTS about this claim.",
]

SONAR_SEARCH_SYSTEM = """You are a web research assistant for a fact-checking system.
You will be given a factual claim from a startup pitch deck and some context.
Search the web and report what the best available sources say that is relevant
to the claim — both supporting and contradicting evidence. Do NOT give a verdict.
Quote concrete figures, dates, and names, and say which source each came from.
If you find nothing relevant, say exactly that.{date_clause}"""

VERDICT_SYSTEM = """You are a rigorous fact-checker. You will be given a factual claim from a
startup pitch deck and the output of a web search assistant (evidence summary
plus a list of source URLs). Assess whether the evidence supports or refutes
the claim. Use ONLY the evidence provided — do not rely on your own knowledge
of the company.{date_clause}

Verdict options:
- "supports":     evidence clearly supports the claim
- "refutes":      evidence clearly contradicts the claim
- "insufficient": sources found but inconclusive
- "no_evidence":  no relevant sources found

Respond ONLY with a JSON object — no preamble, no markdown fences:
{{
  "reasoning": "<2-3 sentences citing specific sources>",
  "verdict": "supports|refutes|insufficient|no_evidence",
  "primary_source_url": "<URL of most relevant source, or empty string>",
  "source_domain": "<domain only e.g. reuters.com, or empty string>"
}}"""


def verify_claim(
    routed_claim: RoutedClaim,
    n_runs: int = config.N_VERIFICATION_RUNS,
    startup_description: str = "",
    search_before_date: str | None = None,
) -> list[EvidenceRun]:
    """Run N independent verification passes (rotated angle framings)."""
    return [
        _single_run(routed_claim, i, startup_description, search_before_date)
        for i in range(n_runs)
    ]


def _single_run(
    routed_claim: RoutedClaim,
    run_index: int,
    startup_description: str,
    search_before_date: str | None,
) -> EvidenceRun:
    claim_text = routed_claim.claim.claim_text
    context = (
        f"startup={routed_claim.claim.startup_id} | "
        f"description={startup_description} | "
        f"page={routed_claim.claim.source_page} | "
        f"category={routed_claim.claim.category.value} | "
        f"claim_type={routed_claim.claim.claim_type.value}"
    )

    try:
        evidence_text, citations = _sonar_search(
            claim_text, context, run_index, search_before_date)
        verdict_data = _gpt4o_verdict(
            claim_text, context, evidence_text, citations, search_before_date)

        source_url    = (verdict_data.get("primary_source_url") or "").strip()
        source_domain = (verdict_data.get("source_domain") or "").strip() or _domain(source_url)
        if not source_url and citations:
            source_url = citations[0]
            source_domain = source_domain or _domain(source_url)

        return EvidenceRun(
            run_index=run_index,
            evidence_text=evidence_text,
            source_url=source_url,
            source_domain=source_domain,
            source_tier=config.tier_for_domain(source_domain),
            verdict=_safe_verdict(verdict_data.get("verdict", "no_evidence")),
            reasoning=verdict_data.get("reasoning", ""),
            raw_response=json.dumps({
                "context": context,
                "angle": ANGLE_PROMPTS[run_index % len(ANGLE_PROMPTS)],
                "citations": citations,
                "verdict": verdict_data,
            }, ensure_ascii=False),
        )
    except Exception as e:
        return EvidenceRun(
            run_index=run_index,
            evidence_text="",
            source_url="",
            source_domain="",
            source_tier=config.UNKNOWN_TIER,
            verdict="api_error",
            reasoning=f"Verification failed: {e}",
        )


# ── (a) Sonar search ─────────────────────────────────────────────────────────

def _date_clause(search_before_date: str | None) -> str:
    if not search_before_date:
        return ""
    return (f"\nIMPORTANT: Only use and cite sources published before "
            f"{search_before_date} (MM/DD/YYYY). Ignore anything published "
            f"on or after this date.")


def _sonar_search(
    claim_text: str,
    context: str,
    run_index: int,
    search_before_date: str | None,
) -> tuple[str, list[str]]:
    """One Sonar call: returns (evidence text, citation URLs)."""
    angle = ANGLE_PROMPTS[run_index % len(ANGLE_PROMPTS)]
    user_message = (
        f"Context: {context}\n"
        f"Claim: \"{claim_text}\"\n\n"
        f"{angle}\n"
        f"Report the relevant evidence with sources."
    )
    kwargs = dict(
        model=config.SONAR_MODEL,
        max_tokens=config.MAX_TOKENS_SONAR,
        messages=[
            {"role": "system",
             "content": SONAR_SEARCH_SYSTEM.format(
                 date_clause=_date_clause(search_before_date))},
            {"role": "user", "content": user_message},
        ],
    )
    # FIX #2: single date parameter drives BOTH the prompt and the API filter.
    if search_before_date:
        kwargs["extra_body"] = {"search_before_date_filter": search_before_date}

    t0 = time.time()
    response = with_retry(lambda: sonar_client.chat.completions.create(**kwargs))
    costlog.log_openai_response(
        response, stage="sonar_search", model=config.SONAR_MODEL,
        latency_s=time.time() - t0,
        request_fee_usd=costlog.SONAR_SEARCH_FEE_USD,
    )

    content = response.choices[0].message.content or ""
    citations = _extract_citations(response)
    return content, citations


def _extract_citations(response) -> list[str]:
    """Perplexity returns extra fields (citations / search_results) that the
    openai client surfaces via model_extra."""
    urls: list[str] = []
    extra = getattr(response, "model_extra", None) or {}
    for c in extra.get("citations") or []:
        if isinstance(c, str):
            urls.append(c)
    for r in extra.get("search_results") or []:
        if isinstance(r, dict) and r.get("url"):
            urls.append(r["url"])
    # de-dup, preserve order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ── (b) gpt-4o verdict (temperature 0) ───────────────────────────────────────

def _gpt4o_verdict(
    claim_text: str,
    context: str,
    evidence_text: str,
    citations: list[str],
    search_before_date: str | None,
) -> dict:
    cite_block = "\n".join(f"- {u}" for u in citations) or "(no sources returned)"
    user_message = (
        f"Context: {context}\n"
        f"Claim: \"{claim_text}\"\n\n"
        f"Web search evidence:\n{evidence_text or '(empty)'}\n\n"
        f"Sources cited by the search assistant:\n{cite_block}\n\n"
        f"Return ONLY a JSON object with keys: reasoning, verdict, "
        f"primary_source_url, source_domain."
    )
    t0 = time.time()
    response = with_retry(lambda: openai_client.chat.completions.create(
        model=config.VERDICT_MODEL,
        max_tokens=config.MAX_TOKENS_VERDICT,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
             "content": VERDICT_SYSTEM.format(
                 date_clause=_date_clause(search_before_date))},
            {"role": "user", "content": user_message},
        ],
    ))
    costlog.log_openai_response(
        response, stage="verdict_gpt4o", model=config.VERDICT_MODEL,
        latency_s=time.time() - t0,
    )
    raw = response.choices[0].message.content or ""
    return _extract_json(raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _safe_verdict(value: str) -> str:
    return value if value in config.VERDICT_LABELS else "no_evidence"


def _extract_json(raw: str) -> dict:
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    matches = list(re.finditer(r"\{.*\}", clean, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group())
        except json.JSONDecodeError:
            pass
    return {}
