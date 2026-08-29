"""
Stage 3 (provider variant): Web Search Verification — Exa search + gpt-4o verdict.

Adapted from the earlier pipeline's verification_exa.py — see PROVENANCE.md (#13).
Second retrieval provider for the recall-probe robustness run. The verdict
stage is IMPORTED from verification_sonar (same prompt, model, temperature,
date clause), so the UNCAPPED Sonar-vs-Exa contrast isolates the retrieval
provider; capped mode additionally differs in date-filter semantics (below).

Each of the N runs:
  (a) one gpt-4o-mini call turns the claim into a tight search query, with
      one of 5 rotated angle framings (Exa takes queries, not prompts);
  (b) one Exa /search call (type=auto, 5 results, page text) — search only,
      NO verdict;
  (c) one gpt-4o temperature-0 verdict call over the formatted results —
      the same stage the Sonar path uses (verification_sonar._gpt4o_verdict).
      The verdict stage runs on every probe, including zero-result searches,
      matching the Sonar arm's always-on verdict.

The date cutoff is one value: `search_before_date` (%m/%d/%Y or None). It is
threaded to all three stages: the query-generation prompt (shared date
clause), Exa's `endPublishedDate` filter — set to the end of the PREVIOUS
day, since Exa's bound is inclusive while the Sonar arm excludes the cutoff
day — and the shared verdict prompt. None = uncapped (no filter, no prompt
lines).

CAPPED-MODE SEMANTICS (disclose wherever capped Exa numbers are reported):
Exa's endPublishedDate filter is NOT trustworthy on its own — the 2026-06-12
smoke test returned an UNDATED page whose content was published 2025-01-08
under a 2024-02-20 cutoff (temporal leakage). Capped mode therefore enforces
the cutoff client-side and FAIL-CLOSED: results lacking published-date
metadata, or dated on/after the cutoff day, are dropped before the verdict
stage. Capped-mode Exa recall is thus a structural LOWER BOUND (dated-pages
only) and is not directly comparable to capped Sonar recall (Perplexity
dates pages internally, with different coverage). The clean provider
contrast is the uncapped mode.
"""

import json
import re
import time
from datetime import datetime, timedelta

import requests

import costlog
from . import config
from .models import EvidenceRun, RoutedClaim
from .retry import with_retry
from .verification_sonar import (
    _date_clause,
    _domain,
    _gpt4o_verdict,
    _safe_verdict,
    openai_client,
)

ANGLE_PROMPTS = [
    "Generate a search query to find EVIDENCE FOR OR AGAINST this claim.",
    "Generate a search query to find the LATEST DATA about this claim.",
    "Generate a search query to find OFFICIAL SOURCES about this claim.",
    "Generate a search query to FACT-CHECK this claim.",
    "Generate a search query to find NEWS REPORTS about this claim.",
]

QUERY_SYSTEM = """You are a search query specialist.

Convert a claim into a tight, effective web search query of 4-7 words.
The query must be optimised for finding factual evidence about the claim.

You will also receive context about where the claim comes from.
Use the context to make the query more specific — add industry, country,
or domain keywords that a researcher would use.

Rules:
- Strip filler words ("it is", "there is", "the fact that")
- Keep the core factual assertion and any specific numbers, names, dates
- Use context to add domain-specific keywords (country, industry, sector)
- Use keywords a journalist or researcher would search for
- Do NOT use quotes, boolean operators, or site: filters
- Return ONLY the search query — nothing else, no explanation

Examples:
  Context: startup=acme-robotics, category=problem, page=4
  Claim: "There is a strong labour shortage"
  Query: warehouse automation labour shortage

  Context: startup=acme-robotics, category=market, page=7
  Claim: "The global cloud-accounting market is expected to grow by a factor of six by 2030"
  Query: global cloud accounting market growth forecast 2030

  Context: startup=acme-robotics, category=solution, page=7
  Claim: "The EU approves regulation requiring reusable packaging quotas from 2030"
  Query: EU packaging regulation reusable quotas 2030

  Context: startup=acme-robotics, category=finance, page=10
  Claim: "Acme Robotics has raised 7 million euros in pre-seed funding"
  Query: Acme Robotics pre-seed funding 7 million

  Context: startup=acme-robotics, category=market, page=5
  Claim: "The subscription model has penetrated the music sector on a large scale"
  Query: music streaming subscription market penetration"""


def verify_claim(
    routed_claim: RoutedClaim,
    n_runs: int = config.N_VERIFICATION_RUNS,
    startup_description: str = "",
    search_before_date: str | None = None,
) -> list[EvidenceRun]:
    """Run N independent verification passes (rotated angle framings).

    Same signature and return type as verification_sonar.verify_claim, so the
    two modules are drop-in interchangeable behind a --provider switch.
    """
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
        query, query_source = _generate_search_query(
            claim_text, run_index, context, search_before_date)
        results, cost_dollars = _exa_search(query, search_before_date)

        # Zero results still go through the shared verdict stage (it renders
        # empty evidence as "(empty)"), matching the Sonar arm's structure.
        evidence_text = _format_evidence(results)
        citations = [r["url"] for r in results if r["url"]]
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
                "query": query,
                "query_source": query_source,
                "citations": citations,
                "verdict": verdict_data,
                "exa_cost_dollars": cost_dollars,
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


# ── (a) Query generation (gpt-4o-mini, temperature 0) ────────────────────────

def _generate_search_query(
    claim_text: str,
    run_index: int,
    context: str,
    search_before_date: str | None,
) -> tuple[str, str]:
    """Returns (query, source) where source is "llm" or "fallback".

    In capped mode the shared date clause is appended to the system prompt so
    the retrieval-side LLM knows the cutoff, mirroring the Sonar arm's search
    prompt (one date value drives prompt AND filter — PROVENANCE #2/#13).
    """
    angle = ANGLE_PROMPTS[run_index % len(ANGLE_PROMPTS)]
    user_message = (
        f"Context: {context}\n"
        f"Claim: \"{claim_text}\"\n\n"
        f"{angle}"
    )
    try:
        t0 = time.time()
        response = with_retry(lambda: openai_client.chat.completions.create(
            model=config.EXTRACT_MODEL,
            max_tokens=50,
            temperature=0,
            messages=[
                {"role": "system",
                 "content": QUERY_SYSTEM + _date_clause(search_before_date)},
                {"role": "user", "content": user_message},
            ],
        ))
        costlog.log_openai_response(
            response, stage="exa_query", model=config.EXTRACT_MODEL,
            latency_s=time.time() - t0)
        query = (response.choices[0].message.content or "").strip().strip('"')
        words = query.split()
        if len(words) > 10:
            query = " ".join(words[:8])
        if query:
            return query, "llm"
        return _fallback_query(claim_text), "fallback"
    except Exception:
        return _fallback_query(claim_text), "fallback"


def _fallback_query(claim_text: str) -> str:
    clean = re.sub(r"[^\w\s]", " ", claim_text)
    return " ".join(clean.split()[:7])


# ── (b) Exa search ────────────────────────────────────────────────────────────

def _exa_end_published_date(search_before_date: str) -> str:
    """%m/%d/%Y -> ISO 8601 end of the PREVIOUS day.

    Exa's endPublishedDate bound is inclusive and date-only page metadata
    normalizes to midnight, so an end-of-previous-day bound excludes pages
    published on the cutoff day — matching the Sonar arm's "before {date}"
    semantics."""
    day_before = (datetime.strptime(search_before_date, "%m/%d/%Y")
                  - timedelta(days=1))
    return day_before.strftime("%Y-%m-%dT23:59:59.999Z")


def _published_before(published: str, search_before_date: str) -> bool:
    """True iff the page carries published-date metadata strictly before the
    cutoff DAY. Undated pages fail closed (False): the API-side
    endPublishedDate filter empirically passes undated pages through
    (smoke test 2026-06-12: undated page with 2025-01-08 content returned
    under a 2024-02-20 cutoff), so capped mode must re-enforce client-side.
    `published` is "YYYY-MM-DD" or ""; string comparison is date order."""
    if not published:
        return False
    cutoff = (datetime.strptime(search_before_date, "%m/%d/%Y")
              .strftime("%Y-%m-%d"))
    return published < cutoff


def _estimated_fee(data: dict) -> float:
    # Fallback if the API response carries no costDollars:
    # $7/1k searches (<=10 results) + $1/1k pages of text content.
    return 0.007 + 0.001 * len(data.get("results") or [])


def _exa_search(
    query: str,
    search_before_date: str | None,
    num_results: int = config.EXA_NUM_RESULTS,
) -> tuple[list[dict], dict]:
    """One Exa call: returns (results, costDollars). Search only, no verdict."""
    if not config.EXA_API_KEY:
        raise RuntimeError("EXA_API_KEY is not set in the environment")

    payload = {
        "query": query,
        "type": "auto",
        "numResults": num_results,
        "contents": {"text": {"maxCharacters": config.EXA_TEXT_MAX_CHARS}},
    }
    if search_before_date:
        payload["endPublishedDate"] = _exa_end_published_date(
            search_before_date)

    def _post():
        resp = requests.post(
            config.EXA_SEARCH_URL,
            headers={"x-api-key": config.EXA_API_KEY,
                     "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.raise_for_status()  # retryable: with_retry backs off
        return resp

    t0 = time.time()
    response = with_retry(_post)
    if not response.ok:  # deterministic 4xx: fail once, keep the error body
        raise RuntimeError(
            f"Exa HTTP {response.status_code}: {response.text[:300]}")
    data = response.json()

    cost_dollars = data.get("costDollars") or {}
    try:
        fee = float(cost_dollars.get("total") or 0.0)
    except (TypeError, ValueError):
        fee = 0.0
    if fee <= 0.0:
        fee = _estimated_fee(data)
    costlog.log_call("exa", 0, 0, stage="exa_search",
                     latency_s=time.time() - t0, request_fee_usd=fee)

    results = []
    n_dropped = 0
    for item in data.get("results", []):
        published = (item.get("publishedDate") or "")[:10]
        if search_before_date and not _published_before(
                published, search_before_date):
            n_dropped += 1  # fail-closed cutoff enforcement (see docstring)
            continue
        results.append({
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "published": published,
            "content": (item.get("text") or "")[:config.EXA_TEXT_MAX_CHARS],
        })
    if n_dropped:
        cost_dollars = {**cost_dollars, "n_dropped_by_cutoff": n_dropped}
    return results, cost_dollars


def _format_evidence(results: list[dict]) -> str:
    """Number the results with title, URL, published date and a text excerpt.

    Excerpts are capped at 500 chars/result so the verdict model reasons over
    a context comparable in size to Sonar's synthesized evidence text.
    """
    lines = []
    for i, r in enumerate(results, 1):
        head = f"[{i}] {r['title']}"
        if r["published"]:
            head += f" (published {r['published']})"
        lines.append(head)
        lines.append(f"    URL: {r['url']}")
        if r["content"]:
            lines.append(f"    {r['content'][:500].strip()}")
    return "\n".join(lines)
