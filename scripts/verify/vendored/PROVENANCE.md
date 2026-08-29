# Vendored module provenance — claim verification pipeline

All content in this directory is adapted from the authors' earlier
verification pipeline, which is not publicly released. It is reproduced here
in full so that every stage of the verify-then-argue system is inspectable,
and every deviation is listed below.

| vendored file            | source file(s)                          |
|--------------------------|------------------------------------------|
| `models.py`              | `models.py`                              |
| `retry.py`               | `retry.py`                               |
| `config.py`              | `config.py`                              |
| `l1_extraction.py`       | `l1_extraction.py`                       |
| `claim_router.py`        | `claim_router.py`                        |
| `deduplication.py`       | `deduplication.py`                       |
| `verification_sonar.py`  | `verification_sonar.py`                  |
| `verification_exa.py`    | `verification_exa.py`                    |
| `source_scorer.py`       | `source_scorer.py`                       |
| `entropy_calculator.py`  | `entropy_calculator.py`                  |
| `adjudicator.py`         | `adjudicator.py`                         |

## Deliberate adaptations (documented for the paper)

### Bug fixes (verified in the source before changing)

1. **MAX_TOKENS=1000 truncation** (`config.py` -> `l1_extraction.py`): the source
   used one shared `MAX_TOKENS = 1000` for the page gate, the atomizer AND the
   audit. On dense pages the L1B/L1C JSON arrays get truncated mid-array,
   `_safe_json` fails silently and claims are dropped. Vendored config splits
   this into `MAX_TOKENS_GATE = 400`, `MAX_TOKENS_EXTRACT = 4000`,
   `MAX_TOKENS_AUDIT = 4000`.
2. **Contradictory hard-coded date cutoffs** (`verification_sonar.py`): source
   prompt said "june 6, 2024" while the API filter said `"06/01/2024"` (June 1).
   Replaced with a single `search_before_date: str | None` parameter
   (`%m/%d/%Y`, Perplexity param `search_before_date_filter`) threaded through
   from the CLI. The SAME value is interpolated into the prompt text and the
   API filter so they cannot disagree. `None` = uncapped (no filter line, no
   `extra_body`).
3. **Market- and company-specific few-shot examples** (`claim_router.py`; the
   same examples also exist in `verification.py` / `verification_exa.py` query
   prompts, which are not vendored): the source's few-shot examples quoted
   real market statistics, real legislation debates, and named real companies
   from the source pipeline's evaluation set (names withheld here for
   anonymity). All replaced with neutral, fictional examples (Acme Robotics,
   global cloud-accounting market, EU packaging regulation). Also fixed the
   router prompt typo "...checkable claim.published forecasts..." ->
   "...checkable claim. Published forecasts and cited statistics are
   verifiable, not inference."
4. **Country-biased source-tier taxonomy** (`config.py`): tier structure kept
   (1 gov/academic, 2 established press, 3 industry data, 4 blogs/unknown) but
   the example lists are generalized: single-country government domains and
   real eval-company corporate domains and niche sector blogs removed;
   generic TLD rules (`.gov`, `.edu`, `.int`, `.mil`, `.ac.*`, `gov.*`) and
   international press/aggregators kept. Domain matching hardened from
   `kw in domain` substring matching to suffix/label matching (the source
   matched "gov" anywhere in the domain, so e.g. `governanceblog.com` scored
   Tier 1).
5. **Mislabeled cost stage** (`l1_extraction.py`): `run_l1b` logged its cost as
   "L1B page gate" (copy-paste from L1A); now "L1B extraction".
6. **Wrong client/docstring** (`l1_extraction.py`, `claim_router.py`,
   `entropy_calculator.py`): source pointed the "OpenAI" client at Groq with
   `openai/gpt-oss-120b` while docstrings claimed gpt-4o. Vendored code uses
   the real OpenAI API: extraction/audit/routing on `gpt-4o-mini`, verdict
   reasoning on `gpt-4o` (temperature 0), per the experiment spec.

### Architectural adaptations

7. **Sonar split into search + separate verdict** (`verification_sonar.py`):
   the source had Sonar produce the verdict itself. Per the experiment spec,
   each run is now (a) one Perplexity `sonar` search call with one of 5 rotated
   angle framings that returns evidence text + citations, then (b) one
   `gpt-4o` temperature-0 call that reasons over that evidence and emits the
   verdict JSON (reasoning placed BEFORE verdict, fixing the source's ordering
   note). Verdict labels are a shared constant in `config.py`
   (`VERDICT_LABELS`), fixing the "retyped in 3 files" drift hazard.
8. **Entropy: direct verdict clustering** (`entropy_calculator.py`): the source
   used an LLM call to cluster free-text reasonings into the same four verdict
   labels. Because the gpt-4o verdict step already emits canonical labels at
   temperature 0, the LLM clustering step is removed; clusters are the verdict
   labels themselves ("verdict-cluster entropy"). Shannon entropy is still
   normalized by log(4). Aleatoric/epistemic decomposition kept unchanged.
9. **Adjudicator simplified** (`adjudicator.py`): conformal calibration file
   I/O removed (no calibration set exists for this corpus); the prediction-set
   construction, nonconformity scores, label priority and confidence logic are
   kept verbatim with the default threshold 0.5 now an explicit config value
   (`CONFORMAL_DEFAULT_THRESHOLD`). `api_error` early-exit kept.
10. **L1C returns dropped claims too** (`l1_extraction.py`): `run_l1c` returns
    `(valid, dropped)` so the caller can report audited-out counts. The
    programmatic quote-span check (reject spans not literally in the page
    text) is kept verbatim.
11. **Cost logging** (`all`): the source's in-memory `cost_tracker` is replaced
    by `scripts/verify/costlog.py`, which appends records to
    `data/cache/cost_log.jsonl` in the exact format of
    `scripts/harness/cost.py` (loaded by file path, harness not modified),
    with prices added for `sonar` ($1/$1 per 1M tokens + $5/1k search
    requests, folded into `cost_usd`) and `text-embedding-3-small`
    ($0.02/1M input).
12. **Retry helper**: copied verbatim except the unicode warning glyphs in
    print statements were replaced with ASCII.
13. **Exa provider variant** (`verification_exa.py`, vendored 2026-06-12 for
    the recall-probe second-provider robustness run):
    - **Hard-coded date cutoff removed** (same class of bug as #2): the source
      hard-coded `endPublishedDate: "2023-01-03T00:00:00Z"` inside
      `_exa_search`. Replaced with the same `search_before_date: str | None`
      (`%m/%d/%Y`) parameter as the Sonar path, threaded to ALL stages:
      query-generation prompt (shared date clause), Exa `endPublishedDate`
      filter, and the shared verdict prompt; `None` = uncapped. The filter is
      set to the END OF THE PREVIOUS DAY because Exa's bound is inclusive
      while the Sonar arm's "before {date}" excludes the cutoff day.
      Semantics caveat: the OpenAPI spec reads as if undated pages are
      excluded by the filter, but the 2026-06-12 smoke test EMPIRICALLY
      showed an undated page with post-cutoff content (published 2025-01-08)
      returned under a 2024-02-20 cutoff — the API-side filter passes
      undated pages through. Capped mode therefore re-enforces the cutoff
      client-side, fail-closed (`_published_before`: results lacking
      published-date metadata, or dated on/after the cutoff day, are dropped
      before the verdict stage; drop counts recorded in raw_response as
      `n_dropped_by_cutoff`). Capped-mode Exa recall is thus a structural
      lower bound (dated pages only), not directly comparable to capped
      Sonar; the clean provider contrast is uncapped mode (disclosed in both
      docstrings). The two contaminated capped smoke records were moved to
      `recall_probe_results_exa.capped_leakage_evidence.jsonl` and kept as
      documented evidence of the leakage (part of the withheld canary run
      artifacts, not included in this release).
    - **Verdict stage replaced by the shared Sonar verdict** (extends #7):
      the source had its own verifier prompt with `verdict` ordered before
      `reasoning`, no date clause, no temperature pin and no JSON response
      format. The module now imports `verification_sonar._gpt4o_verdict`
      (gpt-4o, temperature 0, reasoning-first, shared `VERDICT_LABELS`,
      shared date clause), so the uncapped Sonar-vs-Exa contrast isolates the
      retrieval provider. The evidence block necessarily differs in kind
      (Sonar synthesizes; Exa returns raw page excerpts) — excerpts are
      capped at 500 chars/result x 5 results to keep verdict-context size
      comparable. The verdict stage runs on every probe, including
      zero-result searches (evidence rendered "(empty)"), matching the Sonar
      arm's always-on verdict. Known residual asymmetry (disclosed): the Exa
      arm is deterministic per (claim, angle) — temp-0 query gen + search —
      while Sonar samples at its default temperature, so verdict-cluster
      entropy (and thus the Ignorance/consistency signals) is not comparable
      across arms; the probe therefore also reports an adjudication-free
      run-level supports rate and a distinct-queries-per-probe statistic,
      and per-run queries (+ fallback flags) are persisted in the records.
    - **Market- and company-specific few-shots scrubbed** (same as #3): the
      query-generation examples (real company names and market-specific
      statistics, the same set as #3) replaced with
      the neutral fictional set (Acme Robotics, cloud-accounting market,
      EU packaging regulation).
    - **Substring tier matching dropped** (same as #4): the source's local
      `_tier_for_domain` (`kw in domain` over `config["examples"]`) replaced
      with the hardened shared `config.tier_for_domain`.
    - **Cost tracking** (same as #11): `cost_tracker` replaced with
      `costlog`; the Exa request fee is taken from the API response's
      `costDollars.total` (fallback: $7/1k searches + $1/1k text pages),
      stages `exa_query` / `exa_search`.
    - **Retry correctness**: 429/5xx raise inside the `with_retry` lambda so
      they actually trigger backoff (the source called `raise_for_status`
      after the retry wrapper returned); deterministic 4xx fail once,
      preserving the Exa error body in the exception.
    - **Query generation pinned**: gpt-4o-mini (`EXTRACT_MODEL`) at
      temperature 0 (source used the configured default model with default
      temperature); `exa_query` cost stage added; runs that fall back to
      keyword truncation are tagged `query_source="fallback"` in
      `raw_response`. `highlights` content type dropped (text subsumes it;
      halves the contents fee).
