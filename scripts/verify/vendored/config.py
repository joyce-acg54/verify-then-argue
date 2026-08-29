"""
Configuration: models, thresholds, source tier taxonomy, Beta priors.
Adapted from the earlier pipeline's config.py — see PROVENANCE.md (#1, #4, #6, #9).
"""

import os

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EXTRACT_MODEL  = "gpt-4o-mini"   # L1A gate, L1B atomizer, L1C audit, router
VERDICT_MODEL  = "gpt-4o"        # verdict reasoning over Sonar evidence (temp 0)
EMBED_MODEL    = "text-embedding-3-small"

# ── Perplexity Sonar ──────────────────────────────────────────────────────────
PERPLEXITY_API_KEY = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
SONAR_MODEL = "sonar"

# ── Exa (second retrieval provider — recall-probe robustness run) ────────────
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_NUM_RESULTS = 5
EXA_TEXT_MAX_CHARS = 2000

# ── Token limits (FIX #1: split the shared MAX_TOKENS=1000 that truncated
#    dense-page extraction/audit responses) ───────────────────────────────────
MAX_TOKENS_GATE    = 400
MAX_TOKENS_EXTRACT = 4000
MAX_TOKENS_AUDIT   = 4000
MAX_TOKENS_SONAR   = 700
MAX_TOKENS_VERDICT = 500

N_VERIFICATION_RUNS = 5

# Shared verdict labels (single source of truth — see PROVENANCE #7)
VERDICT_LABELS = ("supports", "refutes", "insufficient", "no_evidence")

# ── Source tier taxonomy (generalized — see PROVENANCE #4) ───────────────────
# Tier 1: government, academic, intergovernmental
# Tier 2: established press, Wikipedia, major research/consultancy
# Tier 3: industry publications, company sites, data aggregators
# Tier 4: blogs, forums, social media, unknown
#
# "tlds" are matched against the domain's label sequence from the right
# (e.g. "ac.uk" matches "ox.ac.uk"); "domains" match the registrable domain
# or any of its subdomains exactly.
SOURCE_TIERS: dict[int, dict] = {
    1: {
        "label": "Government / Academic / Intergovernmental",
        "tlds": ["gov", "edu", "int", "mil", "ac.uk", "gov.uk", "edu.au",
                 "gov.au", "gc.ca", "go.jp", "gouv.fr", "europa.eu"],
        "domains": [
            "who.int", "worldbank.org", "un.org", "oecd.org", "imf.org",
            "iea.org", "irena.org", "wto.org", "ecb.europa.eu",
            "pubmed.ncbi.nlm.nih.gov", "arxiv.org", "ssrn.com", "jstor.org",
            "nature.com", "science.org", "springer.com", "sciencedirect.com",
            "ieee.org", "acm.org", "researchgate.net",
        ],
        "beta_alpha": 95.0,
        "beta_beta":  5.0,
    },
    2: {
        "label": "Established press / Wikipedia / Major consultancy",
        "tlds": [],
        "domains": [
            "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "afp.com",
            "theguardian.com", "ft.com", "wsj.com", "bloomberg.com",
            "economist.com", "nytimes.com", "washingtonpost.com",
            "lemonde.fr", "spiegel.de", "zeit.de", "handelsblatt.com",
            "faz.net", "nikkei.com", "scmp.com",
            "wikipedia.org",
            "mckinsey.com", "bcg.com", "bain.com", "deloitte.com", "pwc.com",
            "kpmg.com", "ey.com", "gartner.com", "forrester.com", "idc.com",
        ],
        "beta_alpha": 75.0,
        "beta_beta":  25.0,
    },
    3: {
        "label": "Industry / Corporate / Data aggregators",
        "tlds": [],
        "domains": [
            "crunchbase.com", "pitchbook.com", "dealroom.co", "cbinsights.com",
            "techcrunch.com", "venturebeat.com", "theinformation.com",
            "eu-startups.com", "sifted.eu", "tech.eu",
            "statista.com", "grandviewresearch.com", "marketsandmarkets.com",
            "mordorintelligence.com", "imarcgroup.com",
            "marketresearchfuture.com", "fortunebusinessinsights.com",
            "linkedin.com", "wellfound.com", "prnewswire.com",
            "businesswire.com", "globenewswire.com",
            "morningstar.com", "spglobal.com",
        ],
        "beta_alpha": 55.0,
        "beta_beta":  45.0,
    },
    4: {
        "label": "Blogs / Forums / Social / Unknown",
        "tlds": [],
        "domains": [
            "medium.com", "substack.com", "reddit.com", "twitter.com",
            "x.com", "quora.com", "facebook.com", "youtube.com",
            "news.ycombinator.com",
        ],
        "beta_alpha": 25.0,
        "beta_beta":  75.0,
    },
}
UNKNOWN_TIER = 4


def tier_for_domain(domain: str) -> int:
    """Suffix/label-based tier lookup (FIX #4: no more substring matching)."""
    if not domain:
        return UNKNOWN_TIER
    d = domain.lower().strip().lstrip(".")
    if d.startswith("www."):
        d = d[4:]
    labels = d.split(".")
    for tier_num, info in SOURCE_TIERS.items():
        for tld in info["tlds"]:
            t = tld.split(".")
            if labels[-len(t):] == t:
                return tier_num
        for dom in info["domains"]:
            if d == dom or d.endswith("." + dom):
                return tier_num
    return UNKNOWN_TIER


# ── Adjudicator thresholds (unchanged from source) ───────────────────────────
BELIEF_SOURCE_MIN     = 0.65
DISBELIEF_SOURCE_MIN  = 0.40
ENTROPY_IGNORANCE_MIN = 0.55
NO_EVIDENCE_MAX_RUNS  = 1
CONFORMAL_DEFAULT_THRESHOLD = 0.5   # was the implicit no-calibration default

# ── Deduplication ─────────────────────────────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.85

# ── Credal set routing ────────────────────────────────────────────────────────
CREDAL_AMBIGUITY_THRESHOLD = 0.4
