"""
Data models for the claim verification pipeline.
Vendored from the earlier pipeline's models.py (unchanged except: conformal
prediction_set kept, no other fields removed).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enumerations ────────────────────────────────────────────────────────────

class ClaimType(str, Enum):
    NUMERIC     = "numeric"
    FACTUAL     = "factual"
    COMPARATIVE = "comparative"
    FORECAST    = "forecast"
    NORMATIVE   = "normative"

class ClaimScope(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    MIXED    = "mixed"

class ClaimCategory(str, Enum):
    PROBLEM    = "problem"
    SOLUTION   = "solution"
    MARKET     = "market"
    FINANCE    = "finance"
    OPERATIONS = "operations"
    TEAM       = "team"
    OTHER      = "other"

class VerifiabilityLabel(str, Enum):
    VERIFIABLE   = "verifiable"
    UNVERIFIABLE = "unverifiable"
    INFERENCE    = "inference"
    NORMATIVE    = "normative"

class EpistemicLabel(str, Enum):
    BELIEF      = "belief"
    DISBELIEF   = "disbelief"
    IGNORANCE   = "ignorance"
    NO_EVIDENCE = "no_evidence"
    API_ERROR   = "api_error"


# ── Stage 0: Raw page ─────────────────────────────────────────────────────────

@dataclass
class Page:
    startup_id:   str
    source_file:  str
    page_number:  int
    page_text:    str
    should_extract: bool = False
    gate_reason:  str = ""


# ── Stage 1: Atomic claim after L1A/B/C ─────────────────────────────────────

@dataclass
class AtomicClaim:
    claim_id:     str
    claim_text:   str
    source_page:  int
    source_file:  str
    startup_id:   str
    category:     ClaimCategory
    speaker:      str
    scope:        ClaimScope
    claim_type:   ClaimType

    support_label:      str   = ""
    support_span:       str   = ""
    support_confidence: float = 0.0
    audit_label:        str   = ""
    audit_reason:       str   = ""

    is_duplicate:   bool          = False
    canonical_id:   Optional[str] = None


# ── Stage 2: Claim after dedup + routing ─────────────────────────────────────

@dataclass
class RoutedClaim:
    claim:                AtomicClaim
    verifiability:        VerifiabilityLabel
    verifiability_lower:  float = 0.0
    verifiability_upper:  float = 1.0


# ── Stage 3: Single evidence run from web search ─────────────────────────────

@dataclass
class EvidenceRun:
    run_index:    int
    evidence_text: str
    source_url:   str
    source_domain: str
    source_tier:  int
    verdict:      str
    reasoning:    str
    raw_response: str = ""


# ── Stage 4: Fully scored claim ───────────────────────────────────────────────

@dataclass
class ScoredClaim:
    routed_claim:   RoutedClaim
    evidence_runs:  list[EvidenceRun] = field(default_factory=list)

    source_score:   float = 0.0
    beta_alpha:     float = 1.0
    beta_beta:      float = 1.0

    semantic_entropy:       float = 0.0
    aleatoric_uncertainty:  float = 0.0
    epistemic_uncertainty:  float = 0.0

    prediction_set: list[EpistemicLabel] = field(default_factory=list)

    final_label:    Optional[EpistemicLabel] = None
    confidence:     float = 0.0
    explanation:    str   = ""
