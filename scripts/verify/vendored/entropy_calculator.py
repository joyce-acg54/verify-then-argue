"""
Stage 4b: Verdict-cluster entropy.

Adapted from the earlier pipeline's entropy_calculator.py — see PROVENANCE.md (#8):
the LLM clustering call is removed. The gpt-4o verdict step already emits one
of the four canonical labels (config.VERDICT_LABELS) at temperature 0, so the
clusters ARE the verdict labels. Shannon entropy normalized by log(4);
aleatoric/epistemic decomposition kept unchanged.

api_error runs are excluded from clustering (they carry no verdict signal).
"""

import math
from collections import Counter

from .models import EvidenceRun

N_CLUSTERS = 4  # supports / refutes / insufficient / no_evidence


def compute_semantic_entropy(runs: list[EvidenceRun]) -> tuple[float, float, float]:
    """Returns (entropy, aleatoric, epistemic), all in [0, 1]."""
    labels = _cluster_labels(runs)
    if not labels:
        return 1.0, 0.5, 0.5
    if len(labels) == 1:
        return 0.0, 0.0, 0.0

    entropy = _shannon_entropy(labels)
    aleatoric, epistemic = _decompose(runs, entropy)
    return entropy, aleatoric, epistemic


def cluster_counts(runs: list[EvidenceRun]) -> dict[str, int]:
    return dict(Counter(_cluster_labels(runs)))


def _cluster_labels(runs: list[EvidenceRun]) -> list[str]:
    return [r.verdict for r in runs if r.verdict != "api_error"]


def _shannon_entropy(labels: list[str]) -> float:
    """Shannon entropy normalized to [0, 1] by log(4 clusters)."""
    if not labels:
        return 1.0
    counts  = Counter(labels)
    n       = len(labels)
    max_h   = math.log(N_CLUSTERS)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)
    return min(entropy / max_h, 1.0) if max_h > 0 else 0.0


def _decompose(runs: list[EvidenceRun], total_entropy: float) -> tuple[float, float]:
    """Entropy over high-tier (1-2) sources only ~= aleatoric; the remainder
    is epistemic (noise from low-tier sources)."""
    high_tier = [r.verdict for r in runs
                 if r.verdict != "api_error" and r.source_tier <= 2]
    if len(high_tier) < 2:
        return 0.0, total_entropy

    aleatoric = _shannon_entropy(high_tier)
    epistemic = max(0.0, total_entropy - aleatoric)
    return aleatoric, epistemic
