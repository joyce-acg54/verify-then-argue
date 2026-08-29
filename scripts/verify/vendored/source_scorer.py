"""
Stage 4a: Source Scorer — Beta-distribution reliability scoring.
Vendored from the earlier pipeline's source_scorer.py (unchanged logic; tier
lookup now uses the generalized taxonomy in config.py).
"""

from . import config
from .models import EvidenceRun, RoutedClaim


def compute_source_score(
    routed_claim: RoutedClaim,
    runs: list[EvidenceRun],
) -> tuple[float, float, float]:
    """Aggregated source reliability across runs.
    Returns (posterior_mean, beta_alpha, beta_beta)."""
    if not runs:
        return 0.5, 1.0, 1.0

    informative_runs = [r for r in runs if r.verdict in ("supports", "refutes")]
    if not informative_runs:
        return 0.0, 1.0, 10.0  # no real sources found

    agg_alpha = 0.0
    agg_beta  = 0.0
    total_weight = 0.0

    for run in informative_runs:
        tier_info = config.SOURCE_TIERS.get(run.source_tier,
                                            config.SOURCE_TIERS[config.UNKNOWN_TIER])
        prior_alpha = tier_info["beta_alpha"]
        prior_beta  = tier_info["beta_beta"]

        if run.verdict == "supports":
            posterior_alpha = prior_alpha + 1.0
            posterior_beta  = prior_beta
        else:  # refutes
            posterior_alpha = prior_alpha
            posterior_beta  = prior_beta + 1.0

        tier_weight = 1.0 / run.source_tier
        agg_alpha   += posterior_alpha * tier_weight
        agg_beta    += posterior_beta  * tier_weight
        total_weight += tier_weight

    if total_weight > 0:
        agg_alpha /= total_weight
        agg_beta  /= total_weight

    score = agg_alpha / (agg_alpha + agg_beta)
    return score, agg_alpha, agg_beta


def get_verdict_direction(runs: list[EvidenceRun]) -> str:
    """'for' | 'against' | 'mixed' based on supports/refutes majority."""
    support_count = sum(1 for r in runs if r.verdict == "supports")
    refute_count  = sum(1 for r in runs if r.verdict == "refutes")
    total = support_count + refute_count
    if total == 0:
        return "mixed"
    if support_count / total > 0.6:
        return "for"
    if refute_count / total > 0.6:
        return "against"
    return "mixed"


def count_informative_runs(runs: list[EvidenceRun]) -> int:
    return sum(1 for r in runs if r.verdict in ("supports", "refutes"))
