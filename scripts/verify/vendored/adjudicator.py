"""
Stage 5: Adjudicator (fully deterministic — no LLM).
Combines source_score and verdict-cluster entropy into a final 4-way
epistemic label: belief / disbelief / ignorance / no_evidence.

Vendored from the earlier pipeline's adjudicator.py — see PROVENANCE.md (#9):
conformal calibration file I/O removed (threshold is the explicit default in
config.CONFORMAL_DEFAULT_THRESHOLD); nonconformity scores, prediction-set
construction, priority pick and confidence logic kept verbatim.
"""

from . import config
from .models import EpistemicLabel, EvidenceRun, RoutedClaim, ScoredClaim
from .source_scorer import compute_source_score, count_informative_runs, get_verdict_direction
from .entropy_calculator import compute_semantic_entropy


def adjudicate(
    routed_claim: RoutedClaim,
    runs: list[EvidenceRun],
    n_runs_expected: int = config.N_VERIFICATION_RUNS,
    conformal_threshold: float = config.CONFORMAL_DEFAULT_THRESHOLD,
) -> ScoredClaim:
    """Produce a fully scored and labeled ScoredClaim."""
    # Early exit: all runs failed
    if runs and all(r.verdict == "api_error" for r in runs):
        return ScoredClaim(
            routed_claim=routed_claim,
            evidence_runs=runs,
            source_score=0.0,
            beta_alpha=1.0,
            beta_beta=1.0,
            semantic_entropy=0.0,
            aleatoric_uncertainty=0.0,
            epistemic_uncertainty=0.0,
            prediction_set=[EpistemicLabel.API_ERROR],
            final_label=EpistemicLabel.API_ERROR,
            confidence=0.0,
            explanation="All verification runs failed due to API errors — claim not verified",
        )

    source_score, beta_alpha, beta_beta = compute_source_score(routed_claim, runs)
    entropy, aleatoric, epistemic = compute_semantic_entropy(runs)
    direction = get_verdict_direction(runs)
    n_informative = count_informative_runs(runs)

    prediction_set = _conformal_prediction_set(
        source_score, entropy, direction, n_informative,
        n_runs_expected, conformal_threshold,
    )
    final_label, confidence = _pick_label(prediction_set, source_score, entropy)
    explanation = _build_explanation(
        final_label, source_score, entropy, aleatoric, epistemic,
        direction, n_informative, n_runs_expected, prediction_set,
    )

    return ScoredClaim(
        routed_claim=routed_claim,
        evidence_runs=runs,
        source_score=source_score,
        beta_alpha=beta_alpha,
        beta_beta=beta_beta,
        semantic_entropy=entropy,
        aleatoric_uncertainty=aleatoric,
        epistemic_uncertainty=epistemic,
        prediction_set=prediction_set,
        final_label=final_label,
        confidence=confidence,
        explanation=explanation,
    )


# ── Conformal prediction ──────────────────────────────────────────────────────

def _conformal_prediction_set(
    source_score: float,
    entropy: float,
    direction: str,
    n_informative: int,
    n_runs_expected: int,
    threshold: float,
) -> list[EpistemicLabel]:
    prediction_set = []
    for label in (EpistemicLabel.BELIEF, EpistemicLabel.DISBELIEF,
                  EpistemicLabel.IGNORANCE, EpistemicLabel.NO_EVIDENCE):
        nc = _nonconformity_score(label, source_score, entropy,
                                  direction, n_informative, n_runs_expected)
        if nc <= threshold:
            prediction_set.append(label)
    if not prediction_set:
        prediction_set = [EpistemicLabel.IGNORANCE]
    return prediction_set


def _nonconformity_score(
    label: EpistemicLabel,
    source_score: float,
    entropy: float,
    direction: str,
    n_informative: int,
    n_runs_expected: int,
) -> float:
    if label == EpistemicLabel.BELIEF:
        if direction != "for":
            return 1.0
        return ((1.0 - source_score) + entropy) / 2

    elif label == EpistemicLabel.DISBELIEF:
        if direction != "against":
            return 1.0
        return ((1.0 - source_score) + entropy) / 2

    elif label == EpistemicLabel.IGNORANCE:
        direction_mixed = 1.0 if direction == "mixed" else 0.3
        return max(0.0, (1.0 - entropy) - direction_mixed * 0.3)

    elif label == EpistemicLabel.NO_EVIDENCE:
        return min(n_informative / max(n_runs_expected, 1), 1.0)

    return 1.0


def _pick_label(
    prediction_set: list[EpistemicLabel],
    source_score: float,
    entropy: float,
) -> tuple[EpistemicLabel, float]:
    priority = [
        EpistemicLabel.BELIEF,
        EpistemicLabel.DISBELIEF,
        EpistemicLabel.NO_EVIDENCE,
        EpistemicLabel.IGNORANCE,
    ]
    for label in priority:
        if label in prediction_set:
            if label == EpistemicLabel.BELIEF:
                confidence = (source_score - config.BELIEF_SOURCE_MIN) / (1 - config.BELIEF_SOURCE_MIN)
                confidence = confidence * (1 - entropy)
            elif label == EpistemicLabel.DISBELIEF:
                confidence = (source_score - config.DISBELIEF_SOURCE_MIN) / (1 - config.DISBELIEF_SOURCE_MIN)
                confidence = confidence * (1 - entropy)
            elif label == EpistemicLabel.NO_EVIDENCE:
                confidence = 0.5
            else:
                confidence = entropy
            return label, max(0.0, min(1.0, confidence))

    return EpistemicLabel.IGNORANCE, 0.5


def _build_explanation(
    label: EpistemicLabel,
    source_score: float,
    entropy: float,
    aleatoric: float,
    epistemic: float,
    direction: str,
    n_informative: int,
    n_runs_expected: int,
    prediction_set: list[EpistemicLabel],
) -> str:
    lines = [
        f"Label: {label.value}",
        f"Source score: {source_score:.2f} (Beta posterior) | Direction: {direction}",
        f"Verdict-cluster entropy: {entropy:.2f} (aleatoric={aleatoric:.2f}, epistemic={epistemic:.2f})",
        f"Informative runs: {n_informative}/{n_runs_expected}",
        f"Prediction set: {[l.value for l in prediction_set]}",
    ]
    if len(prediction_set) > 1:
        lines.append("Note: prediction set contains multiple labels — system is uncertain.")
    if epistemic > aleatoric and entropy > 0.4:
        lines.append("Uncertainty is mostly epistemic — more/better sources may resolve this.")
    elif aleatoric > epistemic and entropy > 0.4:
        lines.append("Uncertainty is mostly aleatoric — the claim may be genuinely contested.")
    return " | ".join(lines)
