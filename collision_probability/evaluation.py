"""Evaluate probability forecasts against deterministic reference outcomes."""
import math


def collision_label(minimum_separation_m, combined_radius_m):
    """Reference paths are treated as exact; no covariance is assigned."""
    if not math.isfinite(minimum_separation_m) or minimum_separation_m < 0:
        raise ValueError('Minimum separation must be finite and nonnegative')
    if not math.isfinite(combined_radius_m) or combined_radius_m <= 0:
        raise ValueError('Combined collision distance must be finite and positive')
    return int(minimum_separation_m <= combined_radius_m)


def binary_scores(probability, label, log_probability=None):
    """Proper probability scores for one outcome; lower values are better.

    A finite log_probability retains rare-event log loss when Pc underflows.
    Truly impossible observed outcomes have infinite log loss, represented as
    null plus an explicit flag so output remains standards-compliant JSON.
    """
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Probability must be finite and between zero and one')
    if label not in (0, 1):
        raise ValueError('Reference outcome must be zero or one')
    if log_probability is not None:
        if not math.isfinite(log_probability) or log_probability > 0:
            raise ValueError('Log probability must be finite and nonpositive')
        if not math.isclose(math.exp(log_probability), probability, rel_tol=1e-8, abs_tol=1e-300):
            raise ValueError('Probability and log probability disagree')
    if label == 1:
        loss = (-log_probability if log_probability is not None else
                -math.log(probability) if probability > 0 else math.inf)
    else:
        loss = -math.log1p(-probability) if probability < 1 else math.inf
    return {'brier_score': (probability-label)**2,
            'log_loss': loss if math.isfinite(loss) else None,
            'log_loss_is_infinite': not math.isfinite(loss)}
