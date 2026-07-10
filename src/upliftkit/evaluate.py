"""Evaluation of a fitted uplift learner on a held-out set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import auuc, qini_coefficient, uplift_at_k


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics for one fitted learner on held-out data.

    Attributes:
        learner: Human-readable learner name.
        qini_coefficient: Normalized Qini coefficient (>0 beats random).
        auuc: Area under the cumulative uplift curve vs. random.
        uplift_at_10pct: Observed uplift in the top 10% targeted segment.
        true_uplift_correlation: Pearson correlation between predicted and true
            per-row uplift (only available with synthetic ground truth).
    """

    learner: str
    qini_coefficient: float
    auuc: float
    uplift_at_10pct: float
    true_uplift_correlation: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "learner": self.learner,
            "qini_coefficient": self.qini_coefficient,
            "auuc": self.auuc,
            "uplift_at_10pct": self.uplift_at_10pct,
            "true_uplift_correlation": self.true_uplift_correlation,
        }


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_learner(
    learner,
    X,
    treatment,
    y,
    true_uplift=None,
    name: str | None = None,
) -> EvaluationResult:
    """Evaluate a fitted learner on held-out data.

    Args:
        learner: A fitted object exposing ``predict_uplift(X)``.
        X: Held-out features.
        treatment: Held-out treatment indicators.
        y: Held-out outcomes.
        true_uplift: Optional known per-row uplift for correlation.
        name: Optional display name; defaults to the learner's class name.

    Returns:
        An :class:`EvaluationResult`.
    """
    pred = np.asarray(learner.predict_uplift(X), dtype=float)
    corr = float("nan")
    if true_uplift is not None:
        corr = _safe_pearson(pred, np.asarray(true_uplift, dtype=float))
    return EvaluationResult(
        learner=name or type(learner).__name__,
        qini_coefficient=qini_coefficient(y, treatment, pred),
        auuc=auuc(y, treatment, pred),
        uplift_at_10pct=uplift_at_k(y, treatment, pred, k=0.1),
        true_uplift_correlation=corr,
    )
