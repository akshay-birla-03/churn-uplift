"""churn-uplift: a causal/uplift modeling toolkit for retention marketing.

Public API re-exports the pieces most users need: the synthetic data generator,
the S/T meta-learners, the uplift metrics, and the end-to-end pipeline.
"""

from __future__ import annotations

from .data import SEGMENTS, SyntheticDataset, generate, true_uplift_for
from .evaluate import EvaluationResult, evaluate_learner
from .learners import SLearner, TLearner
from .metrics import (
    auuc,
    cumulative_uplift_curve,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)
from .pipeline import PipelineResult, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SEGMENTS",
    "SyntheticDataset",
    "generate",
    "true_uplift_for",
    "SLearner",
    "TLearner",
    "qini_curve",
    "qini_coefficient",
    "cumulative_uplift_curve",
    "auuc",
    "uplift_at_k",
    "EvaluationResult",
    "evaluate_learner",
    "PipelineResult",
    "run_pipeline",
]
