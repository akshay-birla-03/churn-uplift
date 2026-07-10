"""Synthetic RCT-style dataset for subscription/telecom retention uplift modeling.

The data generator bakes in a *known* heterogeneous treatment effect so tests can
assert that a learner recovers it. Every customer falls into one of four classic
uplift segments, defined by how a retention treatment (e.g. a discount offer)
changes their probability of being retained:

* ``persuadable``  -- retained only *if* treated (large positive uplift).
* ``sure_thing``   -- retained regardless of treatment (~zero uplift).
* ``lost_cause``   -- churns regardless of treatment (~zero uplift).
* ``sleeping_dog`` -- treatment *annoys* them and lowers retention (negative uplift).

The treatment ``T`` is randomized independently of the features (a true RCT), so
the difference in outcomes between treated and control groups is an unbiased
estimate of the causal effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "usage_score",
    "contract_score",
]

SEGMENTS = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for a generated retention experiment.

    Attributes:
        frame: DataFrame with feature columns, ``treatment``, ``outcome`` and
            (for validation only) ``segment`` and ``true_uplift``.
        feature_columns: Ordered list of model input columns.
    """

    frame: pd.DataFrame
    feature_columns: list[str]

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - conventional ML name
        return self.frame[self.feature_columns]

    @property
    def treatment(self) -> np.ndarray:
        return self.frame["treatment"].to_numpy()

    @property
    def outcome(self) -> np.ndarray:
        return self.frame["outcome"].to_numpy()

    @property
    def true_uplift(self) -> np.ndarray:
        return self.frame["true_uplift"].to_numpy()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _segment_from_features(
    tenure: np.ndarray,
    tickets: np.ndarray,
    contract: np.ndarray,
) -> np.ndarray:
    """Assign a latent uplift segment deterministically from features.

    The mapping is intentionally driven by interpretable rules so the effect is
    genuinely heterogeneous *and* recoverable from ``X``:

    * short-tenure, high-complaint customers on flexible contracts are
      ``persuadable`` (a discount changes their mind);
    * long-tenure, low-complaint customers are ``sure_thing`` (loyal anyway);
    * short-tenure, very high complaints are ``lost_cause`` (already gone);
    * a niche of long-tenure customers who dislike being contacted are
      ``sleeping_dog`` (the offer backfires).
    """
    seg = np.empty(tenure.shape[0], dtype=object)
    for i in range(tenure.shape[0]):
        t, c, k = tenure[i], tickets[i], contract[i]
        if t < 0.4 and k < 0.5 and c >= 2:
            seg[i] = "lost_cause" if c >= 4 else "persuadable"
        elif t >= 0.7 and c <= 1:
            # loyal; a small slice are sleeping dogs who resent outreach
            seg[i] = "sleeping_dog" if k >= 0.85 else "sure_thing"
        elif t < 0.55 and c >= 1:
            seg[i] = "persuadable"
        else:
            seg[i] = "sure_thing"
    return seg


# Baseline (untreated) retention probability and treatment effect per segment.
_SEGMENT_BASELINE = {
    "persuadable": 0.25,
    "sure_thing": 0.85,
    "lost_cause": 0.05,
    "sleeping_dog": 0.80,
}
_SEGMENT_EFFECT = {
    "persuadable": 0.45,
    "sure_thing": 0.02,
    "lost_cause": 0.01,
    "sleeping_dog": -0.25,
}


def generate(n: int = 4000, seed: int = 7) -> SyntheticDataset:
    """Generate a synthetic randomized retention experiment.

    Args:
        n: Number of customers.
        seed: RNG seed for full reproducibility.

    Returns:
        A :class:`SyntheticDataset`.
    """
    rng = np.random.default_rng(seed)

    # Raw, interpretable features.
    tenure = rng.beta(2.0, 3.0, size=n)  # 0..1, skewed toward newer customers
    monthly_charges = rng.normal(70, 25, size=n).clip(15, 150)
    support_tickets = rng.poisson(1.2, size=n).clip(0, 8)
    usage_score = rng.normal(0.5, 0.2, size=n).clip(0, 1)
    contract_score = rng.uniform(0, 1, size=n)  # 0=month-to-month .. 1=locked-in

    segment = _segment_from_features(tenure, support_tickets, contract_score)

    baseline = np.array([_SEGMENT_BASELINE[s] for s in segment])
    effect = np.array([_SEGMENT_EFFECT[s] for s in segment])

    # Add mild feature-driven noise to baseline so the problem isn't degenerate.
    baseline = np.clip(
        baseline + 0.05 * (usage_score - 0.5) + 0.03 * (monthly_charges - 70) / 50,
        0.01,
        0.99,
    )

    # Randomized treatment assignment (true RCT: independent of features).
    treatment = rng.binomial(1, 0.5, size=n)

    p_control = baseline
    p_treated = np.clip(baseline + effect, 0.01, 0.99)
    p = np.where(treatment == 1, p_treated, p_control)
    outcome = rng.binomial(1, p)

    # The *true* per-row uplift is the causal contrast for that customer.
    true_uplift = p_treated - p_control

    frame = pd.DataFrame(
        {
            "tenure_months": (tenure * 72).round(1),
            "monthly_charges": monthly_charges.round(2),
            "support_tickets": support_tickets.astype(int),
            "usage_score": usage_score.round(4),
            "contract_score": contract_score.round(4),
            "treatment": treatment.astype(int),
            "outcome": outcome.astype(int),
            "segment": segment,
            "true_uplift": true_uplift.round(6),
        }
    )
    return SyntheticDataset(frame=frame, feature_columns=list(FEATURE_COLUMNS))


def true_uplift_for(frame: pd.DataFrame) -> np.ndarray:
    """Return the known per-row uplift for a frame produced by :func:`generate`."""
    if "true_uplift" not in frame.columns:
        raise KeyError("frame has no 'true_uplift' column; was it produced by generate()?")
    return frame["true_uplift"].to_numpy()
