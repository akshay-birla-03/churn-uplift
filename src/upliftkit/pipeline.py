"""End-to-end pipeline: generate -> split -> fit S/T learners -> evaluate -> pick."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from .data import generate
from .evaluate import EvaluationResult, evaluate_learner
from .learners import SLearner, TLearner


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of a full pipeline run.

    Attributes:
        results: Evaluation result per learner name.
        best_learner: Name of the learner with the highest Qini coefficient.
    """

    results: dict[str, EvaluationResult]
    best_learner: str

    @property
    def best(self) -> EvaluationResult:
        return self.results[self.best_learner]


def run_pipeline(
    n: int = 4000,
    seed: int = 7,
    test_size: float = 0.3,
) -> PipelineResult:
    """Run the full uplift-modeling pipeline on synthetic retention data.

    Args:
        n: Number of synthetic customers.
        seed: RNG / split seed.
        test_size: Fraction held out for evaluation.

    Returns:
        A :class:`PipelineResult`.
    """
    dataset = generate(n=n, seed=seed)
    frame = dataset.frame
    features = dataset.feature_columns

    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["treatment"],
    )

    X_train, t_train, y_train = train[features], train["treatment"], train["outcome"]
    X_test, t_test, y_test = test[features], test["treatment"], test["outcome"]
    true_test = test["true_uplift"].to_numpy()

    results: dict[str, EvaluationResult] = {}
    for name, learner in (("S-learner", SLearner()), ("T-learner", TLearner())):
        learner.fit(X_train, t_train, y_train)
        results[name] = evaluate_learner(
            learner, X_test, t_test, y_test, true_uplift=true_test, name=name
        )

    best = max(results.values(), key=lambda r: r.qini_coefficient)
    return PipelineResult(results=results, best_learner=best.learner)
