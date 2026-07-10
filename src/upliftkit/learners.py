"""Meta-learners for uplift (heterogeneous treatment effect) estimation.

Both learners wrap an arbitrary scikit-learn *classifier* that exposes
``predict_proba``. They estimate, per customer, the incremental effect of the
treatment on the probability of the positive outcome (retention):

    uplift(x) = P(Y=1 | X=x, T=1) - P(Y=1 | X=x, T=0)

* :class:`SLearner` ("single model") trains one model on the features *plus*
  the treatment indicator, then predicts with the flag forced to 1 and to 0.
* :class:`TLearner` ("two models") trains one model on the treated subgroup and
  a separate model on the control subgroup.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier


class _SklearnClassifier(Protocol):
    def fit(self, X, y): ...  # noqa: D401,E704
    def predict_proba(self, X): ...  # noqa: E704


def _default_base() -> _SklearnClassifier:
    return GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        random_state=0,
    )


def _as_2d_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def _positive_proba(model: _SklearnClassifier, X: np.ndarray) -> np.ndarray:
    """Return P(Y=1), robust to a subgroup that saw only one outcome class."""
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1]))
    pos_idx = np.where(classes == 1)[0]
    if pos_idx.size == 0:
        # Model only ever saw class 0 -> P(Y=1) is 0 everywhere.
        return np.zeros(X.shape[0])
    return proba[:, int(pos_idx[0])]


class SLearner:
    """Single-model meta-learner.

    Args:
        base_estimator: Any sklearn classifier with ``predict_proba``. If
            ``None``, a :class:`GradientBoostingClassifier` is used.
    """

    def __init__(self, base_estimator: _SklearnClassifier | None = None) -> None:
        self.base_estimator = base_estimator if base_estimator is not None else _default_base()
        self.model_: _SklearnClassifier | None = None

    def fit(self, X, treatment, y) -> SLearner:
        Xa = _as_2d_array(X)
        t = np.asarray(treatment, dtype=float).reshape(-1, 1)
        y = np.asarray(y)
        design = np.hstack([Xa, t])
        self.model_ = clone(self.base_estimator)
        self.model_.fit(design, y)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("SLearner is not fitted; call fit() first.")
        Xa = _as_2d_array(X)
        n = Xa.shape[0]
        treated = np.hstack([Xa, np.ones((n, 1))])
        control = np.hstack([Xa, np.zeros((n, 1))])
        return _positive_proba(self.model_, treated) - _positive_proba(self.model_, control)


class TLearner:
    """Two-model meta-learner (separate treated / control models).

    Args:
        base_estimator: Any sklearn classifier with ``predict_proba``. If
            ``None``, a :class:`GradientBoostingClassifier` is used.
    """

    def __init__(self, base_estimator: _SklearnClassifier | None = None) -> None:
        self.base_estimator = base_estimator if base_estimator is not None else _default_base()
        self.model_treated_: _SklearnClassifier | None = None
        self.model_control_: _SklearnClassifier | None = None

    def fit(self, X, treatment, y) -> TLearner:
        Xa = _as_2d_array(X)
        t = np.asarray(treatment).astype(int)
        y = np.asarray(y)
        if t.sum() == 0 or (t == 0).sum() == 0:
            raise ValueError("TLearner needs both treated and control samples.")
        self.model_treated_ = clone(self.base_estimator)
        self.model_control_ = clone(self.base_estimator)
        self.model_treated_.fit(Xa[t == 1], y[t == 1])
        self.model_control_.fit(Xa[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X) -> np.ndarray:
        if self.model_treated_ is None or self.model_control_ is None:
            raise RuntimeError("TLearner is not fitted; call fit() first.")
        Xa = _as_2d_array(X)
        return _positive_proba(self.model_treated_, Xa) - _positive_proba(self.model_control_, Xa)
