"""Tests for the S- and T-learner meta-learners."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from upliftkit.data import generate
from upliftkit.learners import SLearner, TLearner


def _split(seed=7):
    ds = generate(n=3000, seed=seed)
    f = ds.frame
    n = len(f)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    cut = int(0.7 * n)
    tr, te = f.iloc[idx[:cut]], f.iloc[idx[cut:]]
    cols = ds.feature_columns
    return (tr[cols], tr["treatment"], tr["outcome"], te, cols)


@pytest.mark.parametrize("cls", [SLearner, TLearner])
def test_fit_predict_shape(cls):
    Xtr, ttr, ytr, te, cols = _split()
    learner = cls().fit(Xtr, ttr, ytr)
    pred = learner.predict_uplift(te[cols])
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (len(te),)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("cls", [SLearner, TLearner])
def test_predict_before_fit_raises(cls):
    with pytest.raises(RuntimeError):
        cls().predict_uplift(np.zeros((3, 5)))


def test_tlearner_requires_both_arms():
    Xtr, ttr, ytr, _, _ = _split()
    with pytest.raises(ValueError):
        TLearner().fit(Xtr, np.ones(len(ttr)), ytr)  # all treated


@pytest.mark.parametrize("cls", [SLearner, TLearner])
def test_recovers_true_uplift_correlation(cls):
    # The core quality bar: predicted uplift correlates with the TRUE uplift.
    Xtr, ttr, ytr, te, cols = _split()
    learner = cls().fit(Xtr, ttr, ytr)
    pred = learner.predict_uplift(te[cols])
    r = np.corrcoef(pred, te["true_uplift"].to_numpy())[0, 1]
    assert r >= 0.3, f"{cls.__name__} correlation with true uplift too low: {r:.3f}"


def test_accepts_custom_base_estimator():
    Xtr, ttr, ytr, te, cols = _split()
    learner = SLearner(base_estimator=LogisticRegression(max_iter=1000)).fit(Xtr, ttr, ytr)
    pred = learner.predict_uplift(te[cols])
    assert pred.shape == (len(te),)


def test_slearner_ranks_persuadables_above_sleeping_dogs():
    Xtr, ttr, ytr, te, cols = _split()
    learner = SLearner().fit(Xtr, ttr, ytr)
    te = te.copy()
    te["pred"] = learner.predict_uplift(te[cols])
    pers = te[te.segment == "persuadable"]["pred"].mean()
    dogs = te[te.segment == "sleeping_dog"]["pred"].mean()
    assert pers > dogs
