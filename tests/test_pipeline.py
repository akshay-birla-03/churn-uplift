"""Tests for the end-to-end pipeline, evaluation and CLI."""

import numpy as np

from upliftkit import __version__
from upliftkit.cli import main
from upliftkit.data import generate
from upliftkit.evaluate import EvaluationResult, evaluate_learner
from upliftkit.learners import TLearner
from upliftkit.pipeline import run_pipeline


def test_pipeline_runs_and_reports_both_learners():
    result = run_pipeline(n=3000, seed=7)
    assert set(result.results) == {"S-learner", "T-learner"}
    assert result.best_learner in result.results


def test_pipeline_quality_bar():
    # Genuine recovery of heterogeneous effects on held-out data.
    result = run_pipeline(n=4000, seed=7)
    for name, r in result.results.items():
        assert r.qini_coefficient > 0.0, f"{name} Qini not positive: {r.qini_coefficient}"
        assert r.true_uplift_correlation >= 0.3, (
            f"{name} corr too low: {r.true_uplift_correlation}"
        )


def test_best_learner_has_max_qini():
    result = run_pipeline(n=3000, seed=7)
    best = result.results[result.best_learner]
    assert best.qini_coefficient == max(r.qini_coefficient for r in result.results.values())


def test_evaluate_learner_returns_dataclass():
    ds = generate(n=2000, seed=7)
    f = ds.frame
    cols = ds.feature_columns
    learner = TLearner().fit(f[cols], f["treatment"], f["outcome"])
    res = evaluate_learner(
        learner, f[cols], f["treatment"], f["outcome"], true_uplift=f["true_uplift"]
    )
    assert isinstance(res, EvaluationResult)
    d = res.as_dict()
    assert set(d) >= {"qini_coefficient", "uplift_at_10pct", "true_uplift_correlation"}
    assert np.isfinite(res.qini_coefficient)


def test_evaluate_without_true_uplift_gives_nan_corr():
    ds = generate(n=1000, seed=7)
    f = ds.frame
    cols = ds.feature_columns
    learner = TLearner().fit(f[cols], f["treatment"], f["outcome"])
    res = evaluate_learner(learner, f[cols], f["treatment"], f["outcome"])
    assert np.isnan(res.true_uplift_correlation)


def test_cli_main_runs(capsys):
    rc = main(["-n", "1500", "--seed", "7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "S-learner" in out and "T-learner" in out
    assert "qini_coefficient" in out
    assert "Best learner" in out


def test_version_exposed():
    assert isinstance(__version__, str) and __version__.count(".") >= 2
