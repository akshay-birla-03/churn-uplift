"""Tests for the synthetic data generator: determinism and known-effect sanity."""

import numpy as np

from upliftkit.data import FEATURE_COLUMNS, SEGMENTS, generate, true_uplift_for


def test_generate_shapes_and_columns():
    ds = generate(n=500, seed=1)
    assert len(ds.frame) == 500
    for col in FEATURE_COLUMNS + ["treatment", "outcome", "segment", "true_uplift"]:
        assert col in ds.frame.columns
    assert ds.X.shape == (500, len(FEATURE_COLUMNS))


def test_generate_is_deterministic():
    a = generate(n=300, seed=42).frame
    b = generate(n=300, seed=42).frame
    assert a.equals(b)


def test_different_seeds_differ():
    a = generate(n=300, seed=1).frame
    b = generate(n=300, seed=2).frame
    assert not a["outcome"].equals(b["outcome"])


def test_treatment_is_randomized_roughly_balanced():
    ds = generate(n=4000, seed=3)
    frac = ds.treatment.mean()
    assert 0.45 < frac < 0.55


def test_treatment_independent_of_features():
    # RCT: treatment should not correlate meaningfully with any feature.
    ds = generate(n=4000, seed=5)
    for col in FEATURE_COLUMNS:
        r = np.corrcoef(ds.frame[col].to_numpy(dtype=float), ds.treatment)[0, 1]
        assert abs(r) < 0.1


def test_all_segments_present():
    ds = generate(n=5000, seed=7)
    present = set(ds.frame["segment"].unique())
    for seg in SEGMENTS:
        assert seg in present, f"missing segment {seg}"


def test_known_effect_signs_by_segment():
    # The baked-in effect must show up in the true uplift per segment.
    ds = generate(n=5000, seed=7)
    f = ds.frame
    mean_up = f.groupby("segment")["true_uplift"].mean()
    assert mean_up["persuadable"] > 0.2
    assert mean_up["sleeping_dog"] < 0.0
    assert abs(mean_up["sure_thing"]) < 0.1
    assert abs(mean_up["lost_cause"]) < 0.1


def test_empirical_ate_matches_true_ate():
    # Difference-in-means over the RCT should approximate the mean true uplift.
    ds = generate(n=8000, seed=11)
    f = ds.frame
    emp = f[f.treatment == 1]["outcome"].mean() - f[f.treatment == 0]["outcome"].mean()
    true_ate = f["true_uplift"].mean()
    assert abs(emp - true_ate) < 0.05


def test_true_uplift_for_helper():
    ds = generate(n=100, seed=1)
    assert np.allclose(true_uplift_for(ds.frame), ds.true_uplift)
