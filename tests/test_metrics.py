"""Correctness tests for uplift metrics on hand-computed toy examples."""

import numpy as np
import pytest

from upliftkit.metrics import (
    auuc,
    cumulative_uplift_curve,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)

# Shared toy example (already sorted by descending uplift).
#   uplift = [4, 3, 2, 1]
#   y      = [1, 1, 1, 0]
#   t      = [1, 0, 1, 0]
# Hand-derived Qini values (with control reweighting): [0, 1, 0, 0, 1].
Y = np.array([1, 1, 1, 0])
T = np.array([1, 0, 1, 0])
U = np.array([4.0, 3.0, 2.0, 1.0])


def test_qini_curve_hand_computed():
    x, g = qini_curve(Y, T, U)
    assert np.allclose(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(g, [0.0, 1.0, 0.0, 0.0, 1.0])


def test_qini_curve_starts_at_origin():
    x, g = qini_curve(Y, T, U)
    assert x[0] == 0.0 and g[0] == 0.0
    assert len(x) == len(Y) + 1


def test_cumulative_uplift_curve_hand_computed():
    x, gain = cumulative_uplift_curve(Y, T, U)
    assert np.allclose(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    # Only at the full population do both arms have members with a rate gap.
    assert np.allclose(gain, [0.0, 0.0, 0.0, 0.0, 2.0])


def test_qini_coefficient_hand_computed():
    # area_curve=0.375, area_random=0.5 -> raw=-0.125; /N(=4) = -0.03125
    assert qini_coefficient(Y, T, U) == pytest.approx(-0.03125)


def test_auuc_hand_computed():
    # area_curve=0.25, area_random=0.5*1*2=1 -> -0.75
    assert auuc(Y, T, U) == pytest.approx(-0.75)


def test_uplift_at_k_hand_computed():
    # top 50% -> first two rows: treated {y=1}, control {y=1} -> 0
    assert uplift_at_k(Y, T, U, k=0.5) == pytest.approx(0.0)


def test_uplift_at_k_full_population_is_ate():
    # Whole population: treated mean = (1+1)/2=1, control mean=(1+0)/2=0.5 -> 0.5
    assert uplift_at_k(Y, T, U, k=1.0) == pytest.approx(0.5)


def test_perfect_ranking_beats_random_qini():
    # Construct data where the top-ranked treated customers all respond and the
    # bottom-ranked do not: a good ranking should give a positive Qini.
    rng = np.random.default_rng(0)
    n = 400
    u = np.linspace(1, 0, n)  # descending scores
    t = rng.binomial(1, 0.5, n)
    # treated-top respond; everyone else responds at base rate
    base = rng.binomial(1, 0.2, n)
    y = np.where((u > 0.5) & (t == 1), 1, base)
    assert qini_coefficient(y, t, u) > 0.0


def test_random_scores_qini_near_zero():
    rng = np.random.default_rng(1)
    n = 2000
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3, n)
    scores = rng.normal(size=n)  # unrelated to outcome
    assert abs(qini_coefficient(y, t, scores)) < 0.02


def test_uplift_at_k_invalid_k():
    with pytest.raises(ValueError):
        uplift_at_k(Y, T, U, k=0.0)
    with pytest.raises(ValueError):
        uplift_at_k(Y, T, U, k=1.5)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        qini_curve([1, 0], [1], [0.5, 0.2])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        qini_curve([], [], [])


def test_uplift_at_k_nan_when_arm_missing():
    # top segment is all treated -> undefined uplift
    y = np.array([1, 0, 1, 0])
    t = np.array([1, 1, 0, 0])
    u = np.array([4.0, 3.0, 2.0, 1.0])
    assert np.isnan(uplift_at_k(y, t, u, k=0.5))
