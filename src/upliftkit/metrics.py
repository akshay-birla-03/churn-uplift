"""Uplift-specific evaluation metrics.

These metrics answer the question a churn team actually cares about: *if we sort
customers by predicted uplift and treat them from the top down, how much extra
retention do we buy?* Ordinary classification metrics (AUC, accuracy) cannot
answer this because the ground-truth per-customer uplift is never observed --
each customer is either treated or not, never both.

Conventions used throughout:

* ``y``         -- binary outcomes (1 = retained).
* ``treatment`` -- binary treatment indicator (1 = offered retention treatment).
* ``uplift``    -- predicted uplift score; larger = target sooner.

All curve functions return ``(x, y)`` where ``x`` is the fraction of the
population targeted (0..1, starting at 0) and ``y`` is the cumulative benefit.
The **Qini** family uses the classic uplift correction that reweights the
control group so treated and control counts are comparable at every prefix.
"""

from __future__ import annotations

import numpy as np


def _validate(y, treatment, uplift):
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    uplift = np.asarray(uplift, dtype=float)
    if not (len(y) == len(treatment) == len(uplift)):
        raise ValueError("y, treatment and uplift must have the same length.")
    if len(y) == 0:
        raise ValueError("inputs must be non-empty.")
    return y, treatment, uplift


def _order_desc(uplift: np.ndarray) -> np.ndarray:
    """Indices that sort by uplift descending, ties broken deterministically."""
    return np.argsort(-uplift, kind="stable")


def qini_curve(y, treatment, uplift) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve with control-group reweighting.

    At each prefix of the population (sorted by descending predicted uplift) the
    cumulative Qini value is::

        g(n) = Y_t(n) - Y_c(n) * N_t(n) / N_c(n)

    where ``Y_t``/``Y_c`` are cumulative positive outcomes among treated/control
    and ``N_t``/``N_c`` the cumulative counts. When no control customers have
    been seen yet the correction term is 0.

    Returns:
        ``(x, g)`` with ``x`` the targeted fraction (0..1, leading 0) and ``g``
        the cumulative Qini values (leading 0).
    """
    y, treatment, uplift = _validate(y, treatment, uplift)
    order = _order_desc(uplift)
    y_s = y[order]
    t_s = treatment[order]

    cum_yt = np.cumsum(y_s * t_s)
    cum_yc = np.cumsum(y_s * (1.0 - t_s))
    cum_nt = np.cumsum(t_s)
    cum_nc = np.cumsum(1.0 - t_s)

    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.where(cum_nc > 0, cum_yc * cum_nt / cum_nc, 0.0)
    g = cum_yt - correction

    n = len(y)
    x = np.arange(1, n + 1, dtype=float) / n
    x = np.concatenate([[0.0], x])
    g = np.concatenate([[0.0], g])
    return x, g


def cumulative_uplift_curve(y, treatment, uplift) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative gain curve (incremental retained customers).

    At each prefix the gain is the difference in *response rates* between treated
    and control scaled back up to the prefix size::

        gain(n) = (Y_t(n)/N_t(n) - Y_c(n)/N_c(n)) * n

    This estimates how many *extra* customers are retained by treating the top
    ``n`` customers. Prefixes lacking either arm contribute 0 gain.

    Returns:
        ``(x, gain)`` with ``x`` the targeted fraction (0..1, leading 0).
    """
    y, treatment, uplift = _validate(y, treatment, uplift)
    order = _order_desc(uplift)
    y_s = y[order]
    t_s = treatment[order]

    cum_yt = np.cumsum(y_s * t_s)
    cum_yc = np.cumsum(y_s * (1.0 - t_s))
    cum_nt = np.cumsum(t_s)
    cum_nc = np.cumsum(1.0 - t_s)
    sizes = np.arange(1, len(y) + 1, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_t = np.where(cum_nt > 0, cum_yt / cum_nt, 0.0)
        rate_c = np.where(cum_nc > 0, cum_yc / cum_nc, 0.0)
    valid = (cum_nt > 0) & (cum_nc > 0)
    gain = np.where(valid, (rate_t - rate_c) * sizes, 0.0)

    x = np.concatenate([[0.0], sizes / len(y)])
    gain = np.concatenate([[0.0], gain])
    return x, gain


def _area_vs_random(x: np.ndarray, curve: np.ndarray) -> float:
    """Trapezoidal area between a curve and the straight random baseline.

    The random baseline is the chord from ``(0, 0)`` to ``(x[-1], curve[-1])``.
    A positive value means the model orders customers better than random.
    """
    area_curve = float(np.trapezoid(curve, x))
    area_random = 0.5 * x[-1] * curve[-1]
    return area_curve - area_random


def qini_coefficient(y, treatment, uplift) -> float:
    """Normalized Qini coefficient.

    Defined as the area between the Qini curve and the random-targeting line,
    normalized by the population size. This per-customer normalization is stable
    even when the overall treatment effect is near zero (unlike dividing by the
    curve's endpoint). It is positive when the model beats random targeting and
    ~0 for a useless (random-scoring) model.
    """
    x, g = qini_curve(y, treatment, uplift)
    n = len(g) - 1  # exclude the leading origin point
    if n <= 0:
        return 0.0
    return _area_vs_random(x, g) / n


def auuc(y, treatment, uplift) -> float:
    """Area Under the Uplift Curve (cumulative gain), vs. random.

    Positive when top-ranked customers yield more incremental retention than a
    random ordering would.
    """
    x, gain = cumulative_uplift_curve(y, treatment, uplift)
    return _area_vs_random(x, gain)


def uplift_at_k(y, treatment, uplift, k: float = 0.1) -> float:
    """Observed uplift within the top-``k`` fraction ranked by predicted uplift.

    This is the empirical treatment effect (treated response rate minus control
    response rate) among the ``k`` fraction of customers the model would target
    first -- the quantity a campaign actually realizes.

    Args:
        k: Fraction of the population to target, in ``(0, 1]``.

    Returns:
        Estimated uplift in the targeted segment. Returns ``nan`` if the segment
        lacks either a treated or a control customer.
    """
    if not 0.0 < k <= 1.0:
        raise ValueError("k must be in (0, 1].")
    y, treatment, uplift = _validate(y, treatment, uplift)
    n = len(y)
    top = max(1, int(round(k * n)))
    order = _order_desc(uplift)[:top]
    y_top = y[order]
    t_top = treatment[order]
    n_t = t_top.sum()
    n_c = (1.0 - t_top).sum()
    if n_t == 0 or n_c == 0:
        return float("nan")
    return float(y_top[t_top == 1].mean() - y_top[t_top == 0].mean())
