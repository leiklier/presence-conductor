"""Analytic empty-noise statistics for the raw evidence statistic (rule 3.7).

When a zone has no empirical statistic calibration for an evidence path,
the centered score (3.2) falls back to the distribution of
``S = max(0, max of m iid standard normals)`` — the raw statistic under
ideal Gaussian noise with a perfectly calibrated floor. ``E[S]`` grows with
``m`` (a max over gates is a multiple-comparison statistic); centering
against it is what keeps the owned-gate count from silently inflating a
zone's false-alarm rate (3.7).

Pure math, deterministic (7.3): no randomness, no clock.
"""

from __future__ import annotations

import math
from functools import lru_cache
from statistics import NormalDist

from .events import GATE_COUNT


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x: float) -> float:
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@lru_cache(maxsize=GATE_COUNT + 1)
def onesided_max_stats(m: int) -> tuple[float, float]:
    """``(mean, std)`` of ``max(0, max of m iid N(0,1))`` (rule 3.7).

    By the tail formula, ``E[S] = integral 0..inf of (1 - Phi(x)^m) dx`` and
    ``E[S^2] = integral 0..inf of 2x(1 - Phi(x)^m) dx``. Simpson integration
    to 8 sigma is exact far below ``sigma_min``'s resolution. For m = 1 this
    gives the textbook ``E[max(0, Z)] = 1/sqrt(2*pi) ~ 0.3989``.
    """
    m = max(1, m)
    upper, steps = 8.0, 4000  # steps must be even for Simpson's rule
    h = upper / steps
    e1 = e2 = 0.0
    for i in range(steps + 1):
        x = i * h
        tail = 1.0 - _phi(x) ** m
        weight = 1.0 if i in (0, steps) else (4.0 if i % 2 else 2.0)
        e1 += weight * tail
        e2 += weight * 2.0 * x * tail
    e1 *= h / 3.0
    e2 *= h / 3.0
    return e1, math.sqrt(max(0.0, e2 - e1 * e1))


@lru_cache(maxsize=64)
def clipped_mean(m: int, neg_cap: float, pos_cap: float) -> float:
    """Mean of the *clamped* centered score under the analytic model (3.2).

    Asymmetric clamping of ``(S - m0) / s0`` to ``[-neg_cap, pos_cap]``
    leaves a positive residual mean for multi-gate maxima (the left tail
    is cut, the right one barely is); rule 3.2 subtracts this value so the
    final score is exactly mean-zero. ``S`` has an atom at 0 of mass
    ``Phi(0)^m`` and density ``m Phi(s)^(m-1) phi(s)`` above it.
    """
    m = max(1, m)
    m0, s0 = onesided_max_stats(m)

    def clip(s: float) -> float:
        return min(pos_cap, max(-neg_cap, (s - m0) / s0))

    total = (0.5**m) * clip(0.0)
    upper, steps = 8.0, 4000
    h = upper / steps
    acc = 0.0
    for i in range(steps + 1):
        x = i * h
        density = m * _phi(x) ** (m - 1) * _pdf(x)
        weight = 1.0 if i in (0, steps) else (4.0 if i % 2 else 2.0)
        acc += weight * clip(x) * density
    return total + acc * h / 3.0


@lru_cache(maxsize=64)
def attack_threshold(m: int, tail: float) -> float:
    """Raw-statistic threshold with ``P_H0(S >= threshold) = tail`` (4.2).

    Mean/std standardization does not equalize tails across gate counts —
    a fixed centered threshold fires ~10x more often for one gate than for
    three — so attack candidacy thresholds on the tail probability of the
    max itself: ``P(S >= s) = 1 - Phi(s)^m``.
    """
    m = max(1, m)
    return NormalDist().inv_cdf((1.0 - tail) ** (1.0 / m))
