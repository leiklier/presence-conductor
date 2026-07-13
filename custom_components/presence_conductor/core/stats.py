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

from .events import GATE_COUNT


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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
