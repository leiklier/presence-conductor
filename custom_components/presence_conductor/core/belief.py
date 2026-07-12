"""Log-odds numerics for the occupancy posterior (spec §0).

All evidence arithmetic happens in the log-odds domain: ``lambda`` is the
log-odds of "zone is occupied" and ``probability = sigmoid(lambda)``. These
helpers are the only place the logit/sigmoid transforms live.
"""

from __future__ import annotations

import math


def logit(p: float) -> float:
    """Log-odds of probability ``p`` (§0)."""
    return math.log(p / (1.0 - p))


def sigmoid(lam: float) -> float:
    """Probability for log-odds ``lam`` (§0)."""
    # Numerically stable for large |lam|.
    if lam >= 0:
        return 1.0 / (1.0 + math.exp(-lam))
    e = math.exp(lam)
    return e / (1.0 + e)


def clamp(lam: float, lo: float, hi: float) -> float:
    """Clamp log-odds into ``[lo, hi]`` (rule 4.5)."""
    return max(lo, min(hi, lam))


def decay_toward(lam: float, target: float, dt: float, tau: float) -> float:
    """Relax ``lam`` toward ``target`` with time constant ``tau`` (rule 4.1).

    Exact exponential relaxation over ``dt`` seconds, so the result is
    independent of how ``dt`` is chopped into ticks (rule 7.3 determinism).
    """
    if dt <= 0.0:
        return lam
    return target + (lam - target) * math.exp(-dt / tau)
