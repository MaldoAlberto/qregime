"""Sampling cost of error mitigation: how many shots does ZNE need?

qregime's QEM reliability is the *unmitigated* success fraction F. Zero-noise
extrapolation (ZNE) can recover the ideal value from it, but the price is
shots: the extrapolation amplifies the statistical error of the noisy
measurements, and the amplification grows as F shrinks.

Model (exponential ZNE, the best case for pure depolarizing noise):
  - at noise scale lambda the measured value is y(lambda) = F ** lambda;
  - each scale factor is run with N shots, so Var(ln y) = (1 - y) / (N y);
  - ln y is fitted linearly in lambda by least squares and extrapolated to
    lambda = 0, so the mitigated value has
        Var = sum_i w_i**2 (1 - y_i) / (N y_i),
    where w are the least-squares intercept weights.

Checked against Mitiq's ExpFactory on depolarizing mirror circuits
(examples/qregime_to_mitiq.py): predicted spread within about 30% of the
measured one from F = 0.99 down to F = 0.16. The model assumes the
extrapolation is unbiased. On real hardware, noise that does not decay
exponentially adds bias, which more shots do not remove.
"""
from __future__ import annotations

import math

DEFAULT_SCALE_FACTORS = (1.0, 2.0, 3.0)
DEFAULT_PRECISION = 0.01


def _intercept_weights(scale_factors) -> list[float]:
    """Least-squares weights giving the intercept of a line fitted at ``scale_factors``."""
    n = len(scale_factors)
    mean = sum(scale_factors) / n
    sxx = sum((x - mean) ** 2 for x in scale_factors)
    return [1 / n - mean * (x - mean) / sxx for x in scale_factors]


def zne_sampling_cost(reliability: float, precision: float = DEFAULT_PRECISION,
                      scale_factors=DEFAULT_SCALE_FACTORS) -> dict:
    """Shots exponential ZNE needs to reach ``precision`` (one standard deviation).

    ``reliability`` is the unmitigated success fraction F, in (0, 1].
    Returns the shots per scale factor and the total over all scale factors.
    """
    if not 0 < precision < 1:
        raise ValueError("precision must be in (0, 1)")
    factors = tuple(float(x) for x in scale_factors)
    if len(factors) < 2 or len(set(factors)) < 2 or min(factors) < 1:
        raise ValueError("need at least two distinct scale factors >= 1")
    base = {"method": "zne_exponential", "precision": precision,
            "scale_factors": list(factors),
            "assumption": "unbiased exponential extrapolation (pure depolarizing noise)"}
    if reliability <= 0 or not math.isfinite(reliability):
        return {**base, "feasible": False, "shots_per_circuit": None, "total_shots": None,
                "reason": "unmitigated reliability is zero: no signal to extrapolate"}
    reliability = min(reliability, 1.0)
    weights = _intercept_weights(factors)
    variance_unit = sum(w * w * (1 - reliability ** x) / reliability ** x
                        for w, x in zip(weights, factors))
    per_circuit = max(1, math.ceil(variance_unit / precision ** 2))
    total = per_circuit * len(factors)
    return {**base, "feasible": True, "shots_per_circuit": per_circuit, "total_shots": total}
