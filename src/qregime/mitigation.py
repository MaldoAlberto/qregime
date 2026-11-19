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

import numpy as np

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


# ---------------------------------------------------------------------------
# Simulated ZNE on a device calibration: bias, not only shots
# ---------------------------------------------------------------------------

def _exponential_fit(scales, values, asymptote):
    """Least-squares fit of ln(y - a) vs. scale; returns (y(0), intercept weights) or None."""
    lifted = np.asarray(values) - asymptote
    if np.any(lifted <= 0):
        return None
    slope, intercept = np.polyfit(scales, np.log(lifted), 1)
    return asymptote + float(np.exp(intercept)), _intercept_weights(scales)


def simulate_zne(qasm_path: str, calibration="ibm_torino",
                 precision: float = DEFAULT_PRECISION,
                 scale_factors=DEFAULT_SCALE_FACTORS, max_shots: int = 10 ** 7,
                 layout=None) -> dict:
    """Run ZNE on a calibrated device model and measure its bias.

    ``calibration`` is a bundled calibration name, a JSON path or a dict
    (see ``qregime.calibration``). The circuit is mapped and compiled to the
    device's native gates, the native ops are locally folded (G G^dag G) to
    each scale factor, and each scaled circuit is simulated exactly with the
    device's gate errors, idle decoherence and readout errors. The metric is
    qregime's success fraction (ideal value 1). Exponential (asymptote =
    support / 2**n) and Richardson extrapolations are compared with 1. ZNE is
    ``viable`` when the least-biased method's bias is within ``precision``
    and the shots it needs for that precision stay within ``max_shots``.
    The bias is reported with and without ideal readout-error mitigation,
    because readout error is not amplified by folding.

    Not modelled: leakage, crosstalk, drift; coherent errors only if the
    calibration sets ``zz_overrotation``.
    """
    from . import calibration as _calibration

    result, reason = _calibration.zne_values(qasm_path, calibration, scale_factors, layout)
    if result is None:
        return {"available": False, "reason": reason}
    info, scales, y_noisy, y_rem = result
    asymptote = info["asymptote"]
    physical = info["physical_qubits"]

    def extrapolate(values):
        exp_fit = _exponential_fit(scales, values, asymptote)
        # Richardson = Lagrange interpolation at 0: y0 = sum c_i y_i,
        # c_i = prod_{j != i} x_j / (x_j - x_i)
        c = [math.prod(xj / (xj - x) for xj in scales if xj != x) for x in scales]
        richardson = float(sum(ci * y for ci, y in zip(c, values)))
        r_unit = sum(ci * ci * y * (1 - y) for ci, y in zip(c, values))
        out = {"values": values,
               "richardson": richardson, "richardson_bias": richardson - 1.0,
               "richardson_total_shots": len(scales) * max(1, math.ceil(r_unit / precision ** 2)),
               "exponential": None, "exponential_bias": None, "exponential_total_shots": None}
        if exp_fit is not None:
            y0, w = exp_fit
            # Var(y0) = (y0 - a)^2 * sum w^2 Var(ln(y - a)), Var(ln(y - a)) = y(1-y)/(N (y-a)^2)
            unit = (y0 - asymptote) ** 2 * sum(
                wi * wi * y * (1 - y) / (y - asymptote) ** 2 for wi, y in zip(w, values))
            out.update(exponential=y0, exponential_bias=y0 - 1.0,
                       exponential_total_shots=len(scales) * max(1, math.ceil(unit / precision ** 2)))
        return out

    raw = extrapolate(y_noisy)
    rem = extrapolate(y_rem)
    # method advice: the extrapolation (after readout mitigation) whose bias
    # on this device model is smallest
    bias, best = min((abs(rem[f"{name}_bias"]), name) for name in ("exponential", "richardson")
                     if rem[f"{name}_bias"] is not None)
    return {
        "available": True,
        "device_model": info["device_model"],
        "physical_qubits": list(physical),
        "scale_factors": scales,
        "precision": precision,
        "unmitigated": y_noisy[0],
        "zne": raw,
        "zne_with_readout_mitigation": rem,
        "best_method": f"{best} + readout mitigation",
        "best_abs_bias": bias,
        "total_shots": rem[f"{best}_total_shots"],
        "max_shots": max_shots,
        "viable": bias <= precision and (rem[f"{best}_total_shots"] or math.inf) <= max_shots,
        "not_modelled": "leakage, crosstalk, drift",
    }
