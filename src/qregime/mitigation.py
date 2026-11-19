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

import functools
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


# ---------------------------------------------------------------------------
# Simulated ZNE under a real device calibration: bias, not only shots
# ---------------------------------------------------------------------------

#: hardware profile -> calibrated fake backend in qiskit-ibm-runtime. The IBM
#: profile's p_2q already comes from FakeTorino (see tool.HARDWARE_PROFILES).
CALIBRATED_BACKENDS = {"superconductor_ibm": "FakeTorino"}
MAX_QUBITS_FOR_ZNE_SIMULATION = 10   # exact density matrices: 4**n entries


@functools.lru_cache(maxsize=None)
def _load_backend(hardware: str):
    """(backend, noise model, None) for a calibrated profile, else (None, None, reason)."""
    name = CALIBRATED_BACKENDS.get(hardware)
    if name is None:
        return None, None, f"no calibrated device model for profile {hardware!r}"
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError:
        return None, None, "needs qiskit-ibm-runtime (pip install 'qregime[realistic]')"
    from qiskit_aer.noise import NoiseModel

    backend = getattr(fake_provider, name)()
    return backend, NoiseModel.from_backend(backend), None


def _fold_gates(circuit, scale: float):
    """Replace a share of the gates G by G G^dag G, spread evenly, so the
    gate count grows by ``scale`` (Mitiq-style local folding of every gate,
    so one- and two-qubit noise are both amplified). Measurements, delays
    and barriers are not folded. Returns the folded circuit and the scale
    actually reached (gate counts are integers)."""
    from qiskit import QuantumCircuit

    skip = {"measure", "delay", "barrier", "reset"}
    gates = [i for i, inst in enumerate(circuit.data) if inst.operation.name not in skip]
    m = len(gates)
    n_fold = min(m, round((scale - 1) / 2 * m)) if m else 0
    chosen = {gates[int(j * m / n_fold)] for j in range(n_fold)} if n_fold else set()
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for i, inst in enumerate(circuit.data):
        out.append(inst)
        if i in chosen:
            out.append(inst.operation.inverse(), inst.qubits, inst.clbits)
            out.append(inst)
    return out, (1 + 2 * n_fold / m) if m else 1.0


def _apply_readout_error(probs, flips):
    """Symmetric per-qubit readout flips; bit i of the index is qubit i."""
    import numpy as np

    n = len(flips)
    t = np.asarray(probs, dtype=float).reshape((2,) * n)
    for i, e in enumerate(flips):
        axis = n - 1 - i
        t = np.moveaxis(np.tensordot(np.array([[1 - e, e], [e, 1 - e]]), t, axes=([1], [axis])),
                        0, axis)
    return t.reshape(-1)


def _exponential_fit(scales, values, asymptote):
    """Least-squares fit of ln(y - a) vs. scale; returns (y(0), intercept weights) or None."""
    import numpy as np

    lifted = np.asarray(values) - asymptote
    if np.any(lifted <= 0):
        return None
    slope, intercept = np.polyfit(scales, np.log(lifted), 1)
    return asymptote + float(np.exp(intercept)), _intercept_weights(scales)


def simulate_zne(qasm_path: str, hardware: str = "superconductor_ibm",
                 precision: float = DEFAULT_PRECISION,
                 scale_factors=DEFAULT_SCALE_FACTORS, max_shots: int = 10 ** 7,
                 seed: int = 7) -> dict:
    """Run ZNE on a calibrated device model and measure its bias.

    The circuit is transpiled to the device (noise-aware layout), its
    gates are folded to each scale factor, and the schedule is
    padded with idle delays, so folding also lengthens idle decoherence.
    Each scaled circuit is simulated exactly (density matrix) with the
    device's gate errors, T1/T2 relaxation and readout errors. The metric is
    qregime's success fraction: probability of landing in the ideal output
    support (ideal value 1). Exponential (asymptote = support / 2**n) and
    Richardson extrapolations are compared with 1. ZNE is ``viable`` when
    the least-biased method's bias is within ``precision`` and the shots it
    needs for that precision stay within ``max_shots``. The bias is reported
    with and without ideal readout-error mitigation, because readout error
    is not amplified by folding.

    Not modelled: coherent over-rotations, crosstalk, leakage and drift,
    which the fake-backend calibration does not contain.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import PassManager, generate_preset_pass_manager
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, BasisTranslator, PadDelay
    from qiskit_aer import AerSimulator

    backend, noise_model, reason = _load_backend(hardware)
    if backend is None:
        return {"available": False, "reason": reason}
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    qc.remove_final_measurements()
    n = qc.num_qubits
    if n > MAX_QUBITS_FOR_ZNE_SIMULATION:
        return {"available": False,
                "reason": f"{n} qubits > {MAX_QUBITS_FOR_ZNE_SIMULATION} (exact density-matrix limit)"}
    support = Statevector.from_instruction(qc).probabilities() > 1e-9
    asymptote = float(support.sum()) / 2 ** n
    if asymptote > 0.5:
        return {"available": False,
                "reason": f"ideal output support covers {asymptote:.0%} of bitstrings: "
                          "the success fraction cannot detect errors"}

    target = backend.target
    # noise-aware layout (level 3: VF2 scores layouts by calibrated error;
    # level 1 keeps a trivial layout whenever it fits) but no gate
    # cancellation anywhere: the level-0 init stage and no optimization
    # stage, so the circuit stays the one the user asked about (a mirror
    # U U^dag would otherwise shrink or vanish)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      seed_transpiler=seed)
    pm.init = generate_preset_pass_manager(optimization_level=0, backend=backend).init
    pm.optimization = None
    tqc = pm.run(qc)
    if not any(inst.operation.num_qubits == 2 for inst in tqc.data):
        return {"available": False, "reason": "no two-qubit gates to fold"}
    physical = tqc.layout.final_index_layout()
    readout = [target["measure"][(q,)].error or 0.0 for q in physical]
    # folded inverses (e.g. sxdg) are translated to native gates, then the
    # circuit is scheduled; no optimization, so the folds survive
    schedule = PassManager([BasisTranslator(SessionEquivalenceLibrary, target.operation_names,
                                            target=target),
                            ALAPScheduleAnalysis(target=target), PadDelay(target=target)])
    sim = AerSimulator(method="density_matrix", noise_model=noise_model,
                       seed_simulator=seed)

    scales, y_noisy, y_rem = [], [], []
    for scale in scale_factors:
        folded, reached = _fold_gates(tqc, float(scale))
        timed = schedule.run(folded)
        timed.save_probabilities(qubits=physical)
        probs = np.asarray(sim.run(timed).result().data()["probabilities"])
        scales.append(reached)
        y_rem.append(float(probs[support].sum()))
        y_noisy.append(float(_apply_readout_error(probs, readout)[support].sum()))

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
        "device_model": f"{backend.name} calibration (qiskit-ibm-runtime fake backend)",
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
        "not_modelled": "coherent over-rotations, crosstalk, leakage, drift",
    }
