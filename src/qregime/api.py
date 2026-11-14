"""
Public, English-language API of qregime.

The research code underneath (``advisor``, ``resources``, ``regime``, ...)
was written during an exploratory research project and its inline
documentation is in Spanish. This module is the stable, documented entry
point: it wraps those functions, translates their output keys to English,
and makes the known limitations explicit instead of hiding them.

Typical use::

    import qregime

    report = qregime.decide_regime("my_circuit.qasm", hardware="superconductor_ibm")
    print(report["regime"])
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Optional

from .advisor import profile_circuit, recommend_techniques
from .partial_qec import best_feasible_within_hard_budget
from .regime import (
    DEFAULT_PHYSICAL_QUBIT_BUDGET,
    DEFAULT_RELIABILITY_TARGET,
    decide_regime_formal,
)
from .resources import DEFAULT_NOISE_MODEL, NOISE_MODELS, estimate_open_ftqc_with_factories
from .tool import HARDWARE_PROFILES, HardwareProfile, analyze_circuit, simulate_expectation

# Largest circuit for which the empirical (noisy-simulation) QEM estimate is
# attempted. It uses an exact statevector, whose memory cost doubles per qubit;
# 20 is the largest size measured to stay fast in practice (see the paper,
# "Bugs found and fixed").
MAX_QUBITS_FOR_SIMULATION = 20

# The noisy-simulation success metric is only informative when the ideal output
# distribution is concentrated. Above this fraction of all bitstrings, the
# analytic proxy is used instead (see decide_regime).
MAX_INFORMATIVE_SUPPORT_FRACTION = 0.5

_RECOMMENDATION_NAMES = {
    "teletransportacion_RZZ": "rzz_gate_teleportation",
    "zx_calculus": "zx_calculus_tcount_reduction",
    "and_gate_descomputo_medicion": "measurement_based_uncomputation",
    "qcp_medicion_intermedia_general": "bounded_qcp_mid_circuit_measurement",
    "posible_atajo_clasico_bond_dimension": "possible_classical_shortcut",
}


_NON_UNITARY_OPS = ("measure", "reset", "barrier")


@contextlib.contextmanager
def _gate_only_qasm(qasm_path: str):
    """Yield a path to a measurement-free copy of ``qasm_path``.

    The analysis layers (pyzx parsing, the QCP tracker, the statevector
    simulation) only understand unitary gates, yet almost every real QASM
    file ends in ``measure`` statements. Final measurements and barriers are
    dropped here, since they do not change the regime decision. Mid-circuit
    measurements and resets do change it (dynamic circuits), so those still
    raise an explicit error instead of being silently removed.
    """
    from qiskit import QuantumCircuit

    from .qasm_export import safe_qasm2_dumps

    qc = QuantumCircuit.from_qasm_file(qasm_path)
    if not any(inst.operation.name in _NON_UNITARY_OPS for inst in qc.data):
        yield qasm_path
        return

    stripped = qc.remove_final_measurements(inplace=False)
    leftover = {inst.operation.name for inst in stripped.data} & {"measure", "reset"}
    if leftover:
        raise ValueError(
            f"{qasm_path}: mid-circuit {'/'.join(sorted(leftover))} found. Dynamic "
            "circuits are not supported yet; only final measurements are accepted.")
    gates_only = QuantumCircuit(*stripped.qregs, name=stripped.name)
    for inst in stripped.data:
        if inst.operation.name != "barrier":
            gates_only.append(inst)

    fd, tmp_path = tempfile.mkstemp(suffix=".qasm")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(safe_qasm2_dumps(gates_only))
        yield tmp_path
    finally:
        os.remove(tmp_path)


def _english_ftqc_reason(reason):
    """The estimator's infeasibility reasons are in Spanish (research code)."""
    if reason is None:
        return None
    if "d=51" in reason:
        return ("no code distance up to d=51 meets the logical error budget "
                "(p_2q is too close to the fitted threshold)")
    if "umbral" in reason:
        return "p_2q is at or above the fitted surface-code threshold"
    return reason


def profile(qasm_path: str) -> dict:
    """Structural profile of an OpenQASM 2 circuit.

    Hardware-agnostic by construction. Returns qubit/gate/T counts,
    interaction-graph degree statistics, qubit-reuse potential, spectral
    sparsifiability, the diagonal-cost-layer motif fraction (the signal
    behind Proposition 1 of the paper), and the final-state entanglement
    fraction (``None`` above 20 qubits).
    """
    with _gate_only_qasm(qasm_path) as path:
        return profile_circuit(path)


def recommend(qasm_path: str, circuit_profile: Optional[dict] = None) -> dict:
    """Which circuit-reduction techniques are worth trying on this circuit.

    Returns ``{technique: {"recommended": bool | "WARNING", "reason": str}}``;
    "WARNING" means low confidence: verify before relying on it.
    Pass a ``circuit_profile`` you already computed with :func:`profile` to
    avoid recomputing it (this was the cause of a 4x slowdown, fixed in v0.1).

    Note: the ``reason`` strings are currently in Spanish (the language the
    research was done in); translating them is an open, welcome contribution.
    """
    with _gate_only_qasm(qasm_path) as path:
        _, recs = recommend_techniques(path, profile=circuit_profile)
    out = {}
    for key, value in recs.items():
        rec = value.get("recomendada")
        out[_RECOMMENDATION_NAMES.get(key, key)] = {
            "recommended": "WARNING" if rec == "ADVERTENCIA" else rec,
            "reason": value.get("razon"),
        }
    return out


def estimate_ftqc(num_logical_qubits: int, tcount: int, p_phys: float,
                  error_budget: float = 1e-2,
                  noise_model: str = DEFAULT_NOISE_MODEL) -> dict:
    """Physical-qubit and runtime estimate for full fault tolerance.

    Rotated surface code, with code distance chosen from this project's own
    threshold fit for ``noise_model`` (one of ``NOISE_MODELS``: "sd6",
    standard circuit-level depolarizing; "si1000", superconducting-inspired;
    "toy", cheap idles, used for trapped ions; all fitted with Deltakit, see
    research/qec/fit_threshold_deltakit.py), plus Litinski (2019) 15-to-1
    magic-state factories. ``p_phys`` is the physical two-qubit error rate.
    """
    r = estimate_open_ftqc_with_factories(
        num_logical_qubits=num_logical_qubits, tcount=tcount, p_phys=p_phys,
        error_budget=error_budget, verbose=False, noise_model=noise_model)
    if not r["feasible"]:
        r = dict(r, reason=_english_ftqc_reason(r["reason"]))
    return r


# Above this fraction of a fit's threshold, the noise-model choice can move
# the answer by orders of magnitude, so reports flag it.
NEAR_THRESHOLD_FRACTION = 0.5


def estimate_ftqc_band(num_logical_qubits: int, tcount: int, p_phys: float,
                       error_budget: float = 1e-2) -> dict:
    """:func:`estimate_ftqc` under every fitted noise model, as a range.

    The threshold differs by ~2x between noise models (0.40-0.85%), which
    near threshold changes the answer far more than the fit uncertainty does.
    Report this band instead of a single number whenever the noise model of
    the target hardware is not well known.

    Returns ``{"per_model": {model: {"feasible", "code_distance",
    "physical_qubits", "p_over_p_th"}}, "min_qubits", "max_qubits",
    "feasible_models", "near_threshold"}``; ``max_qubits`` is ``None`` when
    at least one model finds FTQC infeasible (unbounded).
    """
    from .resources import _load_fit

    per_model = {}
    for model in NOISE_MODELS:
        r = estimate_ftqc(num_logical_qubits, tcount, p_phys, error_budget, noise_model=model)
        per_model[model] = {
            "feasible": r["feasible"],
            "code_distance": r.get("codeDistance"),
            "physical_qubits": r.get("physicalQubits"),
            "p_over_p_th": p_phys / _load_fit(model)["p_th"],
        }
    costs = [m["physical_qubits"] for m in per_model.values() if m["feasible"]]
    feasible = [k for k, m in per_model.items() if m["feasible"]]
    return {
        "per_model": per_model,
        "min_qubits": min(costs) if costs else None,
        "max_qubits": max(costs) if len(costs) == len(per_model) else None,
        "feasible_models": feasible,
        "near_threshold": any(m["p_over_p_th"] > NEAR_THRESHOLD_FRACTION
                              for m in per_model.values()),
    }


def best_within_budget(num_logical_qubits: int, tcount: int, p_phys: float,
                       physical_qubit_budget: int,
                       noise_model: str = DEFAULT_NOISE_MODEL) -> dict:
    """Best reliability reachable under a hard physical-qubit budget.

    Instead of a pass/fail answer, returns the largest affordable code
    distance and the confidence quartile its reliability falls into.
    Proposition 3 of the paper proves that this answer can only improve as
    the budget grows.

    Limitation: small code distances are an extrapolation of a threshold model
    calibrated for full fault tolerance, not a validated partial-QEC scheme.
    """
    raw = best_feasible_within_hard_budget(
        num_logical_qubits=num_logical_qubits, tcount=tcount, p_phys=p_phys,
        physical_qubit_budget=physical_qubit_budget, verbose=False,
        noise_model=noise_model)
    if not raw.get("realizable"):
        return {"realizable": False, "reason": raw.get("reason")}
    rel = raw["reliability"]
    quartile = 1 if rel < 0.25 else 2 if rel < 0.5 else 3 if rel < 0.75 else 4
    label = {1: "low", 2: "low-moderate", 3: "moderate", 4: "high"}[quartile]
    return {
        "realizable": True,
        "code_distance": raw["chosen_d"],
        "cost_physical_qubits": raw["cost_qubits"],
        "reliability": rel,
        "confidence_quartile": quartile,
        "confidence_label": label,
        "note": "Not fault tolerance: the best point this budget allows, "
                "reported with its actual confidence, not rounded up.",
    }


def decide_regime(qasm_path: str, hardware: str = "superconductor_ibm",
                  physical_qubit_budget: int = DEFAULT_PHYSICAL_QUBIT_BUDGET,
                  reliability_target: float = DEFAULT_RELIABILITY_TARGET,
                  qem_success_fraction: Optional[float] = None,
                  shots: int = 150) -> dict:
    """QEM, partial QEC, or FTQC for this circuit on this hardware?

    Solves ``argmin cost(regime)`` subject to ``reliability >= target`` and
    ``cost <= budget``. If no regime satisfies both, the answer is an explicit,
    quantified gap rather than a forced recommendation.

    The QEM reliability comes from, in order of preference:
      1. ``qem_success_fraction`` if you pass one (e.g. measured on hardware);
      2. a noisy simulation (depolarizing two-qubit noise at the hardware
         profile's error rate), for circuits up to 20 qubits whose ideal
         output is concentrated (support <= 50% of all bitstrings) -- the
         metric is the fraction of shots landing in that support, which
         cannot detect errors when the support is nearly everything;
      3. otherwise a crude analytic proxy, ``(1 - p_2q) ** two_qubit_count``
         (two-qubit gate errors only), labelled as such in the output.
    """
    if hardware not in HARDWARE_PROFILES:
        raise ValueError(f"unknown hardware profile {hardware!r}; "
                         f"choose one of {sorted(HARDWARE_PROFILES)}")
    hw = HARDWARE_PROFILES[hardware]
    with _gate_only_qasm(qasm_path) as path:
        report = _decide_regime(path, hw, physical_qubit_budget, reliability_target,
                                qem_success_fraction, shots)
    report["qasm_path"] = qasm_path
    return report


def _decide_regime(qasm_path, hw, physical_qubit_budget, reliability_target,
                   qem_success_fraction, shots):
    prof = profile_circuit(qasm_path)

    proxy = (1 - hw.p_2q) ** prof["two_qubit_count"]
    if qem_success_fraction is not None:
        qem_source = "user_supplied"
    elif prof["num_qubits"] <= MAX_QUBITS_FOR_SIMULATION:
        sim = simulate_expectation(qasm_path, hw, shots=shots, verbose=False)
        support_fraction = sim["ideal_support_size"] / 2 ** prof["num_qubits"]
        if support_fraction <= MAX_INFORMATIVE_SUPPORT_FRACTION:
            qem_success_fraction = sim["success_fraction_measured"]
            qem_source = f"noisy_simulation ({shots} shots)"
        else:
            # The simulation metric counts shots landing in the ideal output
            # support. When that support is (nearly) every bitstring -- as for
            # QAOA, QFT on superposed inputs, VQE ansatze -- every shot "succeeds"
            # regardless of noise, so the metric carries no information.
            qem_success_fraction = proxy
            qem_source = (f"analytic_proxy (1-p_2q)^two_qubit_count; noisy simulation "
                          f"not used: ideal output support covers {support_fraction:.0%} "
                          f"of all bitstrings, so it cannot detect errors")
    else:
        qem_success_fraction = proxy
        qem_source = "analytic_proxy (1-p_2q)^two_qubit_count"

    raw = decide_regime_formal(
        qasm_path, tcount=prof["tcount"], num_qubits=prof["num_qubits"],
        p_phys=hw.p_2q, qem_success_fraction=qem_success_fraction,
        reliability_target=reliability_target,
        physical_qubit_budget=physical_qubit_budget,
        verbose=False, profile=prof, noise_model=hw.noise_model)

    regime = raw["chosen_regime"]
    if regime.startswith("QEC_PARCIAL"):
        regime = regime.replace("QEC_PARCIAL", "PARTIAL_QEC")
    elif regime.startswith("BRECHA"):
        regime = "GAP"

    gap = None
    if raw["gap_report"] is not None:
        gap = {
            "qem_reliability_shortfall": raw["gap_report"]["gap_de_confiabilidad_QEM"],
            "ftqc_qubit_overrun": raw["gap_report"]["gap_de_qubits_FTQC"],
            "best_partial_qec_attempt": raw["gap_report"]["qec_parcial_mejor_intento"],
        }

    return {
        "qasm_path": qasm_path,
        "hardware": hw.name,
        "p_2q": hw.p_2q,
        "noise_model": hw.noise_model,
        "regime": regime,
        "cost_physical_qubits": raw["chosen_cost_qubits"],
        "qem": {"cost_qubits": raw["cost_qem_qubits"],
                "reliability": qem_success_fraction,
                "reliability_source": qem_source,
                "meets_target": raw["qem_meets_target"]},
        "ftqc": {"cost_qubits": raw["cost_ftqc_qubits"],
                 "fits_budget": raw["ftqc_feasible_in_budget"],
                 "meets_target": raw["ftqc_meets_target"],
                 "infeasible_reason": _english_ftqc_reason(raw["ftqc_infeasible_reason"])},
        "physical_qubit_budget": physical_qubit_budget,
        "reliability_target": reliability_target,
        "gap": gap,
        "classical_shortcut_advisory": raw["classical_shortcut_advisory"],
        "ftqc_noise_band": estimate_ftqc_band(prof["num_qubits"], prof["tcount"], hw.p_2q),
        "profile": prof,
    }


def analyze(qasm_path: str, hardware: str = "superconductor_ibm",
            simulate: bool = False, shots: int = 150) -> dict:
    """Full consolidated report: profile, recommendations, structural
    feasibility on the hardware's connectivity, and FTQC estimates with and
    without qubit reuse. With ``simulate=True`` it also runs the noisy
    simulation (only practical up to ~20 qubits)."""
    with _gate_only_qasm(qasm_path) as path:
        return analyze_circuit(path, hardware, verbose=False,
                               simulate=simulate, shots=shots)


__all__ = [
    "profile", "recommend", "estimate_ftqc", "best_within_budget",
    "decide_regime", "analyze", "HARDWARE_PROFILES", "HardwareProfile",
    "MAX_QUBITS_FOR_SIMULATION", "NOISE_MODELS", "estimate_ftqc_band",
]
