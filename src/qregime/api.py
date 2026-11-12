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

from typing import Optional

from ._qasm_input import unitary_qasm
from .advisor import profile_circuit, recommend_techniques
from .partial_qec import best_feasible_within_hard_budget
from .regime import (
    DEFAULT_PHYSICAL_QUBIT_BUDGET,
    DEFAULT_RELIABILITY_TARGET,
    decide_regime_formal,
)
from .resources import effective_tcount, estimate_open_ftqc_with_factories
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


def profile(qasm_path: str) -> dict:
    """Structural profile of an OpenQASM 2 circuit.

    Hardware-agnostic by construction. Returns qubit/gate/T counts,
    interaction-graph degree statistics, qubit-reuse potential, spectral
    sparsifiability, the diagonal-cost-layer motif fraction (the signal
    behind Proposition 1 of the paper), and the final-state entanglement
    fraction (``None`` above 20 qubits).

    ``tcount`` is pyzx's count (every non-Clifford phase counts as one T);
    ``tcount_exact`` and ``rotation_count`` split it into exact T gates and
    arbitrary-angle rotations, whose synthesis costs many T gates each.
    Final measurements and barriers are ignored; mid-circuit measurements
    raise ``ValueError``.
    """
    with unitary_qasm(qasm_path) as path:
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
    with unitary_qasm(qasm_path) as path:
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
                  error_budget: float = 1e-2) -> dict:
    """Physical-qubit and runtime estimate for full fault tolerance.

    Rotated surface code, with code distance chosen from this project's own
    Stim + PyMatching threshold fit, plus Litinski (2019) 15-to-1 magic-state
    factories. ``p_phys`` is the physical two-qubit error rate.
    """
    return estimate_open_ftqc_with_factories(
        num_logical_qubits=num_logical_qubits, tcount=tcount, p_phys=p_phys,
        error_budget=error_budget, verbose=False)


def best_within_budget(num_logical_qubits: int, tcount: int, p_phys: float,
                       physical_qubit_budget: int) -> dict:
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
        physical_qubit_budget=physical_qubit_budget, verbose=False)
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

    The FTQC and partial-QEC estimates charge the Clifford+T synthesis cost
    of arbitrary-angle rotations (``ftqc.tcount_effective``).
    """
    if hardware not in HARDWARE_PROFILES:
        raise ValueError(f"unknown hardware profile {hardware!r}; "
                         f"choose one of {sorted(HARDWARE_PROFILES)}")
    if not 0 < reliability_target <= 1:
        raise ValueError(f"reliability_target must be in (0, 1], got {reliability_target}")
    if physical_qubit_budget < 1:
        raise ValueError(f"physical_qubit_budget must be >= 1, got {physical_qubit_budget}")
    with unitary_qasm(qasm_path) as path:
        return _decide_regime(qasm_path, path, hardware, physical_qubit_budget,
                              reliability_target, qem_success_fraction, shots)


def _decide_regime(qasm_path, path, hardware, physical_qubit_budget,
                   reliability_target, qem_success_fraction, shots):
    hw = HARDWARE_PROFILES[hardware]
    prof = profile_circuit(path)

    proxy = (1 - hw.p_2q) ** prof["two_qubit_count"]
    if qem_success_fraction is not None:
        qem_source = "user_supplied"
    elif prof["num_qubits"] <= MAX_QUBITS_FOR_SIMULATION:
        sim = simulate_expectation(path, hw, shots=shots, verbose=False)
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
        path, tcount=prof["tcount_exact"], num_qubits=prof["num_qubits"],
        p_phys=hw.p_2q, qem_success_fraction=qem_success_fraction,
        reliability_target=reliability_target,
        physical_qubit_budget=physical_qubit_budget,
        verbose=False, profile=prof, rotation_count=prof["rotation_count"])

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
        "regime": regime,
        "cost_physical_qubits": raw["chosen_cost_qubits"],
        "qem": {"cost_qubits": raw["cost_qem_qubits"],
                "reliability": qem_success_fraction,
                "reliability_source": qem_source,
                "meets_target": raw["qem_meets_target"]},
        "ftqc": {"cost_qubits": raw["cost_ftqc_qubits"],
                 "fits_budget": raw["ftqc_fits_budget"],
                 "reliability": raw["ftqc_reliability"],
                 "meets_target": raw["ftqc_meets_target"],
                 "tcount_effective": raw["tcount_ftqc"],
                 "infeasible_reason": raw["ftqc_infeasible_reason"]},
        "physical_qubit_budget": physical_qubit_budget,
        "reliability_target": reliability_target,
        "gap": gap,
        "classical_shortcut_advisory": raw["classical_shortcut_advisory"],
        "profile": prof,
    }


def analyze(qasm_path: str, hardware: str = "superconductor_ibm",
            simulate: bool = False, shots: int = 150) -> dict:
    """Full consolidated report: profile, recommendations, structural
    feasibility on the hardware's connectivity, and FTQC estimates with and
    without qubit reuse. With ``simulate=True`` it also runs the noisy
    simulation (only practical up to ~20 qubits)."""
    with unitary_qasm(qasm_path) as path:
        report = analyze_circuit(path, hardware, verbose=False,
                                 simulate=simulate, shots=shots)
    report["qasm_path"] = qasm_path
    return report


__all__ = [
    "profile", "recommend", "estimate_ftqc", "best_within_budget",
    "decide_regime", "analyze", "HARDWARE_PROFILES", "HardwareProfile",
    "MAX_QUBITS_FOR_SIMULATION",
]
