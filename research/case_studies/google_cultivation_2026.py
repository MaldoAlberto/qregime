"""Case study: magic state cultivation on Google Willow (arXiv:2512.13908;
Science, 15 Oct 2026).

Rosenfeld, Gidney, Satzinger et al. grew |T> states in a distance-3 color code
on Willow (105 qubits; 12-14 used for cultivation) and bounded the logical
error at 1(1)e-4, retaining 8% of attempts. Simulations in the paper use
SI1000 noise with p = 2.3e-3, and the error is expected to scale as p^3.
A distance-5 graft into a surface-code-compatible encoding (54 qubits)
performed worse than simulated, which the paper attributes possibly to
leakage, not included in its noise model.

This script asks:

  1. qregime: is a 1e-4 T state good enough for a real circuit, the paper's
     flagship (16 logical qubits, T-count 3,930)? If not, what physical error
     rate would cultivation need (p^3 scaling), and what does adding one
     15-to-1 distillation round on cultivated inputs buy?
  2. Deltakit: how much can leakage degrade a distance-5 surface-code memory
     at the paper's simulated noise strength (SI1000, p = 2.3e-3)? Deltakit's
     SI1000Noise has a leakage parameter pL (arXiv:2411.10343).

    pip install 'qregime[deltakit]'
    python research/case_studies/google_cultivation_2026.py   # writes google_cultivation_2026_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qec"))

import qregime

# arXiv:2512.13908 (read in the HTML version)
T_ERROR_MEASURED = 1e-4        # upper bound, 1(1)e-4
KEEP_FRACTION = 0.08           # "retaining 8% of attempts"
P_SIM = 2.3e-3                 # SI1000 strength used for the cultivation simulations
CULTIVATION_QUBITS = 14        # qubits used in the cultivation experiment (Fig. 3)
INJECTION_ONLY_ERROR = 4.4e-3  # "4.4(2)e-3" without cultivation

# qregime's flagship circuit (paper, README): 16 logical qubits, T-count 3,930.
N_LOGICAL, TCOUNT = 16, 3930
ERROR_BUDGET = 1e-2
LEAKAGE_SCAN = (0.0, 1e-4, 3e-4, 1e-3, 2.3e-3)


def t_state_analysis() -> dict:
    """What T-state error does the flagship circuit need, and what would it take?"""
    needed = ERROR_BUDGET / TCOUNT  # whole budget spent on T states (optimistic)
    success_measured = (1 - T_ERROR_MEASURED) ** TCOUNT
    # p^3 scaling anchored at the measured bound (an upper bound, so the
    # p it implies is conservative)
    p_needed = P_SIM * (needed / T_ERROR_MEASURED) ** (1 / 3)
    # one 15-to-1 round on cultivated inputs: output error ~ 35 eps_in^3
    distilled = 35 * T_ERROR_MEASURED ** 3
    ftqc_willow = qregime.estimate_ftqc(N_LOGICAL, TCOUNT, P_SIM, noise_model="si1000")
    d = ftqc_willow.get("codeDistance")
    return {
        "t_error_needed_for_1pct_budget": needed,
        "gap_factor_vs_measured": T_ERROR_MEASURED / needed,
        "circuit_success_from_T_errors_alone_at_measured": success_measured,
        "p_needed_for_cultivation_alone_p3_scaling": p_needed,
        "one_15to1_round_on_cultivated_inputs_error": distilled,
        "attempts_per_accepted_state": 1 / KEEP_FRACTION,
        "cultivation_qubits": CULTIVATION_QUBITS,
        "qregime_ftqc_at_p_sim_si1000": {
            k: ftqc_willow.get(k) for k in ("feasible", "codeDistance", "physicalQubits",
                                            "physicalQubitsForFactories", "reason")},
        "litinski_15to1_factory_qubits_at_that_d": (22 * d * d) if d else None,
    }


def leakage_scan() -> list[dict]:
    from deltakit.explorer.qpu import SI1000Noise
    from fit_threshold_deltakit import measure_circuit, memory_circuit

    rows = []
    t0 = time.time()
    for pl in LEAKAGE_SCAN:
        m = measure_circuit(memory_circuit(5, SI1000Noise(p=P_SIM, pL=pl)))
        rows.append({"pL": pl, **m})
        print(f"  d=5 SI1000 p={P_SIM}  pL={pl:.1e}  p_L={m['p_L']:.4f} "
              f"+/- {m['stderr']:.4f}   ({time.time() - t0:.0f} s)", flush=True)
    base = rows[0]["p_L"]
    for r in rows:
        r["factor_vs_no_leakage"] = r["p_L"] / base
    return rows


def main():
    out = {"source": "arXiv:2512.13908 (Science, 15 Oct 2026)",
           "inputs": {"t_error_measured": T_ERROR_MEASURED, "keep_fraction": KEEP_FRACTION,
                      "p_sim_si1000": P_SIM, "n_logical": N_LOGICAL, "tcount": TCOUNT,
                      "error_budget": ERROR_BUDGET}}
    out["t_state"] = t_state_analysis()
    for k, v in out["t_state"].items():
        print(f"{k}: {v}")
    print("Deltakit leakage scan (distance-5 surface-code memory, 5 rounds):")
    out["leakage_scan"] = leakage_scan()
    path = os.path.join(HERE, "google_cultivation_2026_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
