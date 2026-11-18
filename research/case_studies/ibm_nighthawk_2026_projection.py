"""What the IBM Nighthawk r2 memory results mean for a real circuit.

Reads ibm_nighthawk_2026_results.json (written by ibm_nighthawk_2026.py) and
runs qregime's fault-tolerance estimators for the paper's flagship circuit
(16 logical qubits, T-count 3,930) at the Nighthawk two-qubit error rate,
under three surface-code models:

  qregime_si1000      qregime's default superconducting model: SI1000 noise,
                      threshold fitted with Deltakit (p_th ~0.40%).
  deltakit_median     the calibration-derived Deltakit model of
                      ibm_nighthawk_2026.py (median calibration, echo).
  measured_ml         the error per round and Lambda measured on hardware
                      (arXiv:2610.23840, Z memory, exact ML decoding).

Each model gives an error per round eps(d) = eps(3) * Lambda ** (-(d - 3) / 2).
qregime's model is per patch per d rounds, p_L(d) = A * (p / p_th) ** ((d+1)/2),
so p_L(d) = d * eps(d) is fitted to that form over d = 3..51 (least squares
on log p_L; the factor d is absorbed into A and p_th). The projection
extrapolates Lambda far beyond the measured d = 3..6, which the paper itself
warns weakens over long memories: read it as an optimistic bound.

    python research/case_studies/ibm_nighthawk_2026_projection.py
"""
from __future__ import annotations

import json
import os
from unittest import mock

import numpy as np

import qregime
from qregime import partial_qec, resources

HERE = os.path.dirname(os.path.abspath(__file__))
N_LOGICAL, TCOUNT = 16, 3930
P_2Q = 1.6e-3          # Nighthawk r2 median CZ error, day 2 (Table 1)
BUDGETS = (10_000, 100_000, 1_000_000)


def fit_from_eps(eps3: float, lam: float, p: float = P_2Q) -> dict:
    ds = np.arange(3, 53, 2)
    p_l = ds * eps3 * lam ** (-(ds - 3) / 2)
    slope, intercept = np.polyfit((ds + 1) / 2, np.log(p_l), 1)
    return {"A": float(np.exp(intercept)), "p_th": float(p / np.exp(slope)),
            "p_th_stderr": 0.0, "eps3": eps3, "lambda": lam}


def models() -> dict:
    with open(os.path.join(HERE, "ibm_nighthawk_2026_results.json")) as f:
        sim = json.load(f)
    base = next(r for r in sim["scale_scan"] if r["scale"] == 1.0)
    measured = sim["measured"]["Z"]
    return {
        "qregime_si1000": resources._load_fit("si1000"),
        "deltakit_median": fit_from_eps(base["eps_d3"], base["lambda"]),
        "measured_ml": fit_from_eps(measured["eps"]["3"], measured["lambda"]),
    }


def project(fit: dict) -> dict:
    with mock.patch.object(resources, "_load_fit", lambda *_a, **_k: fit), \
            mock.patch.object(partial_qec, "_load_fit", lambda *_a, **_k: fit):
        ftqc = qregime.estimate_ftqc(N_LOGICAL, TCOUNT, P_2Q)
        budget = {b: qregime.best_within_budget(N_LOGICAL, TCOUNT, P_2Q, b) for b in BUDGETS}
    return {"ftqc": {k: ftqc.get(k) for k in ("feasible", "codeDistance", "physicalQubits",
                                               "physicalQubitsForFactories", "reason")},
            "best_within_budget": budget}


def main():
    out = {"n_logical": N_LOGICAL, "tcount": TCOUNT, "p_2q": P_2Q, "models": {}}
    for name, fit in models().items():
        res = project(fit)
        out["models"][name] = {"fit": {k: fit[k] for k in ("A", "p_th")}, **res}
        f = res["ftqc"]
        print(f"{name:16s} p/p_th={P_2Q / fit['p_th']:.2f}  "
              + (f"FTQC d={f['codeDistance']}, {f['physicalQubits']:,} qubits"
                 if f["feasible"] else f"FTQC infeasible: {f['reason']}"))
        for b, r in res["best_within_budget"].items():
            print(f"    budget {b:>9,}: " + (
                f"d={r['code_distance']}, reliability {r['reliability']:.3f} ({r['confidence_label']})"
                if r["realizable"] else f"not realizable: {r['reason']}"))
    path = os.path.join(HERE, "ibm_nighthawk_2026_projection.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
