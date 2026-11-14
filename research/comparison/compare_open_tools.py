"""Compare qregime with the open-source tools that cover part of its job.

Three comparisons, each run on the example circuits in examples/circuits/:

  A. FTQC cost vs. Microsoft's Azure Quantum Resource Estimator (qdk.estimator,
     open source). Azure parses the same QASM files on its own, so its T-count
     is an independent check of qregime's. Then the cost gap is split into its
     causes by re-running qregime with Azure's assumptions one at a time:
     threshold model, layout (logical qubits), and number of T factories.
  B. The surface-code formula of QREChem (Argonne, arXiv:2404.16351),
     p_L = 0.1 (100 p)^((d+1)/2), plugged into qregime's estimator.
  C. The QEM side vs. Mitiq (Unitary Foundation): zero-noise extrapolation on
     the same noisy simulation qregime uses for its QEM column.

Run (about 3-5 minutes):

    pip install -e ".[compare]"
    python research/comparison/compare_open_tools.py

Writes research/comparison/compare_open_tools_results.json.
"""
from __future__ import annotations

import json
import math
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import qregime
from qregime import resources

HERE = os.path.dirname(os.path.abspath(__file__))
CIRCUITS = os.path.join(HERE, "..", "..", "examples", "circuits")
NAMES = ("adder8", "qft10", "grover", "vqe_ansatz", "qaoa_farm",
         "kicked_ising_analog_16q", "shor_like_sequential")
ERROR_BUDGET = 0.01
AZURE_PROFILES = {1e-3: "qubit_gate_ns_e3", 1e-4: "qubit_gate_ns_e4"}

# Azure's default surface-code model (read back from jobParams.qecScheme):
# p_L = 0.03 (p / 0.01)^((d+1)/2) per logical qubit per logical cycle.
AZURE_FIT = {"A": 0.03, "p_th": 0.01, "p_th_stderr": 0.0, "noise_model": "azure"}
# QREChem (arXiv:2404.16351): p_L = 0.1 (100 p)^((d+1)/2), i.e. A = 0.1, p_th = 1%.
QRECHEM_FIT = {"A": 0.1, "p_th": 0.01, "p_th_stderr": 0.0, "noise_model": "qrechem"}

_original_load_fit = resources._load_fit


def _patched_load_fit(noise_model="sd6"):
    return {"azure": AZURE_FIT, "qrechem": QRECHEM_FIT}.get(noise_model) \
        or _original_load_fit(noise_model)


resources._load_fit = _patched_load_fit


def qasm3(path: str) -> str:
    """OpenQASM 2 -> 3 (header and register syntax only) for qdk.openqasm."""
    import re
    src = open(path).read()
    src = src.replace("OPENQASM 2.0;", "OPENQASM 3.0;")
    src = src.replace('include "qelib1.inc";', 'include "stdgates.inc";')
    src = re.sub(r"qreg (\w+)\[(\d+)\];", r"qubit[\2] \1;", src)
    src = re.sub(r"creg (\w+)\[(\d+)\];", r"bit[\2] \1;", src)
    src = re.sub(r"^u\(", "U(", src, flags=re.M)  # qelib1 u(θ,φ,λ) = the QASM 3 built-in U
    return src


def azure_layout_qubits(n: int) -> int:
    """Azure's 2D layout: 2n + ceil(sqrt(8n)) + 1 logical qubits (Beverland et al. 2022)."""
    return 2 * n + math.ceil(math.sqrt(8 * n)) + 1


# ---------------------------------------------------------------- A. Azure QRE
def section_azure():
    from qdk import openqasm
    import qdk.estimator as est

    print("\nA. FTQC cost: qregime vs. Azure Quantum Resource Estimator (error budget 1%)")
    rows = []
    for name in NAMES:
        path = os.path.join(CIRCUITS, f"{name}.qasm")
        prof = qregime.profile(path)
        n, t = prof["num_qubits"], max(prof["tcount"], 0)
        for p, azure_name in AZURE_PROFILES.items():
            params = {"qubitParams": {"name": azure_name}, "qecScheme": {"name": "surface_code"},
                      "errorBudget": ERROR_BUDGET}
            # Azure on the QASM (its own parsing and rotation synthesis)
            try:
                az = openqasm.estimate(qasm3(path), params)
                az_lc = az["logicalCounts"]
            except Exception as exc:  # pragma: no cover
                az, az_lc = None, {"error": str(exc)[:120]}
            # Azure on qregime's logical counts (same input, model-only comparison)
            az2 = est.LogicalCounts({"numQubits": n, "tCount": max(t, 1), "rotationCount": 0,
                                     "rotationDepth": 0, "cczCount": 0,
                                     "measurementCount": n}).estimate(params)
            row = {"circuit": name, "p": p, "n": n, "tcount_qregime": t,
                   "azure_qasm_tcount": az_lc.get("tCount"),
                   "azure_qasm_rotations": az_lc.get("rotationCount"),
                   "azure_qasm_ccz": az_lc.get("cczCount"),
                   "azure_qasm_d": az["logicalQubit"]["codeDistance"] if az else None,
                   "azure_qasm_qubits": az["physicalCounts"]["physicalQubits"] if az else None}
            pc = az2["physicalCounts"]
            row.update({
                "azure_d": az2["logicalQubit"]["codeDistance"],
                "azure_qubits": pc["physicalQubits"],
                "azure_logical_qubits": pc["breakdown"]["algorithmicLogicalQubits"],
                "azure_factories": pc["breakdown"]["numTfactories"],
                "azure_factory_qubits": pc["breakdown"]["physicalQubitsForTfactories"],
                "azure_runtime_s": pc["runtime"] * 1e-9,
            })
            # qregime, its own fits
            for nm in ("sd6", "toy"):
                e = qregime.estimate_ftqc(n, t, p, noise_model=nm)
                row[f"qregime_{nm}_d"] = e.get("codeDistance")
                row[f"qregime_{nm}_qubits"] = e.get("physicalQubits")
            # Reconciliation: run qregime's estimator with Azure's assumptions,
            # adding one at a time, to see which one explains the gap.
            nf = row["azure_factories"]
            nl = row["azure_logical_qubits"]
            steps = {  # (logical qubits, share of the error budget for logical errors)
                "1_azure_threshold": (n, 1.0),
                "2_plus_azure_layout": (nl, 1.0),
                "3_plus_azure_budget_split": (nl, 1 / 3),
            }
            for key, (nq, share) in steps.items():
                e = resources.estimate_open_ftqc_with_factories(
                    nq, max(t, 1), p, ERROR_BUDGET * share, noise_model="azure")
                row[f"recon_{key}_d"] = e.get("codeDistance")
                row[f"recon_{key}_qubits"] = e.get("physicalQubits")
                data_qubits = e.get("physicalQubitsForData")
            # 4: Azure's own factories (its count and size) instead of one Litinski 15-to-1
            row["recon_4_plus_azure_factories_d"] = row["recon_3_plus_azure_budget_split_d"]
            row["recon_4_plus_azure_factories_qubits"] = data_qubits + row["azure_factory_qubits"]
            row["recon_ratio_to_azure"] = row["recon_4_plus_azure_factories_qubits"] / row["azure_qubits"]
            rows.append(row)
            print(f"    {name:24s} p={p:.0e} T={t:5d} [Azure QASM: T={row['azure_qasm_tcount']} "
                  f"rot={row['azure_qasm_rotations']} {row['azure_qasm_qubits'] or 0:,}]  "
                  f"Azure d={row['azure_d']:2d} {row['azure_qubits']:>8,} ({nf} fact.)  "
                  f"qregime sd6 d={row['qregime_sd6_d']} {row['qregime_sd6_qubits'] or 0:>7,}  "
                  f"toy d={row['qregime_toy_d']} {row['qregime_toy_qubits'] or 0:>7,}  recon: "
                  + " -> ".join(f"{row[f'recon_{k}_qubits']:,}" for k in
                                ("1_azure_threshold", "2_plus_azure_layout",
                                 "3_plus_azure_budget_split", "4_plus_azure_factories"))
                  + f"  ({row['recon_ratio_to_azure']:.2f}x Azure)")
    return rows


# ------------------------------------------------------------- B. QREChem
def section_qrechem():
    print("\nB. QREChem's formula p_L = 0.1 (100 p)^((d+1)/2) vs. qregime's fits")
    n, t = 16, 3930
    rows = []
    for p in (1e-3, 5e-4, 1e-4):
        row = {"p": p}
        for nm in ("qrechem", "azure", "sd6", "toy", "si1000"):
            e = resources.estimate_open_ftqc_with_factories(n, t, p, ERROR_BUDGET, noise_model=nm)
            row[nm] = (e.get("codeDistance"), e.get("physicalQubits")) if e["feasible"] else None
        rows.append(row)
        print("    p={:.0e}  ".format(p) + "  ".join(
            f"{k} " + (f"d={v[0]} {v[1]:,}" if v else "infeasible") for k, v in row.items() if k != "p"))
    return rows


# --------------------------------------------------------------- C. Mitiq ZNE
def section_mitiq(shots=2000, repeats=3):
    from mitiq import zne
    from mitiq.zne.inference import LinearFactory, RichardsonFactory
    from mitiq.zne.scaling import fold_gates_at_random
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    print(f"\nC. QEM: qregime's raw success vs. Mitiq ZNE ({shots} shots per circuit, {repeats} repeats)")
    rows = []
    for name in NAMES:
        path = os.path.join(CIRCUITS, f"{name}.qasm")
        qc = QuantumCircuit.from_qasm_file(path)
        if qc.num_qubits > 18:
            continue
        qc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
        probs = Statevector.from_instruction(qc).probabilities()
        support = set(np.nonzero(probs > 1e-9)[0].tolist())
        if len(support) / 2 ** qc.num_qubits > 0.5:
            continue  # success metric uninformative, qregime uses its proxy here too
        for key in ("superconductor_ibm", "trapped_ion_quantinuum_h2"):
            hw = qregime.HARDWARE_PROFILES[key]
            noise = NoiseModel()
            noise.add_all_qubit_quantum_error(depolarizing_error(hw.p_2q, 2), ["cx"])
            sim = AerSimulator(method="statevector", noise_model=noise)
            seed = [0]

            def execute(circ):
                c = circ.copy()
                c.measure_all()
                seed[0] += 1
                counts = sim.run(c, shots=shots, seed_simulator=seed[0]).result().get_counts()
                ok = sum(v for b, v in counts.items() if int(b.replace(" ", ""), 2) in support)
                return ok / shots

            raw, lin, rich = [], [], []
            for _ in range(repeats):
                raw.append(execute(qc))
                lin.append(zne.execute_with_zne(qc, execute, factory=LinearFactory([1, 2, 3]),
                                                scale_noise=fold_gates_at_random))
                rich.append(zne.execute_with_zne(qc, execute, factory=RichardsonFactory([1, 2, 3]),
                                                 scale_noise=fold_gates_at_random))
            row = {"circuit": name, "hardware": key, "p_2q": hw.p_2q,
                   "two_qubit_gates": qc.count_ops().get("cx", 0),
                   "proxy": (1 - hw.p_2q) ** qc.count_ops().get("cx", 0),
                   "raw_mean": float(np.mean(raw)), "raw_std": float(np.std(raw)),
                   "zne_linear_mean": float(np.mean(lin)), "zne_linear_std": float(np.std(lin)),
                   "zne_richardson_mean": float(np.mean(rich)),
                   "zne_richardson_std": float(np.std(rich)),
                   "ideal": 1.0, "shots_per_estimate_raw": shots,
                   "shots_per_estimate_zne": 3 * shots}
            rows.append(row)
            print(f"    {name:24s} {key:27s} cx={row['two_qubit_gates']:4d}  proxy {row['proxy']:.3f}  "
                  f"raw {row['raw_mean']:.3f}+/-{row['raw_std']:.3f}  "
                  f"ZNE lin {row['zne_linear_mean']:.3f}+/-{row['zne_linear_std']:.3f}  "
                  f"rich {row['zne_richardson_mean']:.3f}+/-{row['zne_richardson_std']:.3f}  (ideal 1)")
    return rows


def main():
    out = {"azure": section_azure(), "qrechem": section_qrechem(), "mitiq": section_mitiq()}
    path = os.path.join(HERE, "compare_open_tools_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
