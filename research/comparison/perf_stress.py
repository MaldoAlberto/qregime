"""Performance and stress test: qregime vs. Azure QRE, QREChem's formula and Mitiq.

Each tool runs on generated circuits of growing size until it fails, times
out or runs out of memory. Every run is a separate process, so a crash or a
memory blow-up in one tool does not affect the others, and peak memory is
measured per run.

Circuit families (n qubits, n layers, so the gate count grows as n^2):
  clifford_t  each layer: H on every qubit, T on every qubit, CX brickwork.
              Non-Clifford content: n*n exact T gates.
  rotations   each layer: RZ(theta) on every qubit (theta not a multiple of
              pi/4), CX brickwork. Non-Clifford content: n*n arbitrary
              rotations, which a fault-tolerant machine must synthesize.

Operations measured:
  qregime.profile        circuit parsing + profiling (pyzx T-count, depth, ...)
  qregime.decide_regime  full decision (noisy simulation up to 20 qubits)
  qregime.estimate_ftqc  analytic FTQC cost from (n, T-count)
  azure.qasm             Azure QRE parsing the QASM file and estimating
  azure.counts           Azure QRE from logical counts (model only)
  qrechem.formula        QREChem's p_L formula in qregime's estimator
  mitiq.fold             Mitiq unitary folding at scale 3 (Mitiq overhead only)
  mitiq.zne              full ZNE, Richardson [1, 2, 3], exact density matrix
                         (n <= 12: memory grows as 4^n)

Limits: TIMEOUT_S per run, MEMORY_LIMIT_MB per process (killed above it).
A tool stops climbing the size ladder after its first failure.

Run (about 30-40 minutes on an 8 GB laptop):

    pip install -e ".[compare]"
    python research/comparison/perf_stress.py

Writes research/comparison/perf_stress_results.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT_S = 180
MEMORY_LIMIT_MB = 5000
SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)
ZNE_SIZES = (2, 4, 6, 8, 10, 12)
FAMILIES = ("clifford_t", "rotations")
OPS = ("qregime.profile", "qregime.decide_regime", "qregime.estimate_ftqc",
       "azure.qasm", "azure.counts", "qrechem.formula", "mitiq.fold", "mitiq.zne")
P_PHYS = 1e-3


# ------------------------------------------------------------ circuits
def make_qasm(family: str, n: int) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    for layer in range(n):
        for q in range(n):
            if family == "clifford_t":
                lines += [f"h q[{q}];", f"t q[{q}];"]
            else:
                lines.append(f"rz({0.1 + 0.37 * ((layer * n + q) % 17)}) q[{q}];")
        for q in range(layer % 2, n - 1, 2):
            lines.append(f"cx q[{q}],q[{q + 1}];")
    return "\n".join(lines) + "\n"


def non_clifford_count(n: int) -> int:
    return n * n


# ------------------------------------------------------------ worker
def worker(op: str, family: str, n: int, path: str) -> dict:
    import resource
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, HERE)

    t_import = time.perf_counter()
    if op.startswith(("qregime", "qrechem")):
        import qregime
        from qregime import resources
        if op == "qrechem.formula":
            import compare_open_tools  # noqa: F401  (registers the QREChem fit)
    elif op.startswith("azure"):
        from compare_open_tools import qasm3
        from qdk import openqasm
        import qdk.estimator as est
    else:
        from qiskit import QuantumCircuit
        from mitiq.zne.scaling import fold_gates_at_random
    import_s = time.perf_counter() - t_import
    rss_after_import = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t = non_clifford_count(n)
    params = {"qubitParams": {"name": "qubit_gate_ns_e3"}, "qecScheme": {"name": "surface_code"},
              "errorBudget": 0.01}
    t0 = time.perf_counter()
    if op == "qregime.profile":
        r = qregime.profile(path)
        out = {"tcount": r["tcount"], "two_qubit_count": r["two_qubit_count"]}
    elif op == "qregime.decide_regime":
        r = qregime.decide_regime(path, hardware="trapped_ion_quantinuum_h2",
                                  physical_qubit_budget=100_000, reliability_target=0.5)
        out = {"regime": r["regime"]}
    elif op == "qregime.estimate_ftqc":
        r = qregime.estimate_ftqc(n, t, P_PHYS, noise_model="sd6")
        out = {"physical_qubits": r.get("physicalQubits")}
    elif op == "qrechem.formula":
        r = resources.estimate_open_ftqc_with_factories(n, t, P_PHYS, 0.01, noise_model="qrechem")
        out = {"physical_qubits": r.get("physicalQubits")}
    elif op == "azure.qasm":
        r = openqasm.estimate(qasm3(path), params)
        out = {"physical_qubits": r["physicalCounts"]["physicalQubits"]}
    elif op == "azure.counts":
        lc = {"numQubits": n, "tCount": t if family == "clifford_t" else 0,
              "rotationCount": t if family == "rotations" else 0,
              "rotationDepth": n if family == "rotations" else 0,
              "cczCount": 0, "measurementCount": n}
        r = est.LogicalCounts(lc).estimate(params)
        out = {"physical_qubits": r["physicalCounts"]["physicalQubits"]}
    elif op == "mitiq.fold":
        qc = QuantumCircuit.from_qasm_file(path)
        folded = fold_gates_at_random(qc, 3.0, seed=0)
        out = {"folded_gates": len(folded.data)}
    elif op == "mitiq.zne":
        from mitiq import zne
        from mitiq.zne.inference import RichardsonFactory
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        qc = QuantumCircuit.from_qasm_file(path)
        qc = transpile(qc.compose(qc.inverse()), basis_gates=["u", "cx"], optimization_level=0)
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(P_PHYS, 2), ["cx"])
        sim = AerSimulator(method="density_matrix", noise_model=noise)

        def execute(circ):
            c = circ.copy()
            c.save_probabilities()
            return float(sim.run(c).result().data()["probabilities"][0])

        val = zne.execute_with_zne(qc, execute, factory=RichardsonFactory([1, 2, 3]),
                                   scale_noise=fold_gates_at_random)
        out = {"zne_value": val}
    else:
        raise ValueError(op)
    op_s = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 / 2**20 if sys.platform == "darwin" else 1 / 2**10  # bytes on macOS, KB on Linux
    return {"op_s": op_s, "import_s": import_s, "peak_rss_mb": rss * scale,
            "rss_after_import_mb": rss_after_import * scale, **out}


# ------------------------------------------------------------ driver
def rss_mb(pid: int) -> float:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                             text=True).stdout.strip()
        return int(out) / 1024 if out else 0.0
    except Exception:
        return 0.0


def run_one(op: str, family: str, n: int, path: str) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", op, family, str(n), path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start, peak = time.perf_counter(), 0.0
    status = None
    while proc.poll() is None:
        peak = max(peak, rss_mb(proc.pid))
        if peak > MEMORY_LIMIT_MB:
            proc.kill()
            status = "out_of_memory"
        elif time.perf_counter() - start > TIMEOUT_S:
            proc.kill()
            status = "timeout"
        if status:
            break
        time.sleep(0.2)
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - start
    row = {"op": op, "family": family, "n": n, "gates": None, "wall_s": wall,
           "polled_peak_rss_mb": peak}
    if status:
        row["status"] = status
    elif proc.returncode != 0:
        row["status"] = "error"
        row["error"] = (stderr.strip().splitlines() or ["?"])[-1][:200]
    else:
        row["status"] = "ok"
        row.update(json.loads(stdout.strip().splitlines()[-1]))
    return row


def main():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for family in FAMILIES:
            for op in OPS:
                sizes = ZNE_SIZES if op == "mitiq.zne" else SIZES
                for n in sizes:
                    path = os.path.join(tmp, f"{family}_{n}.qasm")
                    if not os.path.exists(path):
                        with open(path, "w") as f:
                            f.write(make_qasm(family, n))
                    row = run_one(op, family, n, path)
                    row["gates"] = sum(1 for line in open(path) if line[:2] in ("h ", "t ", "rz", "cx"))
                    rows.append(row)
                    msg = (f"{row['op_s']:9.3f} s  {row['peak_rss_mb']:7.0f} MB"
                           if row["status"] == "ok" else
                           f"{row['status'].upper()} after {row['wall_s']:.0f} s "
                           f"(peak {row['polled_peak_rss_mb']:.0f} MB) {row.get('error', '')}")
                    print(f"  {family:10s} {op:22s} n={n:5d} gates={row['gates']:>9,}  {msg}",
                          flush=True)
                    if row["status"] != "ok":
                        break
    path = os.path.join(HERE, "perf_stress_results.json")
    with open(path, "w") as f:
        json.dump({"timeout_s": TIMEOUT_S, "memory_limit_mb": MEMORY_LIMIT_MB,
                   "p_phys": P_PHYS, "rows": rows}, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(json.dumps(worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])))
    else:
        main()
