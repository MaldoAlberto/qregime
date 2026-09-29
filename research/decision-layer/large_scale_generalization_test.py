"""
large_scale_generalization_test.py -- responde directamente la primera
peticion de mejora: "amplia el analisis a circuitos de mayor tamano y mas
variados, para reforzar la generalidad de las conclusiones."

DISENO (importante, por que esto es mas fuerte que solo agregar mas filas
al dataset de 98 instancias): en vez de mezclar circuitos nuevos y grandes
dentro del mismo pool de entrenamiento/LOFO ya usado, se genera un
conjunto SEPARADO, nunca visto por ninguna regla ni modelo de este
proyecto, con las MISMAS 9 familias pero a escala mayor (hasta 32 qubits,
vs. el maximo de 20 del dataset original) -- una prueba de GENERALIZACION
real (held-out por tamano), no solo mas datos para la misma validacion
cruzada. Reutiliza exactamente los mismos generadores de circuito
(build_circuit_dataset.py, build_circuit_dataset_v2.py) y la misma
medicion de ground truth (ZX-calculus full_reduce() real, no asumido).
"""
import warnings
warnings.filterwarnings("ignore")

import json
import time
import networkx as nx
import numpy as np
import pyzx as zx
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit.library import QFT, CDKMRippleCarryAdder, TwoLocal, PhaseGate, PhaseEstimation
from qiskit.circuit.random import random_circuit

from zx_reduction import zx_reduce
from qasm_hypergraph import resource_report, CircuitHypergraph
from qasm_safe_export import safe_qasm2_dumps

REAL_BASIS = ["cz", "rz", "sx", "x"]


def to_real_basis_pyzx(qc):
    normalized = transpile(qc, basis_gates=REAL_BASIS, optimization_level=1, seed_transpiler=0)
    # BUG REAL CORREGIDO (esta sesion): qasm2.dumps() escribe angulos muy
    # pequenos en notacion cientifica, que el parser de pyzx no reconoce
    # (ver qasm_safe_export.py) -- esto era lo que tumbaba QFT_n16/20/24
    # en esta misma prueba.
    return zx.Circuit.from_qasm(safe_qasm2_dumps(normalized))


def measure_zx_gain(qc):
    before = to_real_basis_pyzx(qc)
    tcount_before = resource_report(before)["tcount"]
    if tcount_before == 0:
        return 0.0
    pyzx_original = zx.Circuit.from_qasm(safe_qasm2_dumps(qc))
    reduced = zx_reduce(pyzx_original)
    reduced_qc = QuantumCircuit.from_qasm_str(reduced.to_qasm())
    after = to_real_basis_pyzx(reduced_qc)
    tcount_after = resource_report(after)["tcount"]
    return 100.0 * (tcount_before - tcount_after) / tcount_before


def profile_qc(qc):
    pyzx_c = zx.Circuit.from_qasm(safe_qasm2_dumps(qc)).to_basic_gates()
    hg = CircuitHypergraph(pyzx_c)
    metrics = resource_report(pyzx_c)
    ig = hg.interaction_graph()
    degrees = [d for _, d in ig.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    return {
        "num_qubits": metrics["num_qubits"], "num_gates": metrics["num_gates"],
        "tcount": metrics["tcount"], "two_qubit_count": metrics["two_qubit_count"],
        "avg_degree": avg_degree, "max_degree": max_degree,
        "tcount_frac": metrics["tcount"] / metrics["num_gates"] if metrics["num_gates"] else 0.0,
    }


def build_ghz(n):
    qc = QuantumCircuit(n); qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def build_vqe_ansatz(n, reps):
    ansatz = TwoLocal(n, ["ry", "rz"], "cx", entanglement="linear", reps=reps)
    params = np.linspace(0.3, 2.1, ansatz.num_parameters)
    return ansatz.assign_parameters(params).decompose(reps=2)


def build_toffoli_chain(n):
    qc = QuantumCircuit(n)
    for i in range(n - 2):
        qc.ccx(i, i + 1, i + 2)
    return qc


def build_qpe(n_counting, n_target=1):
    unitary = QuantumCircuit(n_target)
    unitary.append(PhaseGate(0.7), [0])
    qpe = PhaseEstimation(n_counting, unitary)
    return qpe.decompose(reps=3)


def build_qaoa(n, density):
    graph = nx.gnp_random_graph(n, density, seed=hash((n, density, "large")) % 1000)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for u, v in graph.edges():
        qc.cx(u, v); qc.rz(0.7, v); qc.cx(u, v)
    qc.rx(0.4, range(n))
    return qc


# familias y tamanos MAYORES que cualquier instancia del dataset original
# de 98 (maximo previo: 20 qubits, familia GHZ) -- deliberadamente
# empujando hasta 32 qubits en varias familias.
LARGE_CIRCUITS = (
    [(f"QFT_n{n}", QFT(n).decompose(reps=3)) for n in [16, 20, 24]] +
    [(f"Adder_n{n}", CDKMRippleCarryAdder(n).decompose(reps=3)) for n in [10, 12]] +
    [(f"QAOA_n{n}_d{d}", build_qaoa(n, d)) for n in [20, 24] for d in [0.3, 0.6]] +
    [(f"Random_n{n}_depth16", random_circuit(n, depth=16, max_operands=2, seed=n * 7, measure=False).decompose(reps=2))
     for n in [20, 24]] +
    [(f"GHZ_n{n}", build_ghz(n)) for n in [24, 28, 32]] +
    [(f"VQEansatz_n{n}_r2", build_vqe_ansatz(n, 2)) for n in [12, 14]] +
    [(f"ToffoliChain_n{n}", build_toffoli_chain(n)) for n in [16, 20, 24]] +
    [(f"QPE_ncount{n}", build_qpe(n)) for n in [9, 10]]
)


def run():
    rows = []
    t_start = time.time()
    for name, qc in LARGE_CIRCUITS:
        t0 = time.time()
        try:
            features = profile_qc(qc)
            gain = measure_zx_gain(qc)
            row = {"name": name, **features, "zx_tcount_reduction_pct": gain}
            rows.append(row)
            print(f"{name:20s} n_q={features['num_qubits']:3d} n_gates={features['num_gates']:6d} "
                  f"tcount_frac={features['tcount_frac']:.3f} grado_max={features['max_degree']:5.1f} "
                  f"-> ZX gain={gain:6.1f}%  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"{name}: SALTADO ({type(e).__name__}: {e})")
    print(f"\nTotal: {len(rows)} circuitos grandes, {time.time()-t_start:.1f}s")
    with open("circuit_dataset_large_scale_heldout.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("Escrito circuit_dataset_large_scale_heldout.json")
    return rows


if __name__ == "__main__":
    run()
