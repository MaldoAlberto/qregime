"""
Genera un dataset mas grande de circuitos (para reemplazar las 4 muestras
"a mano" con algo que un metodo clasico de aprendizaje pueda calibrar de
verdad), y mide para cada uno:
  - las features estructurales baratas de technique_advisor.profile_circuit
  - el GROUND TRUTH real de si ZX-calculus (full_reduce de PyZX) redujo el
    T-count de forma sustancial, corriendo la tecnica de verdad (no
    asumido) sobre el mismo basis gates real usado en el resto de la
    sesion.

Familias generadas (variando tamano/densidad, no solo 1 instancia cada
una, para tener variacion real dentro de cada familia):
  - QFT para n en [3,4,...,12]
  - Sumador Cuccaro (CDKMRippleCarryAdder) para n en [2,3,...,8] bits
  - QAOA de 1 capa sobre grafos aleatorios, n en [6,8,...,16], densidad
    en [0.2, 0.4, 0.6, 0.8]
  - Circuitos aleatorios (qiskit random_circuit), n en [4,6,...,16],
    profundidad en [8, 16, 24]
"""

import warnings
warnings.filterwarnings("ignore")

import json
import networkx as nx
import pyzx as zx
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit.library import QFT, CDKMRippleCarryAdder
from qiskit.circuit.random import random_circuit

from zx_reduction import zx_reduce
from qasm_hypergraph import resource_report

REAL_BASIS = ["cz", "rz", "sx", "x"]


def to_real_basis_pyzx(qc: QuantumCircuit):
    normalized = transpile(qc, basis_gates=REAL_BASIS, optimization_level=1, seed_transpiler=0)
    return zx.Circuit.from_qasm(qasm2.dumps(normalized))


def measure_zx_gain(qc: QuantumCircuit) -> float:
    """% de reduccion de T-count real por ZX-calculus, mismo basis real
    que el resto de la sesion (Parte 4 del reporte)."""
    before = to_real_basis_pyzx(qc)
    before_metrics = resource_report(before)
    tcount_before = before_metrics["tcount"]
    if tcount_before == 0:
        return 0.0

    pyzx_original = zx.Circuit.from_qasm(qasm2.dumps(qc))
    reduced = zx_reduce(pyzx_original)
    reduced_qc = QuantumCircuit.from_qasm_str(reduced.to_qasm())
    after = to_real_basis_pyzx(reduced_qc)
    after_metrics = resource_report(after)
    tcount_after = after_metrics["tcount"]

    return 100.0 * (tcount_before - tcount_after) / tcount_before


def profile_qc(qc: QuantumCircuit) -> dict:
    """Igual que technique_advisor.profile_circuit pero partiendo de un
    QuantumCircuit ya en memoria (evita escribir/leer qasm por cada uno)."""
    from qasm_hypergraph import CircuitHypergraph

    pyzx_c = zx.Circuit.from_qasm(qasm2.dumps(qc)).to_basic_gates()
    hg = CircuitHypergraph(pyzx_c)
    metrics = resource_report(pyzx_c)
    ig = hg.interaction_graph()
    degrees = [d for _, d in ig.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0

    last_use = {}
    for he in hg.hyperedges:
        for q in he["qubits"]:
            last_use[q] = he["index"]
    midpoint = len(hg.hyperedges) / 2
    ancilla_candidates = sum(1 for q, idx in last_use.items() if idx < midpoint)

    return {
        "num_qubits": metrics["num_qubits"],
        "num_gates": metrics["num_gates"],
        "tcount": metrics["tcount"],
        "two_qubit_count": metrics["two_qubit_count"],
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "ancilla_candidates": ancilla_candidates,
        "tcount_frac": metrics["tcount"] / metrics["num_gates"] if metrics["num_gates"] else 0.0,
    }


def generate_circuits():
    circuits = []

    for n in range(3, 13):
        qc = QFT(n).decompose(reps=3)
        circuits.append((f"QFT_n{n}", qc))

    for n in range(2, 9):
        qc = CDKMRippleCarryAdder(n).decompose(reps=3)
        circuits.append((f"Adder_n{n}", qc))

    for n in [6, 8, 10, 12, 14, 16]:
        for density in [0.2, 0.4, 0.6, 0.8]:
            graph = nx.gnp_random_graph(n, density, seed=hash((n, density)) % 1000)
            if graph.number_of_edges() == 0:
                continue
            qc = QuantumCircuit(n)
            qc.h(range(n))
            for u, v in graph.edges():
                qc.cx(u, v); qc.rz(0.7, v); qc.cx(u, v)
            qc.rx(0.4, range(n))
            circuits.append((f"QAOA_n{n}_d{density}", qc))

    for n in [4, 6, 8, 10, 12, 14, 16]:
        for depth in [8, 16, 24]:
            qc = random_circuit(n, depth=depth, max_operands=2, seed=(n * 100 + depth), measure=False)
            qc = qc.decompose(reps=2)
            circuits.append((f"Random_n{n}_depth{depth}", qc))

    return circuits


def build_dataset():
    circuits = generate_circuits()
    rows = []
    for name, qc in circuits:
        try:
            features = profile_qc(qc)
            gain = measure_zx_gain(qc)
            row = {"name": name, **features, "zx_tcount_reduction_pct": gain}
            rows.append(row)
            print(f"{name:26s} n_q={features['num_qubits']:3d} n_gates={features['num_gates']:4d} "
                  f"tcount_frac={features['tcount_frac']:.3f} avg_deg={features['avg_degree']:.1f} "
                  f"-> ZX gain={gain:6.1f}%")
        except Exception as e:
            print(f"{name}: SALTADO ({e})")

    with open("circuit_dataset.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} circuitos guardados en circuit_dataset.json")
    return rows


if __name__ == "__main__":
    build_dataset()
