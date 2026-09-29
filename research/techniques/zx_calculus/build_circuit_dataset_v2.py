"""
Version 2 del dataset (Parte 9 del reporte): en vez de solo mas VARIANTES
de las 4 familias originales (QFT, Sumador, QAOA, aleatorio), se agregan
FAMILIAS ESTRUCTURALMENTE DISTINTAS -- porque la prueba honesta
(leave-one-family-out) con solo 4 familias mostro que ningun modelo puede
generalizar a una estructura nunca vista con tan poca diversidad. Mas
variantes de la MISMA familia no arreglan eso; mas familias si podrian.

Familias nuevas anadidas (todas con mecanismo real, no inventado):
  - GHZ: preparacion de estado GHZ (H + cadena de CX) -- casi todo
    Clifford, T-count muy bajo por diseno; sirve de "negativo estructural"
    limpio, distinto del resto.
  - VQE_ansatz: TwoLocal (RY+RZ + CX de entrelazado lineal), tipico ansatz
    variacional de VQE -- estructura de capas, distinta de QAOA (que tiene
    RZZ sobre un grafo de un problema, no capas parametrizadas genericas).
  - Grover: operador de Grover real (GroverOperator de Qiskit) sobre un
    oracle diagonal simple -- trae Toffolis/T-gates por construccion,
    estructura muy distinta de QFT o el sumador.
  - Toffoli_chain: cadena de compuertas CCX (Toffoli) sobre qubits
    consecutivos -- T-count muy alto y estructura de interaccion en
    cadena corta (a diferencia del sumador, que tambien es cadena pero
    con CX/CCX intercalados de forma especifica del algoritmo aritmetico).
  - QPE: estimacion de fase cuantica (PhaseEstimation de Qiskit) con una
    unitaria de fase simple -- estructura de control-fase repetida,
    distinta de la QFT sola (tiene un registro de conteo + fases
    controladas de la unitaria, no solo QFT).
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pyzx as zx
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit.library import GroverOperator, TwoLocal, PhaseEstimation

from zx_reduction import zx_reduce
from qasm_hypergraph import resource_report, CircuitHypergraph

REAL_BASIS = ["cz", "rz", "sx", "x"]


def to_real_basis_pyzx(qc: QuantumCircuit):
    normalized = transpile(qc, basis_gates=REAL_BASIS, optimization_level=1, seed_transpiler=0)
    return zx.Circuit.from_qasm(qasm2.dumps(normalized))


def measure_zx_gain(qc: QuantumCircuit) -> float:
    before = to_real_basis_pyzx(qc)
    tcount_before = resource_report(before)["tcount"]
    if tcount_before == 0:
        return 0.0
    pyzx_original = zx.Circuit.from_qasm(qasm2.dumps(qc))
    reduced = zx_reduce(pyzx_original)
    reduced_qc = QuantumCircuit.from_qasm_str(reduced.to_qasm())
    after = to_real_basis_pyzx(reduced_qc)
    tcount_after = resource_report(after)["tcount"]
    return 100.0 * (tcount_before - tcount_after) / tcount_before


def profile_qc(qc: QuantumCircuit) -> dict:
    pyzx_c = zx.Circuit.from_qasm(qasm2.dumps(qc)).to_basic_gates()
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
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def build_vqe_ansatz(n, reps):
    ansatz = TwoLocal(n, ["ry", "rz"], "cx", entanglement="linear", reps=reps)
    # parametros fijos (no en 0, para que las rotaciones sean genuinamente
    # no triviales -- un ansatz sin parametrizar seria degenerado)
    params = np.linspace(0.3, 2.1, ansatz.num_parameters)
    return ansatz.assign_parameters(params).decompose(reps=2)


def build_grover(n_oracle_qubits):
    oracle = QuantumCircuit(n_oracle_qubits)
    # oracle diagonal simple: marca el estado |11...1> con una fase -1
    # (multi-controlled Z), sintetizado por Qiskit a Toffolis/T reales
    oracle.h(n_oracle_qubits - 1)
    oracle.mcx(list(range(n_oracle_qubits - 1)), n_oracle_qubits - 1)
    oracle.h(n_oracle_qubits - 1)
    grover_op = GroverOperator(oracle)
    return grover_op.decompose(reps=3)


def build_toffoli_chain(n):
    qc = QuantumCircuit(n)
    for i in range(n - 2):
        qc.ccx(i, i + 1, i + 2)
    return qc


def build_qpe(n_counting, n_target=1):
    from qiskit.circuit.library import PhaseGate
    unitary = QuantumCircuit(n_target)
    unitary.append(PhaseGate(0.7), [0])
    qpe = PhaseEstimation(n_counting, unitary)
    return qpe.decompose(reps=3)


def generate_circuits():
    circuits = []

    for n in [4, 6, 8, 10, 12, 16, 20]:
        circuits.append((f"GHZ_n{n}", build_ghz(n)))

    for n in [4, 6, 8, 10]:
        for reps in [1, 2, 3]:
            circuits.append((f"VQEansatz_n{n}_r{reps}", build_vqe_ansatz(n, reps)))

    for n in [3, 4, 5, 6, 7]:
        circuits.append((f"Grover_n{n}", build_grover(n)))

    for n in [4, 5, 6, 7, 8, 10, 12]:
        circuits.append((f"ToffoliChain_n{n}", build_toffoli_chain(n)))

    for n_count in [3, 4, 5, 6, 7]:
        circuits.append((f"QPE_ncount{n_count}", build_qpe(n_count)))

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
            print(f"{name:22s} n_q={features['num_qubits']:3d} n_gates={features['num_gates']:5d} "
                  f"tcount_frac={features['tcount_frac']:.3f} grado_max={features['max_degree']:.1f} "
                  f"-> ZX gain={gain:6.1f}%")
        except Exception as e:
            print(f"{name}: SALTADO ({type(e).__name__}: {e})")

    with open("circuit_dataset_v2_new_families.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} circuitos nuevos guardados en circuit_dataset_v2_new_families.json")
    return rows


if __name__ == "__main__":
    build_dataset()
