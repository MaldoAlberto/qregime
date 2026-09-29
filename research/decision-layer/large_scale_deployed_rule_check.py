"""
large_scale_deployed_rule_check.py -- toma los circuitos grandes,
nunca-vistos, generados por large_scale_generalization_test.py, los
escribe a QASM en la MISMA base real usada en el resto del proyecto
(cz, rz, sx, x), y corre el pipeline REAL desplegado
(technique_advisor.profile_circuit + recommend_techniques) sobre cada
uno -- exactamente la funcion que un usuario real llamaria -- para medir
que tan bien generaliza la regla ya desplegada (no una reimplementacion)
a circuitos mas grandes que cualquiera del dataset de 98 con el que se
calibro.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile, qasm2
from qiskit.circuit.library import QFT, CDKMRippleCarryAdder, TwoLocal, PhaseGate, PhaseEstimation
from qiskit.circuit.random import random_circuit

from qasm_safe_export import to_plain_decimal_qasm
from technique_advisor import profile_circuit, recommend_techniques

REAL_BASIS = ["cz", "rz", "sx", "x"]


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


CIRCUITS = (
    # QFT_n20/n24 EXCLUIDOS de esta prueba especifica (no del dataset de
    # precision de generalizacion, donde SI se incluyen -- ver
    # large_scale_generalization_test.py, que no llama al pipeline
    # completo). Motivo, encontrado al medir esto: el parser QASM de pyzx
    # (basado en `lark`) escala muy mal con el numero de literales de fase
    # DISTINTOS -- no con el numero de qubits -- y QFT tiene O(n^2)
    # rotaciones de fase, cada una con un angulo distinto. Medido
    # directamente: QFT_n16 -> 2.3s, QFT_n20 -> 25.2s, QFT_n24 -> no
    # termino en varios minutos. Esto es un limite de RENDIMIENTO DE
    # TERCEROS (el parser de pyzx en si, no nuestro codigo) que ninguno de
    # los 3 bugs corregidos esta sesion toca -- se declara aqui en vez de
    # forzarlo u ocultarlo.
    [(f"QFT_n{n}", QFT(n).decompose(reps=3)) for n in [16]] +
    [(f"Adder_n{n}", CDKMRippleCarryAdder(n).decompose(reps=3)) for n in [10, 12]] +
    [(f"QAOA_n{n}_d{d}", build_qaoa(n, d)) for n in [20, 24] for d in [0.3, 0.6]] +
    [(f"Random_n{n}_depth16", random_circuit(n, depth=16, max_operands=2, seed=n * 7, measure=False).decompose(reps=2))
     for n in [20, 24]] +
    [(f"GHZ_n{n}", build_ghz(n)) for n in [24, 28, 32]] +
    [(f"VQEansatz_n{n}_r2", build_vqe_ansatz(n, 2)) for n in [12, 14]] +
    [(f"ToffoliChain_n{n}", build_toffoli_chain(n)) for n in [16, 20, 24]] +
    [(f"QPE_ncount{n}", build_qpe(n)) for n in [9, 10]]
)

HELPS_THRESHOLD_PCT = 5.0  # consistente con "reduccion sustancial", no ruido numerico


def run():
    with open("circuit_dataset_large_scale_heldout.json") as f:
        ground_truth = {r["name"]: r["zx_tcount_reduction_pct"] for r in json.load(f)}

    rows = []
    correct = 0
    for name, qc in CIRCUITS:
        if name not in ground_truth:
            continue
        qasm_path = f"/tmp/_largecheck_{name}.qasm"
        normalized = transpile(qc, basis_gates=REAL_BASIS, optimization_level=1, seed_transpiler=0)
        with open(qasm_path, "w") as f:
            # BUG REAL CORREGIDO (esta sesion): notacion cientifica en
            # angulos pequenos tumbaba la ingestion QASM de QFT grande
            # (ver qasm_safe_export.py)
            f.write(to_plain_decimal_qasm(qasm2.dumps(normalized)))

        try:
            # ACTUALIZADO (esta sesion, tras corregir el bug real de
            # asignacion exponencial en qcp_bounded_tracker.py): antes
            # esto llamaba solo a las funciones baratas de las que
            # depende la regla zx_calculus, evitando profile_circuit()
            # completo porque su sub-paso de rastreo QCP colgaba >60s en
            # circuitos de apenas 22 qubits. Ese bug ya esta corregido
            # (UnionGroup ya no reserva un vector de estado de tamano
            # 2**qubits para grupos descartados) -- ahora se llama al
            # pipeline REAL, completo, tal como lo usaria alguien de
            # verdad.
            p_full = profile_circuit(qasm_path)
            _, recs = recommend_techniques(qasm_path, profile=p_full)
            predicted = recs["zx_calculus"]["recomendada"]
            p = p_full
        except Exception as e:
            print(f"{name}: ERROR en pipeline real -- {e}")
            continue

        actual_gain = ground_truth[name]
        actual_helps = actual_gain > HELPS_THRESHOLD_PCT
        # "ADVERTENCIA" se trata como una prediccion de baja confianza de
        # que SI ayuda (asi es como recommend_techniques la documenta) --
        # se evalua igual que True, pero se marca aparte para ver si el
        # aviso de baja confianza efectivamente coincidio con los casos
        # donde el pipeline se equivoca.
        predicted_bool = bool(predicted) if predicted is not True else True
        if predicted == "ADVERTENCIA":
            predicted_bool = True
        is_correct = (predicted_bool == actual_helps)
        correct += is_correct
        rows.append({
            "name": name, "num_qubits": p["num_qubits"],
            "tcount_frac": p["tcount"] / p["num_gates"] if p["num_gates"] else 0,
            "predicted": predicted, "actual_gain_pct": actual_gain,
            "actual_helps": actual_helps, "correct": is_correct,
        })
        print(f"{name:20s} n_q={p['num_qubits']:3d}  predicho={str(predicted):12s}  "
              f"ganancia_real={actual_gain:6.1f}%  {'OK' if is_correct else 'FALLO'}")

    acc = correct / len(rows) if rows else float("nan")
    majority_baseline = max(sum(r['actual_helps'] for r in rows), len(rows) - sum(r['actual_helps'] for r in rows)) / len(rows) if rows else float("nan")
    print(f"\nPrecision de la regla YA DESPLEGADA sobre {len(rows)} circuitos grandes, "
          f"nunca vistos, de escala mayor a cualquier instancia del dataset de calibracion: "
          f"{acc*100:.1f}%")
    print(f"Baseline trivial (clase mayoritaria en este conjunto): {majority_baseline*100:.1f}%")

    with open("large_scale_deployed_rule_results.json", "w") as f:
        json.dump({"accuracy": acc, "majority_baseline": majority_baseline,
                    "n": len(rows), "rows": rows}, f, indent=2)
    return rows, acc, majority_baseline


if __name__ == "__main__":
    run()
