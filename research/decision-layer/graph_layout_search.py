"""
Generalizacion de la estrategia del notebook de IBM QGSS 2025 (lab-2,
Ejercicios 4 y 5: github.com/qiskit-community/qgss-2025/blob/main/solutions/
lab-2/lab2-solution.ipynb) para elegir el MEJOR layout inicial:

  Ejercicio 4 de ese notebook construye un grafo dirigido del backend con el
  error de compuerta de 2 qubits como peso de cada arista, y busca TODOS los
  caminos de qubits fisicos cuya conectividad reproduce exactamente la
  conectividad requerida por el circuito (`logical_pair_list`) y cuyo error
  acumulado esta por debajo de un umbral -- antes de siquiera transpilar.
  Solo funciona alli porque su circuito de prueba (Max-cut de 5-7 nodos) es
  disperso: su grafo de interaccion tiene grado maximo 2-3, que es compatible
  con la conectividad heavy-hex real del hardware (grado maximo 3).

Este modulo generaliza esa idea A CUALQUIER circuito real, y la usa como un
TEST DE FACTIBILIDAD ESTRUCTURAL antes de gastar tiempo en cualquier busqueda
de layout: si el grado maximo del grafo de interaccion LOGICO del circuito
supera el grado maximo del mapa de acoplamiento FISICO del backend, entonces
NINGUN layout -- sin importar que tan bien calibrado este -- puede evitar
SWAPs adicionales, porque no existe ningun subgrafo isomorfo dentro del
hardware. Esto explica, con una prueba estructural (no solo empirica), por
que la busqueda conjunta layout+ruteo de `layout_noise_search.py` no puede
mejorar mas alla de cierto punto en circuitos densos como el Shor-like: el
problema no es la busqueda, es que el circuito no cabe.

Cuando el circuito SI es estructuralmente factible (grado maximo <= grado
maximo del hardware), este modulo reproduce la estrategia real del notebook:
construye el grafo de calibracion ponderado por error de compuerta de 2
qubits y busca, via VF2 (isomorfismo de subgrafos, exacto, no heuristico),
los subgrafos fisicos que reproducen la conectividad requerida, ordenados
por error acumulado -- devolviendo layouts candidatos ya construidos con
datos reales de calibracion, no con semillas aleatorias del transpilador.
"""
import itertools
import time

import rustworkx as rx
from qiskit import QuantumCircuit


def circuit_interaction_graph(qasm_path: str) -> rx.PyGraph:
    """Grafo NO dirigido de interaccion logica: un nodo por qubit del
    circuito, una arista por cada par de qubits que comparte al menos una
    compuerta de 2 qubits (peso = numero de veces que interactuan)."""
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    graph = rx.PyGraph()
    graph.add_nodes_from(range(qc.num_qubits))
    weight = {}
    for instruction in qc.data:
        if len(instruction.qubits) == 2:
            a, b = (qc.find_bit(q).index for q in instruction.qubits)
            key = (min(a, b), max(a, b))
            weight[key] = weight.get(key, 0) + 1
    for (a, b), w in weight.items():
        graph.add_edge(a, b, w)
    return graph


def calibration_graph(backend, two_qubit_gate: str = None) -> rx.PyGraph:
    """Grafo NO dirigido del backend real: un nodo por qubit fisico, una
    arista por cada enlace de acoplamiento, con el error de compuerta de 2
    qubits medido/calibrado como peso."""
    if two_qubit_gate is None:
        basis = backend.configuration().basis_gates
        two_qubit_gate = "ecr" if "ecr" in basis else ("cz" if "cz" in basis else "cx")

    props = backend.properties()
    graph = rx.PyGraph()
    graph.add_nodes_from(range(backend.num_qubits))
    seen = set()
    for pair in backend.coupling_map:
        key = (min(pair), max(pair))
        if key in seen:
            continue
        seen.add(key)
        try:
            err = props.gate_error(gate=two_qubit_gate, qubits=pair)
        except Exception:
            continue
        graph.add_edge(key[0], key[1], err)
    return graph


def structural_feasibility(qasm_path: str, backend, verbose: bool = True) -> dict:
    """Test de factibilidad ESTRUCTURAL (no de calibracion): compara el
    grado maximo requerido por el circuito contra el grado maximo que el
    mapa de acoplamiento fisico puede ofrecer. Si el circuito pide mas
    conexiones directas de las que CUALQUIER qubit fisico puede tener,
    ningun layout puede evitar SWAPs -- es una limitación topológica del
    circuito frente al hardware, no una limitación de la busqueda de layout."""
    interaction = circuit_interaction_graph(qasm_path)
    cal = calibration_graph(backend)

    degrees_circuit = [interaction.degree(n) for n in interaction.node_indices()]
    degrees_hw = [cal.degree(n) for n in cal.node_indices()]
    max_deg_circuit = max(degrees_circuit) if degrees_circuit else 0
    max_deg_hw = max(degrees_hw) if degrees_hw else 0

    feasible = max_deg_circuit <= max_deg_hw
    result = {
        "qasm_path": qasm_path,
        "num_qubits_circuit": interaction.num_nodes(),
        "num_interaction_edges": interaction.num_edges(),
        "max_degree_circuit": max_deg_circuit,
        "max_degree_hardware": max_deg_hw,
        "structurally_feasible_without_swaps": feasible,
    }
    if verbose:
        veredicto = "FACTIBLE (podria existir un layout sin SWAPs)" if feasible else \
            "ESTRUCTURALMENTE INFACTIBLE (ningun layout evita SWAPs)"
        print(f"  {qasm_path}: grado_max_circuito={max_deg_circuit}  "
              f"grado_max_hardware={max_deg_hw}  -> {veredicto}")
    return result


def find_layout_candidates_by_isomorphism(qasm_path: str, backend, max_candidates: int = 5,
                                            timeout_s: float = 20.0, verbose: bool = True):
    """Reproduce el Ejercicio 4/5 del notebook de QGSS: busca, via VF2
    (isomorfismo exacto de subgrafos), asignaciones de qubits fisicos cuya
    conectividad reproduce EXACTAMENTE la conectividad requerida por el
    circuito, y las ordena por error de calibracion acumulado -- SOLO
    tiene sentido llamarlo si `structural_feasibility(...)` ya dio True,
    porque si no, la busqueda VF2 devolvera cero resultados por construccion
    (no hay ningun subgrafo isomorfo que buscar)."""
    interaction = circuit_interaction_graph(qasm_path)
    cal = calibration_graph(backend)

    feas = structural_feasibility(qasm_path, backend, verbose=False)
    if not feas["structurally_feasible_without_swaps"]:
        if verbose:
            print(f"  Factibilidad estructural: NO -- se omite la busqueda VF2 "
                  f"(no puede haber ningun subgrafo isomorfo: grado_max_circuito="
                  f"{feas['max_degree_circuit']} > grado_max_hardware="
                  f"{feas['max_degree_hardware']}).")
        return {"feasibility": feas, "candidates": []}

    t0 = time.time()
    matcher = rx.vf2_mapping(cal, interaction, subgraph=True, id_order=False)
    candidates = []
    for mapping in matcher:
        if time.time() - t0 > timeout_s or len(candidates) >= max_candidates * 20:
            break
        # mapping: {physical_node: logical_node}
        total_error = 0.0
        for phys_a, log_a in mapping.items():
            for phys_b, log_b in mapping.items():
                if phys_a < phys_b and cal.has_edge(phys_a, phys_b) and interaction.has_edge(log_a, log_b):
                    total_error += cal.get_edge_data(phys_a, phys_b)
        candidates.append({"physical_to_logical": dict(mapping), "total_calibration_error": total_error})

    candidates.sort(key=lambda c: c["total_calibration_error"])
    candidates = candidates[:max_candidates]
    if verbose:
        print(f"  Factibilidad estructural: SI -- VF2 encontro "
              f"{len(candidates)} candidatos en {time.time()-t0:.1f}s "
              f"(de un espacio de busqueda mucho mayor, cortado por timeout/limite).")
        for i, c in enumerate(candidates):
            print(f"    candidato {i}: error_calibracion_acumulado={c['total_calibration_error']:.5f}")
    return {"feasibility": feas, "candidates": candidates}


if __name__ == "__main__":
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    backend = FakeTorino()
    print("Test de factibilidad estructural (grado del circuito vs. grado del hardware)"
          " sobre todos los circuitos de referencia del proyecto:\n")

    circuits = [
        "adder8.qasm", "qft6.qasm", "qft10.qasm", "random_circuit8.qasm",
        "qaoa_small_test.qasm", "qaoa_farm.qasm", "vqe_ansatz.qasm",
        "grover.qasm", "shor_like_sequential.qasm",
    ]
    results = []
    for path in circuits:
        try:
            results.append(structural_feasibility(path, backend, verbose=True))
        except Exception as e:
            print(f"  {path}: no se pudo evaluar ({e})")

    print("\nPara el/los circuito(s) estructuralmente factibles, se corre la busqueda VF2"
          " (estrategia real del notebook QGSS lab-2):\n")
    for r in results:
        if r["structurally_feasible_without_swaps"] and r["num_qubits_circuit"] <= 10:
            print(f"-- {r['qasm_path']} --")
            find_layout_candidates_by_isomorphism(r["qasm_path"], backend, max_candidates=3)
            print()
