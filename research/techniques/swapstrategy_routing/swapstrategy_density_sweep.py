"""
Comparacion SwapStrategy (ruteo estructurado de Qiskit para circuitos con
compuertas de 2 qubits mutuamente conmutables, p.ej. QAOA) vs SabreSwap
(busqueda heuristica generica), sobre grafos aleatorios de 17 qubits con
densidad creciente -- Parte 8.3 del reporte.

Bug encontrado y corregido en la sesion original (se deja documentado aqui
porque es facil de repetir): envolver un PauliEvolutionGate crudo en un
PassManager([Commuting2qGateRouter(strategy)]) SIN antes correr
FindCommutingPauliEvolutions() hace que el router simplemente copie el nodo
sin rutear (0 SWAPs, resultado fisicamente imposible). Hay que insertar
FindCommutingPauliEvolutions() ANTES de Commuting2qGateRouter.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy, FindCommutingPauliEvolutions, Commuting2qGateRouter,
)

NUM_QUBITS = 17
DENSITIES = [0.15, 0.30, 0.45, 0.60, 0.80, 1.00]


def build_cost_operator(graph, gamma=0.7):
    n = graph.number_of_nodes()
    pauli_terms = []
    for u, v in graph.edges():
        label = ["I"] * n
        label[u] = "Z"
        label[v] = "Z"
        pauli_terms.append(("".join(label), gamma))
    return SparsePauliOp.from_list(pauli_terms)


def swapstrategy_route(graph):
    op = build_cost_operator(graph)
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(PauliEvolutionGate(op, time=1.0), range(NUM_QUBITS))

    strategy = SwapStrategy.from_line(list(range(NUM_QUBITS)))
    pm = PassManager([
        FindCommutingPauliEvolutions(),   # sin este pase, el router no-opea
        Commuting2qGateRouter(strategy),
    ])
    routed = pm.run(qc)
    # contar los SWAP en el nivel MAS ALTO (sin decompose): en cuanto se
    # descompone un solo nivel, el router ya sintetiza cada SWAP dentro del
    # bloque conmutante en 3 CX y el conteo de "swap" cae a 0 -- verificado
    # explicitamente (reps=0 -> swap=120; reps=1 -> swap=0, cx=360)
    counts = routed.count_ops()
    # la profundidad SI necesita expandir los bloques compuestos para ser
    # comparable a la profundidad de Sabre (que ya trabaja a nivel de CX)
    depth = routed.decompose(reps=2).depth()
    return counts.get("swap", 0), depth


def sabre_route(graph):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    for u, v in graph.edges():
        qc.cx(u, v)
        qc.rz(0.7, v)
        qc.cx(u, v)

    coupling = CouplingMap.from_line(NUM_QUBITS)
    # SIN forzar basis_gates aqui: si se forzara a ["cx","rz","h"], cada
    # SWAP insertado se sintetizaria en 3 CX y desaparecerian del conteo --
    # se deja el SWAP como instruccion nombrada para poder contarlo
    routed = transpile(
        qc, coupling_map=coupling, routing_method="sabre",
        layout_method="sabre",
        optimization_level=1, seed_transpiler=0,
    )
    counts = routed.count_ops()
    return counts.get("swap", 0), routed.depth()


def run_sweep():
    results = []
    for density in DENSITIES:
        graph = nx.gnp_random_graph(NUM_QUBITS, density, seed=0)
        sw_swaps, sw_depth = swapstrategy_route(graph)
        sa_swaps, sa_depth = sabre_route(graph)
        row = {
            "density": density, "num_edges": graph.number_of_edges(),
            "swapstrategy_swaps": sw_swaps, "swapstrategy_depth": sw_depth,
            "sabre_swaps": sa_swaps, "sabre_depth": sa_depth,
        }
        results.append(row)
        print(f"densidad={density:.2f}  aristas={graph.number_of_edges():4d}   "
              f"SwapStrategy: SWAPs={sw_swaps:4d} depth={sw_depth:4d}   "
              f"Sabre: SWAPs={sa_swaps:4d}  depth={sa_depth:4d}")
    with open("paper/data/density_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nguardado en paper/data/density_sweep.json")
    return results


if __name__ == "__main__":
    run_sweep()
