"""
Etapa 1 del pipeline: circuito QASM -> hipergrafo -> métricas de recursos.

Representación de hipergrafo: los qubits son nodos, y cada compuerta es una
hiperarista que conecta el subconjunto de qubits sobre los que actúa (tamaño
1 para compuertas de un qubit, 2 para CNOT/CZ, 3+ para Toffoli, etc.). Esto
es más general que un grafo simple de interacción y preserva el orden
temporal de aplicación (necesario para calcular profundidad).
"""

import pyzx as zx
import networkx as nx


def gate_qubits(gate):
    """Extrae el conjunto de qubits sobre los que actúa una compuerta de pyzx."""
    qubits = []
    for attr in ("target", "control", "ctrl1", "ctrl2"):
        if hasattr(gate, attr):
            qubits.append(getattr(gate, attr))
    return sorted(set(qubits))


class CircuitHypergraph:
    """
    Hipergrafo de un circuito cuántico.

    - nodes: qubits (0..n-1)
    - hyperedges: lista ordenada de {qubits, gate_type, index}
    """

    def __init__(self, circuit: zx.Circuit):
        self.circuit = circuit
        self.num_qubits = circuit.qubits
        self.hyperedges = []
        for i, gate in enumerate(circuit.gates):
            qs = gate_qubits(gate)
            self.hyperedges.append({
                "qubits": qs,
                "gate_type": type(gate).__name__,
                "is_two_qubit": len(qs) == 2,
                "index": i,
            })

    def interaction_graph(self) -> nx.Graph:
        """
        Colapsa el hipergrafo a un grafo simple ponderado: una arista por cada
        par de qubits que interactúa vía una compuerta de 2 qubits, con peso
        = número de veces que interactúan. Este es el grafo "lógico" que
        alimenta la etapa de layout (igual estructura que usamos con los
        Hamiltonianos de Ising, pero viniendo de un circuito real en vez de
        un problema de optimización).
        """
        g = nx.Graph()
        g.add_nodes_from(range(self.num_qubits))
        for he in self.hyperedges:
            if he["is_two_qubit"]:
                u, v = he["qubits"]
                if g.has_edge(u, v):
                    g[u][v]["weight"] += 1
                else:
                    g.add_edge(u, v, weight=1)
        return g

    def compute_depth_metrics(self):
        """
        Calcula profundidad total y profundidad de compuertas de 2 qubits
        vía layering: cada compuerta se agenda en la capa
        max(capa_actual_de_sus_qubits) + 1.
        """
        last_layer = [0] * self.num_qubits
        last_layer_2q = [0] * self.num_qubits
        total_depth = 0
        two_qubit_depth = 0

        for he in self.hyperedges:
            qs = he["qubits"]
            if not qs:
                continue
            layer = max(last_layer[q] for q in qs) + 1
            for q in qs:
                last_layer[q] = layer
            total_depth = max(total_depth, layer)

            if he["is_two_qubit"]:
                layer2 = max(last_layer_2q[q] for q in qs) + 1
                for q in qs:
                    last_layer_2q[q] = layer2
                two_qubit_depth = max(two_qubit_depth, layer2)

        return total_depth, two_qubit_depth


def load_qasm_circuit(path: str) -> zx.Circuit:
    return zx.Circuit.from_qasm_file(path)


def t_and_rotation_counts(circuit: zx.Circuit) -> tuple:
    """(exact T gates, arbitrary-angle rotations) of a pyzx circuit.

    pyzx's ``tcount()`` counts every non-Clifford phase as ONE T gate. That
    is exact for phases that are odd multiples of pi/4, but a rotation by
    any other angle (denominator > 4 in units of pi) needs many T gates once
    synthesised in Clifford+T. Those are counted separately here so the
    resource estimator can charge their synthesis cost (audit finding #4)."""
    exact = rotations = 0
    for gate in circuit.gates:
        t = gate.tcount()
        if not t:
            continue
        denominator = getattr(getattr(gate, "phase", None), "denominator", None)
        if denominator is not None and denominator > 4:
            rotations += 1
        else:
            exact += t
    return exact, rotations


def resource_report(circuit: zx.Circuit) -> dict:
    """Perfil de métricas de recursos: profundidad, compuertas, T-count."""
    hg = CircuitHypergraph(circuit)
    total_depth, two_qubit_depth = hg.compute_depth_metrics()
    stats = circuit.stats_dict()

    return {
        "num_qubits": circuit.qubits,
        "num_gates": len(circuit.gates),
        "total_depth": total_depth,
        "two_qubit_depth": two_qubit_depth,
        "two_qubit_count": circuit.twoqubitcount(),
        "tcount": circuit.tcount(),
        "tcount_exact": t_and_rotation_counts(circuit)[0],
        "rotation_count": t_and_rotation_counts(circuit)[1],
        "clifford_count": stats.get("clifford", None),
        "hadamard_count": stats.get("had", None),
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_circuit.qasm"
    circuit = load_qasm_circuit(path)
    hg = CircuitHypergraph(circuit)

    print(f"Circuito: {path}")
    print(f"Hiperaristas (compuertas): {len(hg.hyperedges)}")
    ig = hg.interaction_graph()
    print(f"Grafo de interacción: {ig.number_of_nodes()} qubits, "
          f"{ig.number_of_edges()} pares interactuantes")

    report = resource_report(circuit)
    print("\nMétricas de recursos:")
    for k, v in report.items():
        print(f"  {k}: {v}")
