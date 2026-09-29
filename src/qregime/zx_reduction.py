"""
Etapa 2 del pipeline: reducción del circuito vía ZX-calculus.

Convierte el circuito a un ZX-diagram, aplica las reglas de reescritura de
`full_reduce` (fusión de spiders, pivotes, eliminación de identidades, etc.),
y extrae de vuelta un circuito equivalente. El T-count del circuito extraído
es la métrica que de verdad importa para estimar el costo de FTQC -- no el
T-count "as compiled" del circuito original.
"""

import pyzx as zx

from .hypergraph import CircuitHypergraph, resource_report


def zx_reduce(circuit: zx.Circuit) -> zx.Circuit:
    """Reduce el circuito vía ZX-calculus y devuelve el circuito extraído."""
    g = circuit.to_graph()
    zx.full_reduce(g)
    g.normalize()
    reduced = zx.extract_circuit(g.copy())
    return reduced.to_basic_gates()


def compare_before_after(circuit: zx.Circuit):
    """
    Reduce el circuito vía ZX-calculus y compara métricas de recursos
    antes/después.

    IMPORTANTE: el circuito "antes" se normaliza primero con
    `to_basic_gates()`. Sin este paso, un circuito que usa compuertas
    genéricas u(theta,phi,lambda) (como las que produce Qiskit al
    sintetizar QFT) reporta T-count = 0 -- no porque no tenga contenido
    no-Clifford, sino porque el contador de T-gates busca compuertas T
    explícitas y no reconoce fases no-Clifford escondidas dentro de una
    compuerta u() genérica. Comparar un circuito con compuertas compuestas
    contra uno ya descompuesto en compuertas elementales (que es lo que
    hace la extracción post-ZX) es una comparación entre bases de
    compuertas distintas, no una comparación válida de "antes vs después".
    """
    normalized_before = circuit.to_basic_gates()
    before = resource_report(normalized_before)
    reduced = zx_reduce(circuit)
    after = resource_report(reduced)
    return before, after, reduced


def print_comparison(before: dict, after: dict):
    keys = ["num_qubits", "num_gates", "total_depth", "two_qubit_depth",
            "two_qubit_count", "tcount"]
    print(f"{'métrica':20s} {'antes':>10s} {'después ZX':>12s} {'delta':>10s}")
    for k in keys:
        b, a = before[k], after[k]
        delta = a - b
        print(f"{k:20s} {b:10d} {a:12d} {delta:+10d}")


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test_circuit.qasm"
    circuit = zx.Circuit.from_qasm_file(path)

    print(f"Reduciendo {path} vía ZX-calculus...")
    before, after, reduced_circuit = compare_before_after(circuit)
    print()
    print_comparison(before, after)
