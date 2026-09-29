"""
Implementacion directa (no pyzx, algebra lineal sobre GF(2)) del criterio
EXACTO de fusion de polinomios de fase -- el metodo de Amy, Maslov, Mosca
("Polynomial-Time T-Depth Optimization of Clifford+T Circuits via Matroid
Partitioning", 2014, la base de la herramienta `t-par`), que NO es una
aproximacion ni una heuristica: es un calculo algebraico determinista.

Idea: dentro de un bloque de circuito compuesto solo por CNOT y rotaciones-Z
(sin Hadamard, que cambia de base y "rompe" el bloque), cada rotacion-Z que
actua sobre el qubit q en realidad esta rotando la PARIDAD acumulada de q
(el XOR de que qubits de entrada contribuyeron a el via los CNOTs previos),
no el qubit q "puro". Dos rotaciones-Z se pueden fusionar en una sola
(sumando sus angulos) si y solo si tienen EXACTAMENTE la misma paridad en
el momento en que se aplican -- sin importar que tan lejos esten una de
otra en el circuito, y SIN necesitar ningun pivoting/reestructuracion
heuristica del diagrama. Esto es lo que `gadget_simp` de pyzx SI puede
perder (solo fusiona lo que ya esta adyacente en su representacion de
grafo), pero el algebra lineal nunca pierde, porque no depende de la forma
del diagrama intermedio.

Limitacion declarada: esto solo aplica DENTRO de un bloque CNOT+RZ (entre
compuertas H, que cambian de base Z<->X y rompen la validez del rastreo
de paridad). Un circuito con muchos H intercalados (como el Toffoli
descompuesto, o QFT con H en cada qubit) queda partido en varios bloques,
y solo se fusiona DENTRO de cada bloque -- exactamente como una
implementacion real de t-par lo haria.
"""

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
import pyzx as zx


def parity_blocks(circuit: zx.Circuit):
    """
    Recorre las compuertas y devuelve una lista de bloques. Cada bloque es
    una lista de (paridad: frozenset(qubits), angulo) para cada rotacion-Z
    encontrada, con la paridad acumulada de CNOTs DENTRO de ese bloque
    (todos los bloques arrancan con paridad = {q} para cada qubit q, es
    decir, se reinicia en cada H -- una aproximacion conservadora y
    correcta: no se intenta rastrear a traves de un cambio de base).
    """
    n = circuit.qubits
    blocks = []
    parity = {q: frozenset([q]) for q in range(n)}
    current_block = []

    def gate_qubits(gate):
        qs = []
        for attr in ("target", "control"):
            if hasattr(gate, attr):
                qs.append(getattr(gate, attr))
        return qs

    for gate in circuit.gates:
        name = type(gate).__name__
        qs = gate_qubits(gate)

        if name == "HAD":
            # cambio de base -- cerrar el bloque actual y reiniciar paridad
            if current_block:
                blocks.append(current_block)
            current_block = []
            q = qs[0]
            parity[q] = frozenset([q])  # se reinicia (conservador y correcto)

        elif name in ("CNOT", "CZ"):
            # CNOT: la paridad del target se vuelve XOR con la del control.
            # CZ es diagonal, no cambia parity para efectos de rotaciones-Z
            # posteriores -- se ignora aqui (no rompe ni modifica bloques)
            if name == "CNOT":
                c, t = gate.control, gate.target
                parity[t] = parity[t].symmetric_difference(parity[c])

        elif name in ("ZPhase", "S", "T", "Z"):
            q = qs[0] if qs else gate.target
            phase = getattr(gate, "phase", None)
            adjoint = getattr(gate, "adjoint", False)
            defaults = {"S": 0.5, "Z": 1.0, "T": 0.25}
            p = float(phase) if phase is not None else defaults.get(name, 0.25)
            if adjoint:
                p = -p
            current_block.append((parity[q], p % 2.0))

        elif name in ("NOT", "XPhase"):
            # Pauli-X puro (fase multiplo de 1.0): no es una rotacion, no
            # cuenta como no-Clifford, se ignora igual que antes.
            # XPhase con angulo arbitrario (p.ej. el mezclador RX de QAOA,
            # o el RY de un ansatz VQE ya descompuesto a ZPhase-XPhase-
            # ZPhase): BUG CORREGIDO -- antes se ignoraba por completo y
            # desaparecia del conteo "despues", inflando artificialmente
            # la ganancia calculada (la fusion en base-X necesitaria su
            # propio rastreo de paridad dual, que NO se implementa aqui;
            # lo honesto es contarla como no-fusionable en este analisis,
            # no borrarla del conteo)
            phase = getattr(gate, "phase", None)
            if phase is not None:
                p = float(phase) % 2.0
                is_clifford = abs(p % 0.5) < 1e-9 or abs((p % 0.5) - 0.5) < 1e-9
                if not is_clifford:
                    unique_key = frozenset([f"__X_BASIS_NOFUSE__{id(gate)}"])  # unico -> nunca coincide con otro
                    current_block.append((unique_key, p))

        else:
            # cualquier otra compuerta desconocida: cerrar el bloque para
            # no arriesgar una fusion invalida
            if current_block:
                blocks.append(current_block)
            current_block = []

    if current_block:
        blocks.append(current_block)
    return blocks


def exact_phase_poly_tcount_after_fusion(circuit: zx.Circuit) -> int:
    """
    Cuenta cuantas rotaciones no-Clifford quedarian tras fusionar TODAS las
    que comparten paridad EXACTA dentro de cada bloque (algebra lineal
    exacta, sin heuristica). Una rotacion Clifford (multiplo de 0.5, es
    decir 0, pi/2, pi, 3pi/2) no cuenta como "T-like"; el resto si.
    """
    blocks = parity_blocks(circuit)
    total_after = 0
    for block in blocks:
        groups = defaultdict(float)
        for parity, phase in block:
            groups[parity] = (groups[parity] + phase) % 2.0
        for parity, total_phase in groups.items():
            # es Clifford (y por tanto no cuenta como no-Clifford) si el
            # angulo combinado es multiplo de 0.5 (en unidades de pi)
            if abs(total_phase % 0.5) > 1e-9 and abs((total_phase % 0.5) - 0.5) > 1e-9:
                total_after += 1
    return total_after


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from build_circuit_dataset import generate_circuits as gen1
    from build_circuit_dataset_v2 import generate_circuits as gen2
    from qiskit import qasm2
    from qasm_hypergraph import resource_report

    print(f"{'circuito':26s} {'T-count antes':>14s} {'ganancia REAL (full_reduce)':>28s} {'ganancia ALGEBRA LINEAL':>24s}")
    for gen in (gen1, gen2):
        for name, qc in gen():
            qasm_str = qasm2.dumps(qc)
            circuit = zx.Circuit.from_qasm(qasm_str).to_basic_gates()
            tcount_before = resource_report(circuit)["tcount"]
            if tcount_before == 0:
                continue
            tcount_algebra = exact_phase_poly_tcount_after_fusion(circuit)
            gain_algebra = 100.0 * (tcount_before - tcount_algebra) / tcount_before
            print(f"{name:26s} {tcount_before:14d} {'(ver script anterior)':>28s} {gain_algebra:23.1f}%")
