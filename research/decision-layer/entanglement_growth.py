"""
entanglement_growth.py -- respuesta directa al punto debil encontrado al
comparar contra Tindall et al. 2023: nuestros propios estimadores de
recursos (el de Azure y el nuestro) razonan SOLO sobre qubits, profundidad
y T-count, sin preguntar nunca si el circuito tiene estructura
clasicamente explotable (correlaciones tipo-arbol / bond dimension bajo,
que es EXACTAMENTE lo que Tindall aprovecho contra el circuito heavy-hex
de IBM via belief propagation).

Este script implementa la version MAS SIMPLE y honesta de esa pregunta
que se puede calcular con lo que ya tenemos (sin instalar una libreria
nueva de redes tensoriales): la ENTROPIA DE ENTRELAZAMIENTO BIPARTITA
exacta (von Neumann, en bits) a traves de un corte que parte el circuito
en dos mitades de qubits, medida en FUNCION DE LA PROFUNDIDAD del
circuito (no solo al final).

INTERPRETACION (estandar en la literatura de redes tensoriales, no
inventada aqui): si la entropia CRECE linealmente con la profundidad
hasta saturar cerca del maximo teorico (n_pequeño bits, "ley de volumen"),
el circuito es un candidato genuinamente dificil de simular
clasicamente via MPS/redes tensoriales de bond dimension acotada. Si la
entropia se ESTANCA muy por debajo del maximo (o crece solo
logaritmicamente), el circuito puede ser clasicamente simulable con un
bond dimension modesto, sin importar cuantos qubits o compuertas
declare -- exactamente el mecanismo que Tindall exploto en el circuito de
Kim et al. 2023 (heavy-hex, correlaciones tipo-arbol).

LIMITACION DECLARADA: la biparticion usada aqui es la mas simple posible
(primera mitad de indices de qubit vs segunda mitad, segun el orden del
registro QASM) -- no necesariamente el corte que minimiza el bond
dimension real (un corte informado por la geometria del grafo de
interaccion podria dar una entropia menor y ser mas representativo de la
dificultad real de compresion tipo-arbol). Esto se declara como una
PRIMERA APROXIMACION, no como una medida optima de "compresibilidad
clasica" -- construir esa version exacta (bond dimension real de una MPS
optimizada, como hace Tindall) requeriria una libreria de redes
tensoriales que este proyecto no tiene instalada.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy


def entanglement_growth(qasm_path: str, num_checkpoints: int = 12, verbose: bool = True) -> dict:
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    n = qc.num_qubits
    n_a = n // 2  # primera mitad de qubits (biparticion simple, declarada arriba)
    max_entropy_bits = min(n_a, n - n_a)

    instructions = qc.data
    total = len(instructions)
    checkpoints = sorted(set(
        max(1, int(total * f)) for f in np.linspace(1 / num_checkpoints, 1.0, num_checkpoints)
    ))

    results = []
    for k in checkpoints:
        sub = QuantumCircuit(n)
        for instr in instructions[:k]:
            sub.append(instr.operation, instr.qubits, instr.clbits)
        sv = Statevector.from_instruction(sub)
        rho_a = partial_trace(sv, list(range(n_a, n)))
        s_bits = float(entropy(rho_a, base=2))
        frac = s_bits / max_entropy_bits if max_entropy_bits else 0.0
        results.append({"instructions_applied": k, "fraction_of_circuit": k / total,
                         "entropy_bits": s_bits, "fraction_of_max_entropy": frac})
        if verbose:
            print(f"  {k:5d}/{total} instrucciones ({k/total*100:5.1f}%):  "
                  f"S={s_bits:.3f} bits  ({frac*100:.1f}% del maximo teorico {max_entropy_bits} bits)")

    return {"qasm_path": qasm_path, "num_qubits": n, "bipartition": f"{n_a} | {n - n_a}",
            "max_entropy_bits": max_entropy_bits, "growth": results,
            "final_fraction_of_max": results[-1]["fraction_of_max_entropy"] if results else None}


if __name__ == "__main__":
    circuits = [
        "shor_like_sequential.qasm", "adder8.qasm", "grover.qasm",
        "qft10.qasm", "random_circuit8.qasm", "kicked_ising_analog_16q_deep.qasm",
    ]
    summary = {}
    for c in circuits:
        print(f"\n=== {c} ===")
        r = entanglement_growth(c)
        summary[c] = r["final_fraction_of_max"]

    print("\n\nResumen -- fraccion del maximo teorico de entropia alcanzada al final del circuito:")
    for c, frac in sorted(summary.items(), key=lambda kv: -kv[1]):
        veredicto = "ley de volumen (dificil de comprimir)" if frac > 0.85 else \
            ("intermedio" if frac > 0.5 else "por debajo del maximo (candidato a compresion tipo-arbol)")
        print(f"  {c:40s} {frac*100:5.1f}%  -> {veredicto}")
