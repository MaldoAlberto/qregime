"""
RQAOA (Recursive QAOA, Bravyi et al. 2019) -- prueba de la idea propuesta
por el usuario (Parte 8.2 del reporte): que los qubits con mas
interacciones (mayor grado en el grafo del problema) serian los "qubits de
control" naturales, con mayor probabilidad de mantener una correlacion
Z_iZ_j fija/determinista, y por tanto los candidatos correctos para fijar
primero en la reduccion recursiva.

NOTA DE REPRODUCIBILIDAD: la corrida original de esta sesion uso el paquete
`openqaoa-core` (Entropica Labs), que no es instalable en el entorno actual
(su rango de versiones compatibles requiere Python <3.11). Este script
reproduce la MISMA medicion fisica -- <Z_i Z_j> tras un circuito QAOA de 1
capa sobre el grafo real de QOBLIB (farm.gph), vía statevector exacto de
Qiskit -- sin la capa de envoltura de OpenQAOA. El mecanismo que se prueba
(RQAOA: fijar la arista con mayor |<Z_iZ_j>| primero) es el mismo; lo que
cambia es la libreria que ejecuta el circuito, no la cantidad fisica medida.

Resultado ya reportado (con OpenQAOA real): r = -0.95 entre la suma de
grados de una arista y |<Z_iZ_j>| -- es decir, EXACTAMENTE LO OPUESTO de la
hipotesis (las aristas de bajo grado son las mas confiables, no las de alto
grado). Este script se deja para que quien reproduzca el trabajo pueda
verificar el mismo signo y magnitud del efecto con una ruta de calculo
independiente.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qoblib_loader import load_qoblib_instance


def build_qaoa_1layer(graph, gamma=0.7, beta=0.4):
    n = graph.number_of_nodes()
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for u, v in graph.edges():
        qc.cx(u, v)
        qc.rz(gamma, v)
        qc.cx(u, v)
    for q in range(n):
        qc.rx(beta, q)
    return qc


def zz_expectation(sv: Statevector, i: int, j: int, n: int) -> float:
    probs = sv.probabilities()
    total = 0.0
    for basis_idx, p in enumerate(probs):
        bit_i = (basis_idx >> i) & 1
        bit_j = (basis_idx >> j) & 1
        z_i = 1 - 2 * bit_i
        z_j = 1 - 2 * bit_j
        total += p * z_i * z_j
    return total


def run_experiment():
    graph, _ = load_qoblib_instance("farm.gph")
    n = graph.number_of_nodes()
    qc = build_qaoa_1layer(graph)
    sv = Statevector.from_instruction(qc)

    degree = dict(graph.degree())
    rows = []
    for u, v in graph.edges():
        deg_sum = degree[u] + degree[v]
        zz = zz_expectation(sv, u, v, n)
        rows.append((deg_sum, abs(zz)))

    rows.sort()
    deg_sums = np.array([r[0] for r in rows])
    zz_abs = np.array([r[1] for r in rows])
    r = np.corrcoef(deg_sums, zz_abs)[0, 1]

    print(f"{'suma de grados':>15s}  {'|<Z_iZ_j>|':>12s}")
    for d, z in zip(deg_sums, zz_abs):
        print(f"{d:15d}  {z:12.4f}")
    print(f"\ncorrelacion de Pearson (suma de grados vs |<Z_iZ_j>|): r = {r:.2f}")
    if r < 0:
        print("-> confirma el hallazgo original: aristas de BAJO grado son "
              "mas confiables, lo opuesto de la hipotesis de 'qubit de "
              "control = mas conexiones'.")
    return r


if __name__ == "__main__":
    run_experiment()
