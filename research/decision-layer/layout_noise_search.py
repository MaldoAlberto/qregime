"""
ETAPA 0 (nueva, antes de estimar QEM o FTQC): buscar, entre varias
configuraciones REALES de transpilacion (no una heuristica de calibracion
aislada -- ver mas abajo por que esa fallo), la que da el MENOR error
fisico efectivo medido para ESTE circuito en ESTE backend.

Por que esto y no una heuristica de "elegir la region mejor calibrada":
se probo eso primero (buscar el subgrafo de 16 qubits con menor error
promedio de enlace) y dio PEOR resultado que una busqueda multi-semilla
del transpilador completo (2,071,530 vs 470,970 qubits fisicos finales
en el circuito de prueba) -- la calidad de calibracion de una region NO
predice el error real una vez que se rutea el circuito ahi (una region
bien calibrada en promedio puede forzar mas SWAPs si su topologia no
encaja con el grafo de interaccion del circuito, y esos SWAPs adicionales
pesan mas que la ganancia de calibracion). La busqueda conjunta
layout+ruteo (que SabreLayout ya hace internamente, de forma distinta con
cada semilla) es la que hay que explorar, no una metrica de calibracion
por separado.

Este modulo simplemente prueba N semillas del transpilador real
(optimization_level=3, que ya activa SabreLayout con refinamiento
iterativo) y se queda con la que da el menor avg_error MEDIDO (no
estimado) sobre el circuito real ya transpilado -- el mismo
real_two_qubit_error_rate que el resto del proyecto ya usa para todo.
"""
import time

from qiskit import QuantumCircuit, transpile
from case_study import real_two_qubit_error_rate


def find_best_layout(qasm_path: str, backend, n_trials: int = 20,
                       optimization_level: int = 3, verbose: bool = False):
    qc = QuantumCircuit.from_qasm_file(qasm_path)

    best = None
    t0 = time.time()
    for seed in range(n_trials):
        transpiled = transpile(qc, backend=backend, optimization_level=optimization_level,
                                 seed_transpiler=seed)
        avg_error, fidelity, n2q, shot_time_s = real_two_qubit_error_rate(transpiled, backend)
        if best is None or avg_error < best["avg_error"]:
            best = {"seed": seed, "avg_error": avg_error, "fidelity": fidelity,
                    "num_2q_gates": n2q, "shot_time_s": shot_time_s,
                    "transpiled_circuit": transpiled}
        if verbose:
            print(f"    semilla={seed:2d}  2q_gates={n2q:5d}  avg_error={avg_error:.5f}"
                  f"{'  <- mejor hasta ahora' if best['seed'] == seed else ''}")

    if verbose:
        print(f"  mejor semilla: {best['seed']}  avg_error={best['avg_error']:.5f}  "
              f"({n_trials} configuraciones probadas en {time.time()-t0:.1f}s)")
    return best


if __name__ == "__main__":
    import sys
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from pipeline_v2 import estimate_with_azure_qre

    qasm_path = sys.argv[1] if len(sys.argv) > 1 else "shor_like_sequential.qasm"
    backend = FakeTorino()

    print(f"Buscando la mejor configuracion de layout/ruteo para {qasm_path} ({20} semillas)...")
    best = find_best_layout(qasm_path, backend, n_trials=20, verbose=True)

    from qasm_hypergraph import load_qasm_circuit, resource_report
    metrics = resource_report(load_qasm_circuit(qasm_path).to_basic_gates())
    tcount, nq = metrics["tcount"], metrics["num_qubits"]

    az = estimate_with_azure_qre(num_logical_qubits=nq, tcount=tcount,
                                   p_phys=max(best["avg_error"], 1e-6))
    print(f"\n  con la mejor configuracion encontrada: d={az['logicalQubit']['codeDistance']}  "
          f"qubits_fisicos={az['physicalCounts']['physicalQubits']:,}")
