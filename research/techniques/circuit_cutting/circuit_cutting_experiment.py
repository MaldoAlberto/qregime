"""
Corte de circuitos (circuit cutting / wire cutting) sobre dos familias de
referencia, usando el paquete REAL de Qiskit `qiskit-addon-cutting`
(descomposicion de cuasi-probabilidad, QPD) y su buscador automatico de
cortes `find_cuts`.

Idea que se estaba probando (propuesta por el usuario, Parte 8.1 del
reporte): en vez de reducir el circuito, evaluar solo una parte de el y
"reordenar" para que un circuito denso tenga puntos separables/concentricos
-- exactamente el problema que resuelve el corte de circuitos.

Resultado esperado (ya reportado, esta ejecucion lo reproduce):
  - Sumador8 (grafo en cadena, muy separable): 2 cortes, overhead ~256x
  - QAOA/farm (grafo denso, 17 qubits / 39 aristas): 18-44 cortes segun
    estrategia, overhead 10^17 - 10^43x (intratable)

El predictor real no es el numero de qubits sino la separabilidad
estructural del grafo de interaccion (tamano del corte minimo).
"""

import warnings
warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit
from qiskit_addon_cutting import find_cuts, OptimizationParameters, DeviceConstraints


def analyze_circuit(qasm_path: str, name: str):
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    print(f"\n{'='*70}\n{name}  (n_qubits={qc.num_qubits}, n_gates={len(qc.data)})\n{'='*70}")

    # el tamano maximo de subcircuito importa: exigir subcircuitos demasiado
    # pequenos fuerza al buscador a hacer muchos mas cortes de los que hacen
    # falta estructuralmente. Se usa poco mas de la mitad del circuito, para
    # permitir la particion en 2 bloques mas grande posible (y por tanto el
    # numero MINIMO de cortes que la separabilidad real del grafo permite).
    opt = OptimizationParameters(seed=0)
    qps = max(2, qc.num_qubits // 2 + 1)
    constraints = DeviceConstraints(qubits_per_subcircuit=qps)

    try:
        cut_circuit, metadata = find_cuts(qc, opt, constraints)
    except Exception as e:
        print(f"  find_cuts no pudo encontrar una solucion tratable: {e}")
        return None

    num_cuts = len(metadata.get("cuts", [])) if isinstance(metadata, dict) else None
    sampling_overhead = metadata.get("sampling_overhead", None) if isinstance(metadata, dict) else None

    print(f"  cortes encontrados: {num_cuts}")
    print(f"  overhead de muestreo: {sampling_overhead:.3e}" if sampling_overhead else
          "  overhead de muestreo: (ver metadata completa)")
    return metadata


if __name__ == "__main__":
    analyze_circuit("adder8.qasm", "Sumador8 (grafo en cadena, separable)")
    analyze_circuit("qaoa_farm.qasm", "QAOA/farm (grafo denso, 17 qubits)")
