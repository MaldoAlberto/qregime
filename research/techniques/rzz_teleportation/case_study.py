"""
Caso de estudio end-to-end, v2 -- con unidades de costo consistentes.

Para un circuito y un BACKEND REAL específico (con su mapa de calibración
real, no un proxy genérico), responde en UNA SOLA unidad comparable
(tiempo esperado, en segundos, hasta obtener una respuesta confiable):

  1. ¿Cuál es la tasa de error real de 2 qubits que este circuito
     experimentaría en este backend específico?
  2. QEM: tiempo esperado = (tiempo real de un shot, calculado desde las
     duraciones de compuerta reales del backend) x (repeticiones
     necesarias, ~1/fidelidad²).
  3. FTQC: tiempo esperado = runtime real que reporta Azure Quantum
     Resource Estimator (qdk.estimator, corre local) para ejecutar el
     circuito completo bajo tolerancia a fallos.
  4. Comparar ambos números, en la misma unidad, más la restricción dura
     de si el presupuesto de qubits alcanza para FTQC.

Esto reemplaza el intento anterior (case_study.py v1), donde mezclar un
costo real (QEM, vía 1/fidelidad) con proxies ilustrativos (FTQC/Híbrido)
producía una comparación inválida -- QEM "ganaba" siempre, no por ser
mejor, sino por estar en otra escala. Ahora los dos lados de la
comparación son números reales, en la misma unidad física.

Advertencia de unidades que SÍ queda (honestidad, no ocultarla): la tasa
de error por compuerta de 2 qubits que reporta el backend NO es
exactamente la misma cantidad física que "p_phys por ciclo de QEC" que usa
Azure QRE -- son cantidades del mismo orden de magnitud y se usan así de
forma estándar en estimadores de recursos, pero no son idénticas por
definición. El modelo de "1/fidelidad²" para QEM es una aproximación
polinomial declarada, no una reproducción de ningún protocolo específico
(ZNE/PEC pueden escalar peor en algunos regímenes).
"""

from qiskit import QuantumCircuit, transpile

from qasm_hypergraph import load_qasm_circuit, resource_report
from zx_reduction import zx_reduce
from pipeline_v2 import estimate_with_azure_qre


def real_two_qubit_error_rate(transpiled_circuit: QuantumCircuit, backend) -> tuple:
    """
    Recorre las compuertas de 2 qubits REALES del circuito ya transpilado a
    ESTE backend, y calcula:
      - el promedio de la tasa de error de esas compuertas específicas
      - la fidelidad esperada de primer orden: producto de (1 - error) de
        cada compuerta de 2 qubits usada (aproximación estándar; ignora
        errores de 1 qubit, de lectura y de decoherencia por tiempo, que
        en un estimador de producción también se incluirían)
      - el tiempo total de ejecución de UN shot del circuito (suma de
        duraciones reales de cada compuerta de 2 qubits, del mapa de
        calibración del backend) -- esto es lo que permite comparar el
        costo de QEM contra el de FTQC en la MISMA unidad (tiempo), en
        vez de mezclar un costo real con proxies ilustrativos sin unidad
        común, que es lo que rompió la comparación en el intento anterior.
    """
    target = backend.target
    two_qubit_ops = [op for op in target.operation_names
                      if target.operation_from_name(op).num_qubits == 2]

    errors = []
    fidelity = 1.0
    total_time_s = 0.0
    for instruction in transpiled_circuit.data:
        op_name = instruction.operation.name
        if op_name in two_qubit_ops:
            qubits = tuple(transpiled_circuit.find_bit(q).index for q in instruction.qubits)
            props = target[op_name].get(qubits)
            if props is not None and props.error is not None:
                errors.append(props.error)
                fidelity *= (1 - props.error)
                if props.duration is not None:
                    total_time_s += props.duration

    avg_error = sum(errors) / len(errors) if errors else 0.0
    return avg_error, fidelity, len(errors), total_time_s


def qem_expected_time_to_reliable_answer(shot_time_s: float, expected_fidelity: float) -> float:
    """
    Costo de QEM en la MISMA unidad que el runtime que reporta Azure QRE
    para FTQC: tiempo esperado (segundos) hasta obtener una respuesta
    confiable.

    shots_needed ~ 1 / fidelidad^2 -- aproximación estándar: la relación
    señal-ruido de una estimación de valor esperado escala con la
    fidelidad, y el error estándar de N repeticiones escala como 1/sqrt(N),
    así que para que la señal supere el piso de ruido impuesto por una
    fidelidad imperfecta se necesitan del orden de 1/fidelidad^2
    repeticiones. Esta es una aproximación polinomial -- el overhead real
    de protocolos específicos de QEM (ZNE, PEC) puede escalar peor
    (exponencialmente en algunos regímenes); se declara así, no se oculta.
    """
    if expected_fidelity <= 0:
        return float("inf")
    shots_needed = 1.0 / (expected_fidelity ** 2)
    return shot_time_s * shots_needed





def run_case_study(qasm_path: str, backend, budget_multiplier: int = 50, verbose: bool = True):
    circuit = load_qasm_circuit(qasm_path)  # pyzx, para métricas T-count/profundidad
    qiskit_circuit = QuantumCircuit.from_qasm_file(qasm_path)

    normalized = circuit.to_basic_gates()
    metrics = resource_report(normalized)

    # transpilar al backend real para conocer las compuertas de 2 qubits
    # EXACTAS que este circuito usaría en este hardware específico
    transpiled = transpile(qiskit_circuit, backend=backend, optimization_level=1, seed_transpiler=0)
    avg_error, expected_fidelity, num_2q_gates, shot_time_s = real_two_qubit_error_rate(transpiled, backend)
    qem_time_s = qem_expected_time_to_reliable_answer(shot_time_s, expected_fidelity)

    if verbose:
        print(f"{'='*70}\nCaso de estudio: {qasm_path}  en  {backend.name}\n{'='*70}")
        print(f"\nCalibración real del hardware:")
        print(f"  compuertas de 2 qubits tras transpilar: {num_2q_gates}")
        print(f"  tasa de error promedio (2q, real, de ESTE circuito en ESTE backend): {avg_error:.4%}")
        print(f"  fidelidad esperada de primer orden (solo ruido de 2q): {expected_fidelity:.4%}")
        print(f"  tiempo de un shot (solo compuertas 2q, real): {shot_time_s*1e9:.1f} ns")
        print(f"  tiempo esperado de QEM hasta respuesta confiable: {qem_time_s:.3e} s")

    # el presupuesto de qubits disponibles para QEC es el tamaño del propio
    # backend multiplicado (asumiendo que en el futuro un dispositivo FTQC
    # de la misma familia tendría más qubits físicos disponibles)
    physical_budget = backend.num_qubits * budget_multiplier

    # --- comparación en UNA sola unidad real: tiempo esperado (segundos)
    # hasta obtener una respuesta confiable. QEM: shot_time * 1/fidelidad^2
    # (calculado arriba). FTQC: runtime real que reporta Azure QRE para
    # ejecutar el circuito COMPLETO bajo tolerancia a fallos, con la
    # ventaja de casi-cero probabilidad de fallo (a diferencia de QEM, que
    # solo reduce el sesgo, no lo elimina).
    reduced = zx_reduce(circuit)
    reduced_metrics = resource_report(reduced)
    azure_result = estimate_with_azure_qre(
        num_logical_qubits=reduced_metrics["num_qubits"],
        tcount=reduced_metrics["tcount"],
        p_phys=max(avg_error, 1e-6),  # tasa de error REAL de este backend
    )
    ftqc_qubits = azure_result["physicalCounts"]["physicalQubits"]
    ftqc_time_s = azure_result["physicalCounts"]["runtime"] / 1e9  # ns -> s
    ftqc_feasible = ftqc_qubits <= physical_budget

    if verbose:
        print(f"\nPresupuesto de qubits considerado: {physical_budget} "
              f"({backend.num_qubits} qubits del backend x {budget_multiplier})")
        print(f"\n{'='*70}\nComparación en la MISMA unidad: tiempo esperado hasta respuesta confiable\n{'='*70}")
        print(f"  QEM  : {qem_time_s:.3e} s  (shot real de {shot_time_s*1e9:.0f} ns x "
              f"1/fidelidad² = {1/max(expected_fidelity,1e-9)**2:.2e} repeticiones)")
        print(f"  FTQC : {ftqc_time_s:.3e} s  (runtime real de Azure QRE, "
              f"requiere {ftqc_qubits:,} qubits físicos -- "
              f"{'FACTIBLE' if ftqc_feasible else 'NO FACTIBLE'} con el presupuesto considerado)")

        if not ftqc_feasible:
            recommendation = "QEM"
            reason = f"FTQC necesita {ftqc_qubits:,} qubits, solo hay {physical_budget:,} disponibles -- restricción dura."
        elif qem_time_s < ftqc_time_s:
            recommendation = "QEM"
            reason = f"QEM es {ftqc_time_s/qem_time_s:.1f}x más rápido en tiempo esperado, y FTQC sí es factible en qubits."
        else:
            recommendation = "FTQC"
            reason = f"FTQC es {qem_time_s/ftqc_time_s:.1f}x más rápido en tiempo esperado hasta una respuesta confiable."
        print(f"\n  Recomendación (unidades consistentes): {recommendation}")
        print(f"  Razón: {reason}")

    result = {
        "qasm_path": qasm_path, "backend": backend.name,
        "avg_2q_error": avg_error, "expected_fidelity": expected_fidelity,
        "num_2q_gates": num_2q_gates, "qem_time_s": qem_time_s,
        "ftqc_time_s": ftqc_time_s, "ftqc_qubits": ftqc_qubits,
        "ftqc_feasible": ftqc_feasible,
    }
    return result


if __name__ == "__main__":
    import sys
    from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeSherbrooke

    qasm_files = sys.argv[1:] if len(sys.argv) > 1 else ["qft6.qasm", "adder8.qasm", "qaoa_farm.qasm"]
    backends = [FakeTorino(), FakeSherbrooke()]

    for qasm_path in qasm_files:
        for backend in backends:
            run_case_study(qasm_path, backend)
            print()
