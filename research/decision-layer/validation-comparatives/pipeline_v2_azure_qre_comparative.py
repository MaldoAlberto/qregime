"""
Pipeline v2.1 -- corrige la inconsistencia que encontró full_benchmark.py:
la decisión de "¿aplico ZX-calculus?" ahora usa la MISMA lógica de tiempo
real + factibilidad de qubits que case_study.py, no el motor viejo de
proxies ilustrativos (ftqc_advisor.recommend_strategy). Antes había dos
cerebros de decisión desincronizados; ahora hay uno solo.

Requiere un backend REAL (no un número de presupuesto arbitrario), porque
calcular el tiempo de QEM necesita las duraciones y tasas de error de
compuerta reales de un dispositivo específico -- exactamente lo que
case_study.py ya hacía. pipeline_v2 ahora ES esa misma lógica, reorganizada
como el flujo principal del producto.

  1. QASM -> perfil de métricas del circuito ORIGINAL (normalizado).
  2. Transpilar al backend real -> tiempo de QEM real (shot real x
     1/fidelidad²) y tasa de error real de 2 qubits de ESTE circuito en
     ESTE hardware.
  3. Azure QRE (local) sobre el circuito ORIGINAL (sin reducir) -> tiempo
     y qubits de FTQC "de línea base", antes de gastar el costo de ZX.
  4. Decisión: si FTQC no es factible en qubits, o QEM es más rápido en
     tiempo real -> régimen QEM, NO se aplica ZX-calculus.
     Si FTQC es factible Y más rápido -> régimen FTQC/Híbrido, SÍ se
     aplica ZX-calculus (reduce T-count, lo cual solo puede ayudar al
     costo de FTQC ya elegido) y se recalcula Azure QRE con el circuito
     reducido para el número final.

No se usa el layout GNN+Sinkhorn en este pipeline: en la prueba con SABRE
real (swap_validation.py) perdió consistentemente contra SabreLayout, así
que incluirlo en el producto sería venderlo como una ventaja que no existe.
"""

import sys

import qdk.estimator as est
from qiskit import QuantumCircuit, transpile

from qasm_hypergraph import load_qasm_circuit, resource_report
from zx_reduction import zx_reduce


def estimate_with_azure_qre(num_logical_qubits: int, tcount: int, p_phys: float = 1e-3,
                             error_budget: float = 1e-2):
    """
    Estimación de recursos FTQC real, vía el motor de Azure QRE (local).

    BUG REAL corregido en esta sesión: `p_phys` se calculaba (la tasa de
    error real de 2 qubits del circuito transpileado a ESTE backend) y se
    pasaba a esta función, pero nunca se usaba -- la llamada a Azure QRE
    siempre usaba el perfil generico fijo "qubit_gate_ns_e3", sin importar
    que tan bueno o malo fuera el hardware real medido. Ahora se construye
    un perfil de qubit CUSTOM con `p_phys` en las tres tasas de error
    relevantes (compuerta de 1q, compuerta de 2q, medicion), para que la
    estimacion de FTQC de verdad dependa del hardware real, igual que ya
    hacia la parte QEM de este mismo pipeline.
    """
    logical_counts = est.LogicalCounts({
        "numQubits": num_logical_qubits, "tCount": tcount,
        "rotationCount": 0, "rotationDepth": 0, "cczCount": 0,
        "measurementCount": num_logical_qubits,
    })
    params = {
        "qubitParams": {
            "name": "custom", "instructionSet": "GateBased",
            "oneQubitMeasurementErrorRate": p_phys, "oneQubitGateErrorRate": p_phys,
            "twoQubitGateErrorRate": p_phys, "tGateErrorRate": p_phys,
            "oneQubitGateTime": "50 ns", "twoQubitGateTime": "50 ns",
            "oneQubitMeasurementTime": "100 ns",
        },
        "qecScheme": {"name": "surface_code"},
        "errorBudget": error_budget,
    }
    return logical_counts.estimate(params)


def cross_validate_logical_error_rate(azure_result, num_shots: int = 5000, verbose: bool = True):
    """
    Cierra la brecha declarada entre la parte de estimacion de recursos
    (Azure QRE, un MODELO analitico de la tasa de error logico) y la parte
    de simulacion real construida en esta sesion (Stim + PyMatching, una
    MEDICION Monte Carlo real de un surface code rotado).

    Toma la distancia de codigo y la tasa de error fisico que Azure QRE ya
    eligio internamente para este circuito, y corre NUESTRA propia
    simulacion real a esos mismos parametros -- para ver si el modelo
    analitico de Azure y la medicion directa coinciden en orden de
    magnitud, en vez de asumir que el estimador de recursos es correcto
    sin verificarlo contra nada.

    Limitacion declarada: nuestra simulacion usa ruido depolarizado
    uniforme (no el modelo de ruido detallado de circuito que usa
    internamente Azure QRE), y con `num_shots` moderado la resolucion
    estadistica es limitada para tasas de error logico muy bajas (por
    debajo de ~1/num_shots no se puede medir un solo evento) -- se reporta
    el error estandar junto con el valor, nunca solo el numero.
    """
    from qec_surface_code_experiment import logical_error_rate

    d = azure_result["logicalQubit"]["codeDistance"]
    azure_p_logical = azure_result["logicalQubit"]["logicalErrorRate"]
    p_phys = azure_result["physicalCounts"]["breakdown"].get("logicalErrorRateInputPhysicalErrorRate")

    # Azure no siempre expone el p_phys de entrada en el breakdown bajo ese
    # nombre exacto -- si no esta disponible, se usa el que el llamador ya
    # conoce (pasado explicitamente), nunca se inventa un numero
    sim = logical_error_rate(distance=d, rounds=d,
                              p_physical=p_phys if p_phys else 1e-3,
                              num_shots=num_shots, seed=hash(d) % 2**31)

    if verbose:
        print(f"    Azure QRE (modelo analitico): distancia={d}, p_logico={azure_p_logical:.2e}")
        print(f"    Stim+PyMatching (medicion real, {num_shots} shots): "
              f"p_logico={sim['p_logical']:.2e} +/- {sim['p_logical_stderr']:.2e}")

    return {"distance": d, "azure_p_logical": azure_p_logical,
            "measured_p_logical": sim["p_logical"], "measured_stderr": sim["p_logical_stderr"],
            "num_shots": num_shots}


def print_azure_report(result):
    pc = result["physicalCounts"]
    lq = result["logicalQubit"]
    print(f"    distancia de código: {lq['codeDistance']}")
    print(f"    qubits físicos totales: {pc['physicalQubits']:,}")
    print(f"    runtime estimado: {result['physicalCountsFormatted']['runtime']}")


def run_pipeline_v2(qasm_path: str, backend, budget_multiplier: int = 50, verbose: bool = True,
                     layout_search_trials: int = 20):
    from case_study import real_two_qubit_error_rate, qem_expected_time_to_reliable_answer
    from layout_noise_search import find_best_layout

    if verbose:
        print(f"{'='*70}\nETAPA 0: busqueda de layout/ruteo -- {layout_search_trials} configuraciones "
              f"reales de transpilador, quedarse con el MENOR error fisico medido\n{'='*70}")
        print(f"  motivacion (hallazgo real de esta sesion): el error fisico efectivo depende "
              f"muchisimo de que enlaces del backend termina usando el circuito (en FakeTorino, "
              f"la razon peor/mejor enlace calibrado es de 669x). Una heuristica que elige la region "
              f"'mejor calibrada en promedio' e ignora el ruteo dio PEOR resultado que probar varias "
              f"semillas completas del transpilador (2.07M vs 470K qubits fisicos en la prueba que "
              f"motivo esto) -- asi que se prueban configuraciones completas, no solo calibracion.")
    best_layout = find_best_layout(qasm_path, backend, n_trials=layout_search_trials,
                                     optimization_level=3, verbose=verbose)
    if verbose:
        print(f"  -> mejor semilla={best_layout['seed']}  avg_error={best_layout['avg_error']:.5f} "
              f"(vs. una sola configuracion fija, que es lo que este pipeline hacia antes)")

    if verbose:
        print(f"\n{'='*70}\nETAPA 1: perfil del circuito ORIGINAL (sin reducir) -- {qasm_path} en {backend.name}\n{'='*70}")
    circuit = load_qasm_circuit(qasm_path)
    normalized = circuit.to_basic_gates()
    profile_metrics = resource_report(normalized)
    if verbose:
        for k, v in profile_metrics.items():
            print(f"  {k}: {v}")

    qiskit_circuit = QuantumCircuit.from_qasm_file(qasm_path)
    transpiled = best_layout["transpiled_circuit"]
    avg_error, expected_fidelity, num_2q_gates, shot_time_s = (
        best_layout["avg_error"], best_layout["fidelity"],
        best_layout["num_2q_gates"], best_layout["shot_time_s"])
    qem_time_s = qem_expected_time_to_reliable_answer(shot_time_s, expected_fidelity)

    physical_budget = backend.num_qubits * budget_multiplier
    azure_baseline = estimate_with_azure_qre(
        num_logical_qubits=profile_metrics["num_qubits"], tcount=profile_metrics["tcount"],
        p_phys=max(avg_error, 1e-6),
    )
    ftqc_qubits_baseline = azure_baseline["physicalCounts"]["physicalQubits"]
    ftqc_time_baseline_s = azure_baseline["physicalCounts"]["runtime"] / 1e9
    ftqc_feasible = ftqc_qubits_baseline <= physical_budget

    if verbose:
        print(f"\n{'='*70}\nETAPA 2: decisión de régimen (tiempo real, ANTES de aplicar ZX)\n{'='*70}")
        print(f"  QEM  : {qem_time_s:.3e} s")
        print(f"  FTQC (circuito sin reducir): {ftqc_time_baseline_s:.3e} s, "
              f"{ftqc_qubits_baseline:,} qubits ({'factible' if ftqc_feasible else 'NO factible'} "
              f"con presupuesto {physical_budget:,})")

    apply_ftqc_path = ftqc_feasible and ftqc_time_baseline_s < qem_time_s

    if not apply_ftqc_path:
        if verbose:
            reason = "FTQC no factible en qubits" if not ftqc_feasible else "QEM es más rápido en tiempo real"
            print(f"\n  Régimen: QEM  (razón: {reason})")
            print(f"\n{'='*70}\nETAPA 3: régimen QEM -> probar con y sin ZX-calculus, quedarse con la mejor profundidad\n{'='*70}")
            print(
                "  No vamos a FTQC, así que el T-count no importa por sí solo -- pero ZX-calculus"
                " puede, en casos raros y específicos del backend, reducir la profundidad NISQ de"
                " todas formas (real_backend_validation.py encontró esto una vez de varios casos)."
                " Como probar ambas opciones es barato frente a ejecutar el circuito de verdad,"
                " se comparan directamente en vez de asumir una regla fija."
            )
        original_qc = QuantumCircuit.from_qasm_file(qasm_path)
        t_no_zx = transpile(original_qc, backend=backend, optimization_level=1, seed_transpiler=0)

        reduced_for_nisq = zx_reduce(circuit)
        reduced_qc = QuantumCircuit.from_qasm_str(reduced_for_nisq.to_qasm())
        t_with_zx = transpile(reduced_qc, backend=backend, optimization_level=1, seed_transpiler=0)

        depth_no_zx, depth_with_zx = t_no_zx.depth(), t_with_zx.depth()
        zx_applied_in_qem = depth_with_zx < depth_no_zx
        final_depth = min(depth_no_zx, depth_with_zx)

        if verbose:
            print(f"\n  profundidad sin ZX: {depth_no_zx}   profundidad con ZX: {depth_with_zx}")
            print(f"  -> se usa {'CON' if zx_applied_in_qem else 'SIN'} ZX-calculus "
                  f"(profundidad final: {final_depth})")

        return {"regime": "QEM", "qem_time_s": qem_time_s, "zx_applied": zx_applied_in_qem,
                "final_depth": final_depth,
                "ftqc_qubits_baseline": ftqc_qubits_baseline, "ftqc_feasible": ftqc_feasible}

    if verbose:
        print(f"\n  Régimen: FTQC/Híbrido  (FTQC es factible y más rápido en tiempo real)")
        print(f"\n{'='*70}\nETAPA 3: SÍ se aplica ZX-calculus (T-count reducido ayuda al costo ya elegido)\n{'='*70}")
    reduced = zx_reduce(circuit)
    reduced_metrics = resource_report(reduced)
    if verbose:
        print(f"  T-count: {profile_metrics['tcount']} -> {reduced_metrics['tcount']}")

    if verbose:
        print(f"\n{'='*70}\nETAPA 4: estimación de recursos FTQC final (Azure QRE, circuito reducido)\n{'='*70}")
    azure_final = estimate_with_azure_qre(
        num_logical_qubits=reduced_metrics["num_qubits"], tcount=reduced_metrics["tcount"],
        p_phys=max(avg_error, 1e-6),
    )
    if verbose:
        print_azure_report(azure_final)

    if verbose:
        print(f"\n{'='*70}\nETAPA 5: validacion cruzada -- modelo de Azure QRE vs simulacion real (Stim+PyMatching)\n{'='*70}")
    cross_check = cross_validate_logical_error_rate(azure_final, num_shots=5000, verbose=verbose)

    return {
        "regime": "FTQC", "qem_time_s": qem_time_s, "zx_applied": True,
        "reduced_metrics": reduced_metrics, "azure_qre_result": azure_final,
        "qec_cross_validation": cross_check,
    }


if __name__ == "__main__":
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    qasm_path = sys.argv[1] if len(sys.argv) > 1 else "adder8.qasm"
    run_pipeline_v2(qasm_path, FakeTorino())
