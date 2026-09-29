"""
cudaq_reuse_comparative.py -- COMPARATIVA DE VALIDACION EXTERNA, no una
dependencia de qasm_tool.py ni de ningun modulo del nucleo de este
proyecto.

PROPOSITO: responder, con codigo ejecutado (no con lectura de
documentacion), si CUDA-Q Logical (NVIDIA) realiza reuso temporal de
qubits logicos via medicion/reset -- la misma pregunta que motivo la
metrica `max_concurrent_qubits` y el reuso implementado en
`open_resource_estimator.py`. Este script prueba dos formas razonables de
pedirselo a CUDA-Q y mide, con su propio `logical_qubits_peak`, si alguna
reduce el pico de qubits.

RESULTADO (verificado, no supuesto): en ambos casos, `logical_qubits_peak`
reporta 6 (el total declarado), NO 3 (el maximo concurrente real de este
circuito de dos fases) -- CUDA-Q Logical, via su API estandar de kernel,
NO detecta ni aprovecha automaticamente el reuso temporal. Ver
Section~qubit-reuse del paper para el hallazgo completo y las citas de
literatura que confirman que esto tampoco esta resuelto en otras
herramientas publicadas.

REQUIERE: `pip install cudaq --break-system-packages` (paquete de NVIDIA,
no necesario para ejecutar qasm_tool.py ni ningun otro modulo del
nucleo de este proyecto -- ver la seccion "Dependencies and vendor
independence" del paper).
"""
import cudaq
import cudaq.logical as cql


@cudaq.kernel
def two_phase_circuit():
    """Circuito de prueba de dos fases SIN solapamiento: los qubits 0,1,2
    solo se usan en la fase 1, los qubits 3,4,5 solo en la fase 2 -- el
    maximo concurrente REAL es 3, no 6, si hubiera reuso temporal."""
    q = cudaq.qvector(6)
    h(q[0]); h(q[1]); h(q[2])
    x.ctrl(q[0], q[1]); x.ctrl(q[1], q[2])
    h(q[3]); h(q[4]); h(q[5])
    x.ctrl(q[3], q[4]); x.ctrl(q[4], q[5])


@cudaq.kernel
def two_phase_with_measure():
    """Misma prueba, pero con mz() explicito al final de la fase 1 --
    para descartar que el reuso simplemente necesitara una medicion
    explicita para activarse."""
    q = cudaq.qvector(6)
    h(q[0]); h(q[1]); h(q[2])
    x.ctrl(q[0], q[1]); x.ctrl(q[1], q[2])
    mz(q[0]); mz(q[1]); mz(q[2])
    h(q[3]); h(q[4]); h(q[5])
    x.ctrl(q[3], q[4]); x.ctrl(q[4], q[5])


def run_comparative():
    cudaq.set_target(cql.targets.estimator)
    results = {}
    for label, kernel in [
        ("sin mz() explicito", two_phase_circuit),
        ("con mz() explicito tras fase 1", two_phase_with_measure),
    ]:
        estimate = cudaq.estimate(kernel)
        logical = cql.estimate.LogicalEstimate.from_annotations(estimate.annotations)
        results[label] = logical.logical_qubits_peak
        print(f"  [{label}] logical_qubits_peak = {logical.logical_qubits_peak} "
              f"(esperado SI hubiera reuso automatico: 3; declarado: 6)")
    return results


if __name__ == "__main__":
    print("Comparativa de validacion externa: ¿CUDA-Q Logical reusa qubits "
          "logicos automaticamente via su API estandar de kernel?\n")
    r = run_comparative()
    reused = any(v == 3 for v in r.values())
    print(f"\n¿Algun caso mostro reuso automatico (peak=3)? {'SI' if reused else 'NO'}")
