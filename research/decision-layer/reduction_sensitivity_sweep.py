"""
Convierte la reduccion de circuito (T-count) en una VARIABLE DE DECISION
explicita del veredicto QEM-vs-FTQC, en vez de un paso previo que solo se
aplica o no aplica. Pregunta concreta: para el circuito Shor-like (el unico
caso de este proyecto que hoy no cae claramente en ningun regimen), ¿CUANTA
reduccion de T-count haria falta para que el veredicto cambie?

No encontramos ningun trabajo publicado que formule la reduccion de
circuitos (ZX-calculus, teletransportacion, etc.) como palanca de DECISION
de regimen -- la literatura de reduccion de circuitos optimiza recursos
por si mismos; los estimadores de recursos (Azure QRE) no incorporan
reduccion de circuito antes de estimar. Esta es la pregunta que ese cruce
deja sin responder, y este script la responde para un caso real.

Metodologia (declarada explicitamente):
  - El tiempo de QEM se mantiene FIJO en el valor medido HOY (no se modela
    como funcion del T-count, ya que ZX-calculus reduce T-count, no
    necesariamente el conteo de compuertas de 2 qubits que domina el
    tiempo de QEM) -- esto es una eleccion de modelado, declarada, no un
    supuesto oculto.
  - Solo el T-count (y por tanto el costo de FTQC via Azure QRE y el
    estimador propio) se barre hipoteticamente de 0% a 90% de reduccion.
  - p_phys (tasa de error fisico real medida) y num_qubits permanecen
    fijos en sus valores reales medidos -- no se re-simulan para cada
    punto del barrido (seria carisimo); esto asume que la reduccion de
    T-count no cambia significativamente el layout/ruteo, una
    simplificacion razonable ya que ZX-calculus opera sobre la estructura
    no-Clifford del circuito, no sobre su conectividad de 2 qubits.
"""
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qasm_hypergraph import load_qasm_circuit, resource_report
from layout_noise_search import find_best_layout
from case_study import qem_expected_time_to_reliable_answer
from pipeline_v2 import estimate_with_azure_qre
from open_resource_estimator import estimate_open_ftqc
from evaluate_circuit import _signal_or_noise


def sweep(qasm_path: str, backend=None, budget_multiplier: int = 50,
          reduction_fractions=None, verbose: bool = True):
    backend = backend or FakeTorino()
    reduction_fractions = reduction_fractions or [0.0, 0.1, 0.2, 0.3, 0.392, 0.5, 0.6, 0.7, 0.8, 0.9]

    circuit = load_qasm_circuit(qasm_path).to_basic_gates()
    metrics = resource_report(circuit)
    nq, tcount_original = metrics["num_qubits"], metrics["tcount"]

    if verbose:
        print(f"Circuito: {qasm_path}  (qubits={nq}, T-count original={tcount_original})")
        print("Buscando el mejor layout/ruteo real (20 semillas, ETAPA 0)...")
    best_layout = find_best_layout(qasm_path, backend, n_trials=20, optimization_level=3, verbose=False)
    p_phys = max(best_layout["avg_error"], 1e-6)
    qem_time_s = qem_expected_time_to_reliable_answer(best_layout["shot_time_s"], best_layout["fidelity"])
    physical_budget = backend.num_qubits * budget_multiplier

    if verbose:
        print(f"  p_phys medido (mejor de 20 semillas) = {p_phys:.5f}")
        print(f"  tiempo QEM medido (FIJO durante el barrido) = {qem_time_s:.3e} s")
        print(f"  presupuesto de qubits fisicos = {physical_budget:,}\n")
        print(f"{'reduccion':>10s} {'T-count':>8s} {'qubits Azure':>13s} {'factible?':>10s} "
              f"{'tiempo FTQC (Azure)':>20s} {'veredicto':>32s}")

    rows = []
    for f in reduction_fractions:
        tcount = max(1, round(tcount_original * (1 - f)))
        azure = estimate_with_azure_qre(num_logical_qubits=nq, tcount=tcount, p_phys=p_phys)
        ftqc_qubits = azure["physicalCounts"]["physicalQubits"]
        ftqc_time_azure_s = azure["physicalCounts"]["runtime"] / 1e9
        ftqc_feasible = ftqc_qubits <= physical_budget

        own = estimate_open_ftqc(nq, tcount, p_phys=p_phys)
        ftqc_time_own_s = own["runtime_s"] if own.get("feasible") else None

        veredicto, detalle = _signal_or_noise(qem_time_s, ftqc_time_azure_s, ftqc_time_own_s, ftqc_feasible)
        rows.append({"reduction_fraction": f, "tcount": tcount, "ftqc_qubits": ftqc_qubits,
                     "ftqc_feasible": ftqc_feasible, "ftqc_time_azure_s": ftqc_time_azure_s,
                     "veredicto": veredicto})
        if verbose:
            print(f"{f*100:9.1f}% {tcount:8d} {ftqc_qubits:13,d} {'SI' if ftqc_feasible else 'NO':>10s} "
                  f"{ftqc_time_azure_s:20.3e} {veredicto:>32s}")

    # encontrar el punto de cruce (primera fraccion donde cambia el veredicto respecto al 0%)
    baseline_veredicto = rows[0]["veredicto"]
    crossover = next((r for r in rows if r["veredicto"] != baseline_veredicto), None)
    if verbose:
        print()
        if crossover:
            print(f"CRUCE ENCONTRADO: el veredicto cambia de '{baseline_veredicto}' a "
                  f"'{crossover['veredicto']}' entre {rows[rows.index(crossover)-1]['reduction_fraction']*100:.1f}% "
                  f"y {crossover['reduction_fraction']*100:.1f}% de reduccion de T-count "
                  f"(T-count {rows[rows.index(crossover)-1]['tcount']} -> {crossover['tcount']}).")
        else:
            print(f"SIN CRUCE: el veredicto se mantiene '{baseline_veredicto}' en TODO el rango probado "
                  f"(0% a {reduction_fractions[-1]*100:.0f}% de reduccion de T-count). Ninguna cantidad "
                  f"razonable de reduccion de T-count, por si sola, mueve la aguja para este circuito.")
    return rows, crossover


if __name__ == "__main__":
    import sys
    qasm_path = sys.argv[1] if len(sys.argv) > 1 else "shor_like_sequential.qasm"
    sweep(qasm_path)
