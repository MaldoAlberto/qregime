"""
regime_decision.py -- formaliza la decision QEM vs FTQC (con un aviso
explicito de la BRECHA de QEC que este proyecto no resuelve) como un
problema de OPTIMIZACION CON RESTRICCIONES, en vez de una comparacion ad
hoc caso por caso (que es lo que hacia `decide_regime()` en
`comparative_benchmark_multi.py` hasta ahora: comparaba tiempo esperado
QEM vs FTQC con una regla if/else, sin una formulacion explicita del
problema que se esta resolviendo).

IMPORTANTE, para ser honestos sobre que tipo de contribucion es esto: a
diferencia de las Proposiciones 1 y 2 del paper (afirmaciones matematicas
demostradas), esto NO es un teorema -- es una FORMALIZACION /
DEFINICION de un problema de decision, construida sobre piezas que este
proyecto ya valido por separado (el estimador FTQC propio con fabricas de
Litinski, las mediciones empiricas de exito de QEM via simulacion
ruidosa real, y el certificado estructural de simulabilidad clasica de
las Proposiciones 1-2). El valor no es una prueba nueva, es hacer
EXPLICITO y AUDITABLE el criterio de decision que antes vivia implicito
en un if/else.

------------------------------------------------------------------------
DEFINICION (problema de decision QEM-vs-FTQC, con brecha de QEC declarada)
------------------------------------------------------------------------
Dado un circuito con perfil P = (n, tcount, tw, ef) [qubits, T-count,
treewidth estatico, fraccion de entrelazamiento final -- Proposiciones 1
y 2], un contexto de hardware (p_2q, presupuesto fisico de qubits B) y un
umbral de confiabilidad objetivo epsilon_target in (0,1]:

  regimen* = argmin_{r in {QEM, FTQC}} Costo(r)
             sujeto a   Confiabilidad(r) >= epsilon_target
                    y   Costo(r) <= B

  donde:
    Costo(QEM)          = n                          (qubits fisicos, sin overhead de codigo)
    Confiabilidad(QEM)  = exito medido empiricamente  (simulate_expectation, NO una formula -- una medicion)
    Costo(FTQC)         = physicalQubits              (open_resource_estimator + fabricas de Litinski, ESTE proyecto)
    Confiabilidad(FTQC) = 1 - error_budget            (parametro de diseño; se logra POR CONSTRUCCION si es factible)

  Si el conjunto factible es vacio (ni QEM alcanza epsilon_target, ni
  FTQC cabe en el presupuesto B), se reporta una BRECHA CUANTIFICADA en
  vez de forzar una respuesta: este es exactamente el espacio que un
  codigo de correccion de errores PARCIAL (QEC intermedio, ni NISQ+QEM ni
  FTQC completo) tendria que llenar -- espacio que este proyecto no tiene
  herramienta para estimar todavia, y se declara asi en vez de forzar una
  respuesta con el estimador equivocado.

  ADEMAS, antes de resolver el problema de arriba, se consulta el
  certificado de simulabilidad clasica (Proposiciones 1-2): si es
  positivo (tw<=20 Y ef<50%), se antepone un aviso de "verificar atajo
  clasico antes de gastar CUALQUIER recurso cuantico" -- no fuerza la
  decision (es de confianza baja, como se declaro en las Proposiciones),
  pero se muestra siempre primero.
"""
import json

from .resources import effective_tcount, estimate_open_ftqc_with_factories
from .treewidth import treewidth_simulability_certificate
from .advisor import profile_circuit
from .partial_qec import sweep_partial_qec


# presupuesto fisico de referencia: el mayor procesador superconductor
# real publicamente reportado a la fecha de este proyecto (IBM "Condor",
# 1,121 qubits fisicos, anunciado diciembre 2023) -- se declara como
# ORDEN DE MAGNITUD de lo alcanzable HOY, no como limite fisico teorico;
# ver bibitem ibmcondor2023 en el paper
DEFAULT_PHYSICAL_QUBIT_BUDGET = 1121

# umbral de confiabilidad por defecto: coincide con el veredicto ya usado
# en el dashboard ("success>=50% -> usable hoy") -- declarado, no
# calibrado contra una aplicacion real con requisito de precision propio
DEFAULT_RELIABILITY_TARGET = 0.5


def decide_regime_formal(qasm_path: str, tcount: int, num_qubits: int,
                          p_phys: float, qem_success_fraction: float,
                          error_budget: float = 1e-2,
                          reliability_target: float = DEFAULT_RELIABILITY_TARGET,
                          physical_qubit_budget: int = DEFAULT_PHYSICAL_QUBIT_BUDGET,
                          verbose: bool = True, profile: dict = None,
                          rotation_count: int = 0) -> dict:
    """``tcount`` is the number of exact T gates; ``rotation_count`` the
    arbitrary-angle rotations, whose Clifford+T synthesis cost is added
    before estimating FTQC and partial QEC (see resources.effective_tcount).
    With the default rotation_count=0 the behaviour is unchanged."""
    # paso 0: aviso de atajo clasico (Proposiciones 1-2), consultivo, no vinculante
    tw_cert = treewidth_simulability_certificate(qasm_path, verbose=False)
    # Same duplicated-work pattern as Bug 1 (see advisor.recommend_techniques):
    # reuse a profile the caller already computed instead of re-parsing.
    prof = profile if profile is not None else profile_circuit(qasm_path)
    ef = prof.get("final_state_entanglement_fraction")
    classical_shortcut_flag = (
        tw_cert["best_treewidth_upper_bound"] <= 20 and ef is not None and ef < 0.5
    )

    # candidato QEM: costo = qubits fisicos usados directamente, sin overhead de codigo
    cost_qem = num_qubits
    reliability_qem = qem_success_fraction
    qem_meets_target = reliability_qem >= reliability_target

    # FTQC is designed for reliability 1 - error_budget. When the target is
    # stricter, design it for the target instead, so FTQC meets the target
    # by construction rather than being declared feasible while falling
    # short of it (audit finding #2: the target used to be ignored for FTQC).
    error_budget = min(error_budget, 1 - reliability_target)

    # T gates the FTQC run actually consumes, including the synthesis of
    # arbitrary rotations, and the error budget left for logical errors.
    tcount_ftqc, logical_error_budget = effective_tcount(tcount, rotation_count, error_budget)

    # candidato FTQC: nuestro propio estimador (con fabricas de Litinski, ya validado)
    ftqc = estimate_open_ftqc_with_factories(
        num_logical_qubits=num_qubits, tcount=tcount_ftqc, p_phys=p_phys,
        error_budget=logical_error_budget, verbose=False,
    )
    # An infeasible estimate (p_phys at or above threshold, or d > d_max)
    # has no physicalQubits: it is reported as such instead of raising
    # KeyError (audit finding #7).
    ftqc_estimable = error_budget > 0 and bool(ftqc.get("feasible"))
    cost_ftqc = ftqc["physicalQubits"] if ftqc_estimable else None
    reliability_ftqc = 1 - error_budget  # por diseño, si es factible
    ftqc_fits_budget = ftqc_estimable and cost_ftqc <= physical_qubit_budget
    # FTQC is designed for 1 - error_budget; a stricter target is NOT met
    # by construction (audit finding #2: this used to be ignored).
    ftqc_meets_target = ftqc_estimable and reliability_ftqc >= reliability_target
    ftqc_feasible_in_budget = ftqc_fits_budget and ftqc_meets_target

    # Partial QEC is a candidate of the argmin in its own right, not only a
    # fallback when QEM and FTQC both fail: it can be cheaper than a FTQC
    # that fits the budget (audit finding #3). Ver partial_qec_sweep.py para
    # el metodo y su limitacion declarada (extrapolacion de un modelo
    # calibrado para d grande, no un esquema de QEC parcial real y publicado).
    partial_qec = sweep_partial_qec(
        num_logical_qubits=num_qubits, tcount=tcount_ftqc, p_phys=p_phys,
        reliability_target=reliability_target, physical_qubit_budget=physical_qubit_budget,
        verbose=False,
    )

    # Order matters only on ties: QEM first (no code at all), then FTQC, so
    # a partial-QEC point that costs the same as full FTQC is reported as FTQC.
    feasible = []
    if qem_meets_target:
        feasible.append(("QEM", cost_qem))
    if ftqc_feasible_in_budget:
        feasible.append(("FTQC", cost_ftqc))
    if partial_qec.get("feasible"):
        feasible.append((f"QEC_PARCIAL (d={partial_qec['chosen_d']})", partial_qec["cost_qubits"]))

    if feasible:
        regime, cost = min(feasible, key=lambda kv: kv[1])
        gap_report = None
    else:
        regime = "BRECHA_SIN_CERRAR (ni QEM, ni QEC parcial, ni FTQC alcanzan la restriccion)"
        cost = None
        gap_report = {
            "gap_de_confiabilidad_QEM": max(0.0, reliability_target - reliability_qem),
            "gap_de_qubits_FTQC": (max(0, cost_ftqc - physical_qubit_budget)
                                   if ftqc_estimable else None),
            "qec_parcial_mejor_intento": partial_qec.get("best_attempt"),
        }

    result = {
        "qasm_path": qasm_path,
        "classical_shortcut_advisory": classical_shortcut_flag,
        "treewidth_static": tw_cert["best_treewidth_upper_bound"],
        "entanglement_fraction": ef,
        "cost_qem_qubits": cost_qem,
        "reliability_qem_measured": reliability_qem,
        "qem_meets_target": qem_meets_target,
        "cost_ftqc_qubits": cost_ftqc,
        "ftqc_estimable": ftqc_estimable,
        "ftqc_infeasible_reason": None if ftqc_estimable else ftqc.get("reason"),
        "ftqc_fits_budget": ftqc_fits_budget,
        "ftqc_meets_target": ftqc_meets_target,
        "ftqc_reliability": reliability_ftqc,
        "ftqc_feasible_in_budget": ftqc_feasible_in_budget,
        "tcount_ftqc": tcount_ftqc,
        "physical_qubit_budget": physical_qubit_budget,
        "reliability_target": reliability_target,
        "chosen_regime": regime,
        "chosen_cost_qubits": cost,
        "gap_report": gap_report,
    }
    if verbose:
        flag = " [AVISO: posible atajo clasico -- verificar antes de gastar recursos cuanticos]" if classical_shortcut_flag else ""
        print(f"  {qasm_path:40s} QEM(n={cost_qem},rel={reliability_qem:.2f}) "
              f"FTQC(n={cost_ftqc if cost_ftqc is None else f'{cost_ftqc:,}'},"
              f"factible={ftqc_feasible_in_budget}) -> {regime}{flag}")
    return result


def run_over_dashboard_data(dashboard_json_path: str = "dashboard_data.json",
                             hardware_key: str = "superconductor_ibm"):
    with open(dashboard_json_path) as f:
        data = json.load(f)
    p_2q = data["hardware_profiles"][hardware_key]["p_2q"]

    results = {}
    for circuit_name, by_hw in data["circuits"].items():
        report = by_hw.get(hardware_key)
        if report is None:
            continue
        p = report["profile"]
        sim = report.get("simulated_expectation_today", {})
        qem_success = sim.get("success_fraction_measured", 0.0)
        r = decide_regime_formal(
            qasm_path=circuit_name, tcount=p["tcount"], num_qubits=p["num_qubits"],
            p_phys=p_2q, qem_success_fraction=qem_success,
        )
        results[circuit_name] = r
    return results


if __name__ == "__main__":
    print(f"Formalizando la decision QEM-vs-FTQC como optimizacion con restricciones "
          f"(perfil de hardware: superconductor_ibm, presupuesto={DEFAULT_PHYSICAL_QUBIT_BUDGET} "
          f"qubits fisicos, umbral de confiabilidad={DEFAULT_RELIABILITY_TARGET}):\n")
    results = run_over_dashboard_data()
    print("\nResumen:")
    for c, r in results.items():
        print(f"  {c:40s} -> {r['chosen_regime']}"
              + (f"  (brecha: {r['gap_report']})" if r["gap_report"] else ""))
