"""
Responde la pregunta directa: ¿EN QUE PUNTO se puede aplicar FTQC?

No es una pregunta de "si o no" fija -- es una superficie de tres variables:
  (1) que tan bueno es el hardware fisico (p_fisico, tasa de error de 2 qubits)
  (2) cuantos qubits fisicos hay disponibles (presupuesto de qubits)
  (3) que tan grande es el problema que se quiere resolver (T-count, que
      escala con el tamano de la instancia del algoritmo, no solo su tipo)

Se usan los 4 circuitos de referencia YA MEDIDOS de este proyecto como
arquetipos reales (no numeros inventados: la razon compuertas-2q/T-count de
cada arquetipo es la medida de verdad), escalados a tamanos de problema mas
grandes (mismo tipo de circuito, mas qubits/T-count) -- igual que un QFT o
un Adder reales crecen con el tamano de la instancia sin cambiar de forma.

Para cada combinacion se usa:
  - Azure QRE (real, incluye fabricas de destilacion) para qubits/runtime FTQC
  - el modelo de tiempo QEM ya validado en este proyecto (shot_time x 1/fid^2)
  - la regla de "senal vs ruido" ya construida en evaluate_circuit.py
"""
import json
import numpy as np

from pipeline_v2 import estimate_with_azure_qre
from case_study import qem_expected_time_to_reliable_answer

TWO_QUBIT_GATE_TIME_S = 50e-9  # mismo valor usado en todo el proyecto (perfil custom de Azure QRE)

# arquetipos reales: (nombre, num_qubits medido, tcount medido, two_qubit_count medido)
ARCHETYPES = {
    "QFT (fase controlada)": {"num_qubits": 10, "tcount": 135, "two_qubit_count": 415},
    "Adder Cuccaro (Clifford+T)": {"num_qubits": 18, "tcount": 112, "two_qubit_count": 154},
    "QAOA (grafo denso)": {"num_qubits": 17, "tcount": 56, "two_qubit_count": 79},
    "Random (sin estructura)": {"num_qubits": 8, "tcount": 117, "two_qubit_count": 350},
}

SCALE_FACTORS = [1, 10, 100, 1000]  # "mismo tipo de circuito, problema 1x/10x/100x/1000x mas grande"
P_PHYS_GRID = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]  # de hardware actual a metas de hoja de ruta a largo plazo
QUBIT_BUDGET_GRID = [1_000, 10_000, 100_000, 1_000_000]  # de dispositivos actuales a metas publicas de hoja de ruta


def qem_time_estimate(two_qubit_count: int, p_phys: float) -> float:
    fidelity = (1 - p_phys) ** two_qubit_count
    shot_time_s = two_qubit_count * TWO_QUBIT_GATE_TIME_S
    if fidelity < 1e-150:
        # subyace numericamente a 0 para circuitos grandes/hardware ruidoso --
        # esto es real, no un bug: significa que QEM necesitaria un numero de
        # shots no representable (>1e300), es decir, QEM YA fallo por completo
        # a esta escala. Se reporta como un techo finito muy grande (no inf,
        # para poder seguir comparando razones) en vez de dividir por cero.
        return 1e300
    return qem_expected_time_to_reliable_answer(shot_time_s, fidelity)


def sweep(verbose=True):
    rows = []
    for name, arche in ARCHETYPES.items():
        for scale in SCALE_FACTORS:
            nq = arche["num_qubits"]  # el numero de qubits logicos de la instancia se mantiene
            # (mismo circuito logico, ej. mismo QFT de tamano fijo aplicado repetidamente, o
            # instancia con mas profundidad/repeticiones) -- lo que escala es el T-count real
            # de una instancia MAS GRANDE del mismo tipo de problema, y el conteo de 2 qubits
            # escala en la misma proporcion medida en el circuito de referencia real
            tcount = arche["tcount"] * scale
            two_q = arche["two_qubit_count"] * scale

            qem_t = qem_time_estimate(two_q, p_phys=1e-3)  # referencia fija para reportar junto al mejor caso

            for p_phys in P_PHYS_GRID:
                qem_t_here = qem_time_estimate(two_q, p_phys)
                azure = estimate_with_azure_qre(num_logical_qubits=nq, tcount=tcount, p_phys=p_phys)
                ftqc_qubits = azure["physicalCounts"]["physicalQubits"]
                ftqc_time_s = azure["physicalCounts"]["runtime"] / 1e9

                for budget in QUBIT_BUDGET_GRID:
                    feasible = ftqc_qubits <= budget
                    faster = ftqc_time_s < qem_t_here
                    ratio = (qem_t_here / ftqc_time_s) if ftqc_time_s > 0 else float("inf")
                    clear_advantage = feasible and faster and ratio > 10

                    rows.append({
                        "archetype": name, "scale": scale, "tcount": tcount,
                        "p_phys": p_phys, "qubit_budget": budget,
                        "ftqc_qubits_needed": ftqc_qubits, "ftqc_time_s": ftqc_time_s,
                        "qem_time_s": qem_t_here, "feasible": feasible, "faster": faster,
                        "ratio_qem_over_ftqc": ratio, "clear_ftqc_advantage": clear_advantage,
                    })

    return rows


def find_crossover_points(rows, verbose=True):
    """Para cada arquetipo x escala, el p_fisico MAS GRANDE (peor hardware
    aceptable) y el presupuesto de qubits MAS CHICO (menos qubits necesarios)
    para los que YA hay ventaja FTQC clara -- es decir, el punto MENOS
    exigente en la frontera de factibilidad, no el mejor caso posible."""
    results = {}
    for (archetype, scale) in sorted(set((r["archetype"], r["scale"]) for r in rows),
                                      key=lambda x: (x[0], x[1])):
        subset = [r for r in rows if r["archetype"] == archetype and r["scale"] == scale
                  and r["clear_ftqc_advantage"]]
        if not subset:
            results[(archetype, scale)] = None
            continue
        # el "punto de entrada" menos exigente: el p_fisico mas alto (peor hardware
        # aceptable) entre los que funcionan, y dentro de esos, el presupuesto mas chico
        best = max(subset, key=lambda r: (r["p_phys"], -r["qubit_budget"]))
        results[(archetype, scale)] = best
    return results


if __name__ == "__main__":
    rows = sweep()
    json.dump(rows, open("ftqc_crossover_sweep_results.json", "w"), indent=2)

    crossovers = find_crossover_points(rows)
    print(f"{'='*100}\nPUNTO DE ENTRADA A FTQC: el hardware MENOS exigente (p_fisico mas alto,")
    print(f"presupuesto de qubits mas chico) donde YA hay ventaja FTQC clara (factible + >10x mas rapido)\n{'='*100}")
    for (archetype, scale), best in crossovers.items():
        label = f"{archetype}  (escala x{scale}, T-count={[a for a in ARCHETYPES.values()][0]['tcount']*scale if False else ''})"
        if best is None:
            print(f"\n{archetype} x{scale}: NINGUNA combinacion probada (hasta p_fisico=1e-6, "
                  f"presupuesto=1,000,000 qubits) alcanza ventaja FTQC clara")
        else:
            print(f"\n{archetype} x{scale} (T-count={best['tcount']:,}):")
            print(f"  entra en FTQC con ventaja clara desde: p_fisico <= {best['p_phys']:.0e}  "
                  f"Y  presupuesto >= {best['qubit_budget']:,} qubits fisicos")
            print(f"  en ese punto: qubits_necesarios={best['ftqc_qubits_needed']:,}  "
                  f"tiempo_FTQC={best['ftqc_time_s']:.2e}s  tiempo_QEM={best['qem_time_s']:.2e}s  "
                  f"({best['ratio_qem_over_ftqc']:.0f}x mas rapido)")
