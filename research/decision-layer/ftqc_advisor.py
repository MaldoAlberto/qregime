"""
Etapa 4 del pipeline: dado el perfil de métricas de un circuito (post-ZX,
post-layout), recomendar una estrategia de protección contra errores.

IMPORTANTE (honestidad): las fórmulas de costo usadas aquí son PROXIES
ILUSTRATIVOS de las leyes de escalamiento cualitativas conocidas en la
literatura -- no son estimaciones de recursos calibradas contra un modelo
de ruido real ni contra un código QEC específico. Sirven para ordenar
relativamente las tres estrategias dado un perfil de circuito, no para
predecir un número de qubits o un tiempo de ejecución real. Cualquier uso
serio de estos números requiere sustituir las constantes por parámetros
calibrados del hardware objetivo (tasa de error físico, código QEC elegido,
protocolo de destilación).

Leyes de escalamiento en las que se basan los proxies (ver reporte de idea
de investigación previo, sección 2.1 y sus referencias):
  - QEM: el overhead de muestreo crece EXPONENCIALMENTE con la profundidad
    del circuito.
  - FTQC / destilación de magic states: el overhead crece aprox.
    LINEALMENTE con el T-count (más precisamente, con el T-count post-
    reducción ZX, que es la cifra real relevante).
  - QEC (código de superficie rotado): el overhead de qubits físicos por
    qubit lógico escala como (2d-1)^2 con la distancia del código d.
"""

from dataclasses import dataclass, field


@dataclass
class CircuitProfile:
    num_logical_qubits: int
    total_depth: int
    two_qubit_depth: int
    tcount: int
    two_qubit_count: int
    ancillas_routing: int = 0


@dataclass
class HardwareBudget:
    available_physical_qubits: int
    # profundidad a partir de la cual el overhead de muestreo de QEM se
    # considera "alto" en este proxy (umbral ilustrativo, no calibrado)
    qem_depth_scale: float = 20.0
    # T-gates por unidad de overhead de destilación en este proxy
    ftqc_tgate_scale: float = 15.0
    # distancias de código de superficie a evaluar
    candidate_code_distances: tuple = (3, 5, 7)


def surface_code_overhead(num_logical_qubits: int, distance: int) -> int:
    """Qubits físicos aprox. para un código de superficie rotado de distancia d."""
    return num_logical_qubits * (2 * distance - 1) ** 2


def qem_cost_proxy(profile: CircuitProfile, budget: HardwareBudget) -> float:
    """Proxy adimensional del overhead de muestreo de QEM (crece exponencial en profundidad)."""
    return 2 ** (profile.total_depth / budget.qem_depth_scale)


def ftqc_cost_proxy(profile: CircuitProfile, budget: HardwareBudget, distance: int) -> float:
    """Proxy adimensional del costo de FTQC completo: overhead de QEC + destilación."""
    qec_overhead = surface_code_overhead(profile.num_logical_qubits, distance) / profile.num_logical_qubits
    distillation_overhead = 1.0 + profile.tcount / budget.ftqc_tgate_scale
    return qec_overhead * distillation_overhead


def hybrid_eft_cost_proxy(profile: CircuitProfile, budget: HardwareBudget, distance: int) -> float:
    """
    Proxy del híbrido EFT: QEC solo para la parte Clifford (barata,
    transversal) + QEM solo para el T-count residual (no para toda la
    profundidad, como en QEM puro) -- refleja la idea central de los
    trabajos EFT citados en el reporte anterior.
    """
    qec_overhead = surface_code_overhead(profile.num_logical_qubits, distance) / profile.num_logical_qubits
    residual_qem_overhead = 2 ** (profile.tcount / budget.qem_depth_scale)
    return qec_overhead * 0.5 + residual_qem_overhead  # QEC transversal ~ mitad de costo que con destilación completa


def recommend_strategy(profile: CircuitProfile, budget: HardwareBudget,
                        qem_cost_override: float = None) -> dict:
    """
    qem_cost_override: si se provee (p.ej. 1/fidelidad_real medida en un
    backend real), reemplaza el proxy ilustrativo de costo de QEM por un
    número informado por datos reales -- ver case_study.py, que calcula
    la fidelidad real de un circuito transpilado a un backend específico
    y la pasa aquí en vez de dejar que el proxy genérico decida a ciegas.
    """
    feasible_distances = [
        d for d in budget.candidate_code_distances
        if surface_code_overhead(profile.num_logical_qubits, d) <= budget.available_physical_qubits
    ]

    if not feasible_distances:
        return {
            "recommendation": "QEM",
            "reason": (
                f"No hay presupuesto de qubits suficiente para NINGÚN código QEC "
                f"candidato (se necesitarían al menos "
                f"{surface_code_overhead(profile.num_logical_qubits, min(budget.candidate_code_distances))} "
                f"qubits físicos para distancia {min(budget.candidate_code_distances)}, "
                f"hay {budget.available_physical_qubits} disponibles). Esta es una "
                f"restricción dura, no una comparación de costos."
            ),
            "costs": None,
        }

    best_distance = min(feasible_distances)  # la distancia más barata que alcanza el presupuesto

    costs = {
        "QEM": qem_cost_override if qem_cost_override is not None else qem_cost_proxy(profile, budget),
        "FTQC": ftqc_cost_proxy(profile, budget, best_distance),
        "Híbrido EFT": hybrid_eft_cost_proxy(profile, budget, best_distance),
    }

    best_strategy = min(costs, key=costs.get)

    reason = (
        f"Con distancia de código factible d={best_distance}, el proxy de costo "
        f"más bajo corresponde a '{best_strategy}'."
    )
    if qem_cost_override is not None:
        reason += " (costo de QEM informado por fidelidad real medida en hardware, no por el proxy genérico)"

    return {
        "recommendation": best_strategy,
        "reason": reason,
        "costs": costs,
        "code_distance_used": best_distance,
        "feasible_distances": feasible_distances,
    }


def print_recommendation(profile: CircuitProfile, budget: HardwareBudget, result: dict):
    print("Perfil del circuito:")
    print(f"  qubits lógicos       : {profile.num_logical_qubits}")
    print(f"  profundidad total    : {profile.total_depth}")
    print(f"  profundidad 2-qubits : {profile.two_qubit_depth}")
    print(f"  T-count              : {profile.tcount}")
    print(f"  compuertas 2-qubits  : {profile.two_qubit_count}")
    print(f"  ancillas de routing  : {profile.ancillas_routing}")
    print()
    print(f"Presupuesto de qubits físicos disponibles: {budget.available_physical_qubits}")
    print()

    if result["costs"] is None:
        print(f"Recomendación: {result['recommendation']}")
        print(f"Razón: {result['reason']}")
        return

    print("Proxies de costo (adimensionales, NO calibrados -- ver docstring del módulo):")
    for strategy, cost in sorted(result["costs"].items(), key=lambda kv: kv[1]):
        marker = " <-- recomendado" if strategy == result["recommendation"] else ""
        print(f"  {strategy:15s}: {cost:8.2f}{marker}")
    print()
    print(f"Recomendación: {result['recommendation']}")
    print(f"Razón: {result['reason']}")


if __name__ == "__main__":
    # ejemplo standalone
    profile = CircuitProfile(
        num_logical_qubits=6, total_depth=86, two_qubit_depth=51,
        tcount=25, two_qubit_count=61, ancillas_routing=4,
    )
    budget = HardwareBudget(available_physical_qubits=500)
    result = recommend_strategy(profile, budget)
    print_recommendation(profile, budget, result)
