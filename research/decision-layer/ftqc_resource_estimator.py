"""
Estimador de recursos FTQC, basado en fórmulas PUBLICADAS y citables -- no en
proxies ilustrativos como en ftqc_advisor.py.

Fuente principal: Litinski, "Magic State Distillation: Not as Costly as You
Think", Quantum 3, 205 (2019). https://arxiv.org/abs/1905.06903

Fórmulas usadas de esa fuente:
  - Tasa de error lógico del código de superficie:
        p_L(p_phys, d) = 0.1 * (100 * p_phys) ** ((d+1)/2)
    (Ecuación citada directamente en el paper, válida para p_phys por debajo
    del umbral físico ~1%.)
  - Costo espacio-tiempo de un CNOT lógico a distancia completa: 3 * d^3
    qubit-ciclos (usado como referencia de comparación en el propio paper).
  - Costo de destilación de un magic state vía protocolo (15-to-1) de un
    solo nivel: 20,000-50,000 qubit-ciclos, alcanzando p_out ~ 1e-9 a 1e-10,
    para p_phys = 1e-4 (valores reportados explícitamente en el paper).
    Usamos 30,000 qubit-ciclos como punto medio de ese rango reportado.

Lo que SÍ es una simplificación nuestra (declarada, no escondida):
  - Asumimos que la destilación corre en paralelo al cómputo principal
    durante aproximadamente el mismo número de ciclos que la profundidad
    del circuito, para convertir el costo espacio-tiempo (qubit-ciclos) en
    un número de qubits físicos dedicados a destilación. Esta es una
    simplificación real de ingeniería de recursos -- un estimador de
    producción (p. ej. el "Azure Quantum Resource Estimator") modela el
    paralelismo de fábricas de magic states con más detalle.
"""

from dataclasses import dataclass


LITINSKI_2019_REF = (
    "Litinski, 'Magic State Distillation: Not as Costly as You Think', "
    "Quantum 3, 205 (2019). arXiv:1905.06903"
)

QUBITCYCLES_PER_MAGIC_STATE_15TO1 = 30_000  # punto medio del rango 20k-50k reportado, p_phys=1e-4


def logical_error_rate(p_phys: float, d: int) -> float:
    """p_L(p_phys, d) = 0.1 * (100 * p_phys) ** ((d+1)/2)  -- Litinski (2019)."""
    return 0.1 * (100 * p_phys) ** ((d + 1) / 2)


def min_code_distance(p_phys: float, num_logical_qubits: int, num_qec_rounds: int,
                       target_total_logical_error: float = 0.01, max_distance: int = 51) -> int:
    """
    Distancia de código mínima (impar) tal que la probabilidad de al menos
    un error lógico en TODO el cómputo (num_logical_qubits * num_qec_rounds
    "ranuras" espacio-tiempo) se mantenga bajo target_total_logical_error.
    """
    for d in range(3, max_distance + 1, 2):
        total_error = num_logical_qubits * num_qec_rounds * logical_error_rate(p_phys, d)
        if total_error <= target_total_logical_error:
            return d
    return max_distance  # no se alcanzó el umbral incluso a la distancia máxima probada


@dataclass
class FTQCResourceReport:
    num_logical_qubits: int
    tcount: int
    total_depth: int
    p_phys: float
    target_total_logical_error: float
    code_distance: int
    data_qubits: int
    distillation_qubitcycles: float
    distillation_qubits_estimate: int
    total_physical_qubits: int
    achieved_logical_error_per_slot: float
    achieved_total_logical_error: float
    active_fraction: float = 1.0
    idle_code_distance: int = None
    data_qubits_active: int = None
    data_qubits_idle: int = None


def estimate_ftqc_resources(num_logical_qubits: int, tcount: int, total_depth: int,
                             p_phys: float = 1e-4,
                             target_total_logical_error: float = 0.01,
                             active_fraction: float = 1.0) -> FTQCResourceReport:
    """
    active_fraction: fracción de los qubits lógicos que están ACTIVOS
    (participando en una compuerta fault-tolerant) en un momento dado, vs.
    INACTIVOS (solo necesitan preservar su estado en memoria, sin ejecutar
    compuertas). Esta distinción es la idea central detrás de los "yoked
    surface codes" de Gidney et al. y de la distancia de código variable de
    Litinski -- los qubits inactivos se protegen a una distancia MENOR
    porque no necesitan sobrevivir a la propagación de errores de una
    compuerta fault-tolerant, solo a la decoherencia por almacenamiento.

    Con active_fraction=1.0 (default), el comportamiento es idéntico al
    estimador original (todos los qubits a distancia completa siempre) --
    la validación contra Gidney (2025) mostró que esto sobreestima el
    conteo de qubits de datos por un factor de ~3.4x para RSA-2048, porque
    la mayoría de los 1,409 qubits lógicos de ese algoritmo están inactivos
    (en memoria) en cualquier instante dado, no todos computando a la vez.

    NOTA DE HONESTIDAD: no reproducimos la fórmula exacta de "yoked surface
    codes" de Gidney (requeriría replicar su arquitectura específica en
    detalle) -- esto es una aproximación estructural: los qubits inactivos
    se protegen a una distancia igual a la mitad de la distancia activa
    (redondeada al impar más cercano, mínimo 3), que es una simplificación
    razonable pero no una reproducción exacta de ningún paper específico.
    """
    d_active = min_code_distance(p_phys, num_logical_qubits, total_depth, target_total_logical_error)

    if active_fraction >= 1.0:
        data_qubits = num_logical_qubits * (2 * d_active - 1) ** 2
        d_idle = None
        active_qubits = num_logical_qubits
        idle_qubits = 0
        data_qubits_active = data_qubits
        data_qubits_idle = 0
    else:
        d_idle = max(3, d_active // 2 if (d_active // 2) % 2 == 1 else d_active // 2 - 1)
        active_qubits = max(1, round(num_logical_qubits * active_fraction))
        idle_qubits = num_logical_qubits - active_qubits
        data_qubits_active = active_qubits * (2 * d_active - 1) ** 2
        data_qubits_idle = idle_qubits * (2 * d_idle - 1) ** 2
        data_qubits = data_qubits_active + data_qubits_idle

    distillation_qubitcycles = tcount * QUBITCYCLES_PER_MAGIC_STATE_15TO1
    distillation_qubits_estimate = int(distillation_qubitcycles / max(total_depth, 1))

    total_physical_qubits = data_qubits + distillation_qubits_estimate

    p_l_slot = logical_error_rate(p_phys, d_active)
    achieved_total_error = num_logical_qubits * total_depth * p_l_slot

    return FTQCResourceReport(
        num_logical_qubits=num_logical_qubits,
        tcount=tcount,
        total_depth=total_depth,
        p_phys=p_phys,
        target_total_logical_error=target_total_logical_error,
        code_distance=d_active,
        data_qubits=data_qubits,
        distillation_qubitcycles=distillation_qubitcycles,
        distillation_qubits_estimate=distillation_qubits_estimate,
        total_physical_qubits=total_physical_qubits,
        achieved_logical_error_per_slot=p_l_slot,
        achieved_total_logical_error=achieved_total_error,
        active_fraction=active_fraction,
        idle_code_distance=d_idle,
        data_qubits_active=data_qubits_active,
        data_qubits_idle=data_qubits_idle,
    )


def print_ftqc_report(report: FTQCResourceReport):
    print(f"Estimación de recursos FTQC  (fórmulas: {LITINSKI_2019_REF})\n")
    print(f"  Entradas:")
    print(f"    qubits lógicos           : {report.num_logical_qubits}")
    print(f"    T-count                  : {report.tcount}")
    print(f"    profundidad (≈ rondas QEC): {report.total_depth}")
    print(f"    tasa de error físico p    : {report.p_phys:.1e}")
    print(f"    error lógico total objetivo: {report.target_total_logical_error:.1e}")
    print()
    print(f"  Resultado:")
    print(f"    distancia de código elegida (d): {report.code_distance}")
    if report.active_fraction < 1.0:
        active_n = round(report.num_logical_qubits * report.active_fraction)
        idle_n = report.num_logical_qubits - active_n
        print(f"    qubits activos (distancia completa d={report.code_distance}): "
              f"{active_n} -> {report.data_qubits_active:,} qubits físicos")
        print(f"    qubits inactivos (distancia reducida d={report.idle_code_distance}): "
              f"{idle_n} -> {report.data_qubits_idle:,} qubits físicos")
    print(f"    qubits físicos de datos (total): {report.data_qubits:,}")
    print(f"    costo de destilación     : {report.distillation_qubitcycles:,.0f} qubit-ciclos "
          f"[= {report.tcount} T-gates × {QUBITCYCLES_PER_MAGIC_STATE_15TO1:,} qubit-ciclos/T]")
    print(f"    qubits físicos de destilación (estimado): {report.distillation_qubits_estimate:,}")
    print(f"    TOTAL qubits físicos estimados: {report.total_physical_qubits:,}")
    print()
    print(f"    error lógico logrado (por qubit-ronda): {report.achieved_logical_error_per_slot:.2e}")
    print(f"    error lógico total logrado  : {report.achieved_total_logical_error:.2e} "
          f"(objetivo: {report.target_total_logical_error:.1e})")

    if report.distillation_qubits_estimate > 3 * report.data_qubits:
        print()
        print("  ADVERTENCIA: la destilación domina el conteo total, lo cual contradice el")
        print("  hallazgo central del propio paper de Litinski (que la destilación suele ser")
        print("  ~2% del costo, no el driver principal). Esto es una señal de que nuestra")
        print("  simplificación -- repartir el costo de destilación sobre solo `total_depth`")
        print("  ciclos -- es demasiado agresiva cuando el T-count es alto relativo a la")
        print("  profundidad (muchos T-gates que 'caben' en pocos ciclos exigen paralelismo")
        print("  extremo de fábricas de destilación). Un estimador real permitiría que la")
        print("  destilación tome más ciclos que total_depth (extendiendo el tiempo total de")
        print("  ejecución en vez de forzar más qubits en paralelo) -- ese trade-off tiempo/")
        print("  espacio es precisamente lo que Litinski optimiza y que aquí NO modelamos.")


if __name__ == "__main__":
    # ejemplo: el sumador de 18 qubits reducido vía ZX (T-count=64, depth=232)
    report = estimate_ftqc_resources(
        num_logical_qubits=18, tcount=64, total_depth=232,
        p_phys=1e-4, target_total_logical_error=0.01,
    )
    print_ftqc_report(report)
