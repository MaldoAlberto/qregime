"""
Estimador de recursos FTQC 100% propio y auditable -- SIN llamar a Azure QRE
ni a ningun estimador de NVIDIA/cuQuantum. Cada constante usada viene de:

  (a) la formula publicada del umbral de codigos de superficie (Fowler et
      al. 2012): p_L(d) = A * (p_fisico/p_th)^((d+1)/2) -- la FORMA es
      publica, pero A y p_th se ajustaron a NUESTROS 27 puntos medidos con
      Stim+PyMatching (ver fit_own_threshold_model.py), no se copiaron de
      ningun paper ni de ningun vendor.
  (b) el conteo EXACTO de qubits fisicos por qubit logico que nuestra
      PROPIA simulacion ya usa (extraido empiricamente de
      stim.Circuit.generated("surface_code:rotated_memory_z", distance=d)):
      num_qubits(d) = 2*d^2 + 3*d - 1.
  (c) un modelo de tiempo de ejecucion simplificado y declarado como tal:
      cada compuerta T se asume consumida secuencialmente en d ciclos del
      codigo (regla de bolsillo estandar en la literatura de lattice
      surgery), sin optimizar el paralelismo de fabricas de estados magicos
      (a diferencia de Litinski 2019, que si lo optimiza) -- esto da una
      cota SUPERIOR (mas lenta) del tiempo real, declarada como tal, nunca
      presentada como una prediccion mas precisa que Azure QRE.

Proposito: tener un segundo estimador, completamente independiente e
inspeccionable, para cruzar contra Azure QRE (pipeline_v2.py) y contra la
medicion Monte Carlo real (qec_surface_code_experiment.py) -- si los tres
coinciden en orden de magnitud, la confianza no depende de creer en ningun
vendor.
"""
import json
import math
import os

_FIT_PATH = os.path.join(os.path.dirname(__file__), "data", "threshold_fit_result.json")


def _load_fit():
    if not os.path.exists(_FIT_PATH):
        raise RuntimeError(
            "threshold_fit_result.json no existe -- correr primero "
            "fit_own_threshold_model.py para generar A y p_th desde nuestros "
            "propios datos de Stim+PyMatching."
        )
    with open(_FIT_PATH) as f:
        return json.load(f)


def logical_time_steps(tcount: int) -> int:
    """Logical time steps the circuit occupies: one per T gate, and at least
    one, so a Clifford-only circuit is still protected for its duration."""
    return max(int(tcount), 1)


def circuit_failure_reliability(p_logical: float, num_logical_qubits: int,
                                tcount: int) -> float:
    """Probability that none of the num_logical_qubits patches fails in any
    of the circuit's logical time steps, given the per-patch, per-step
    logical error rate p_logical."""
    return (1 - p_logical) ** (num_logical_qubits * logical_time_steps(tcount))


def rotation_synthesis_tcount(rotation_count: int, synthesis_error_budget: float) -> int:
    """T gates needed to synthesise `rotation_count` arbitrary-angle
    rotations in Clifford+T, sharing `synthesis_error_budget` equally.

    Uses the Ross-Selinger (2016, arXiv:1403.2975) scaling for single-qubit
    z-rotations, T ~= 3 * log2(1/eps) per rotation. It is an asymptotic
    estimate of a well-known algorithm, not an exact gridsynth run; it
    replaces the old behaviour of counting every arbitrary rotation as ONE
    T gate (audit finding #4), which understated the FTQC cost of
    rotation-heavy circuits (VQE, QAOA, Trotterised dynamics) by 1-2 orders
    of magnitude."""
    if rotation_count <= 0:
        return 0
    if not 0 < synthesis_error_budget < 1:
        raise ValueError("synthesis_error_budget must be in (0, 1)")
    eps = synthesis_error_budget / rotation_count
    return rotation_count * math.ceil(3 * math.log2(1 / eps))


def effective_tcount(tcount_exact: int, rotation_count: int,
                     error_budget: float) -> tuple:
    """(T gates the FTQC run consumes, error budget left for logical errors).

    When the circuit has arbitrary rotations, half of `error_budget` goes to
    their synthesis and half to logical errors, so the overall failure
    probability still stays within `error_budget`. Without rotations the
    whole budget goes to logical errors, as before."""
    if rotation_count <= 0 or error_budget <= 0:
        return int(tcount_exact), error_budget
    synth = error_budget / 2
    return int(tcount_exact) + rotation_synthesis_tcount(rotation_count, synth), error_budget - synth


def physical_qubits_per_logical(d: int) -> int:
    """Formula EMPIRICA (no de un paper): exactamente lo que
    stim.Circuit.generated('surface_code:rotated_memory_z', distance=d)
    usa, verificado para d en {3,5,7,9,11} contra stim.Circuit.num_qubits."""
    return 2 * d * d + 3 * d - 1


def predicted_logical_error_rate(d: int, p_phys: float, fit: dict) -> float:
    A, p_th = fit["A"], fit["p_th"]
    return A * (p_phys / p_th) ** ((d + 1) / 2.0)


def solve_min_distance(p_phys: float, p_logical_target: float, fit: dict,
                        d_max: int = 51) -> int:
    """Menor distancia IMPAR d>=3 tal que la tasa de error logico predicha
    (por NUESTRO modelo ajustado) sea <= el objetivo. Si p_phys >= p_th,
    ningun d ayuda (por encima del umbral, mas distancia empeora, no
    mejora) -- se reporta explicitamente como infactible, no se fuerza un
    numero."""
    if p_phys >= fit["p_th"]:
        return None  # por encima del umbral: FTQC no es viable con este p_fisico
    d = 3
    while d <= d_max:
        if predicted_logical_error_rate(d, p_phys, fit) <= p_logical_target:
            return d
        d += 2
    return None  # no se alcanzo el objetivo ni con d_max


def estimate_open_ftqc(num_logical_qubits: int, tcount: int, p_phys: float,
                        error_budget: float = 1e-2,
                        two_qubit_gate_time_s: float = 50e-9,
                        rounds_per_tgate: int = None,
                        verbose: bool = False) -> dict:
    """
    Estimador de recursos FTQC completo, sin Azure QRE.

    Parametros identicos en espiritu a estimate_with_azure_qre() de
    pipeline_v2.py, para poder llamarse en paralelo con los mismos
    argumentos y comparar resultados directamente.

    NOTA (esta sesion): `num_logical_qubits` aqui puede pasarse como el
    MAXIMO DE QUBITS CONCURRENTES (ver `technique_advisor.profile_circuit`,
    campo `max_concurrent_qubits`) en vez del conteo total declarado, para
    modelar reuso temporal de parches via medicion+reset -- a diferencia
    de Azure QRE (ver `open_resource_estimator_reuse_demo` mas abajo),
    ESTE estimador SI soporta esa sustitucion de forma valida, porque
    `total_physical_qubits = num_logical_qubits * physical_qubits_por_
    logico` es una formula LINEAL y SEPARADA del calculo de runtime (que
    depende solo de tcount) -- no hay ningun presupuesto de error
    compartido entre "qubits" y "fabricas" que se reoptimice de forma no
    lineal al cambiar este numero, porque este estimador (a diferencia de
    Azure QRE) NO modela fabricas de destilacion en absoluto (gap ya
    declarado). Esto SI es una forma correcta y honesta de cuantificar el
    ahorro de reuso -- limitada, eso si, a la parte que este estimador
    modela (qubits de datos), no a la parte que le falta (fabricas).
    """
    fit = _load_fit()

    if not 0 < error_budget < 1:
        return {"feasible": False, "fit_used": fit,
                "reason": f"error_budget must be in (0, 1), got {error_budget}: "
                          "a reliability target of 100% cannot be met by any finite code"}
    if tcount < 0:
        raise ValueError("tcount must be >= 0")
    if num_logical_qubits < 1:
        raise ValueError("num_logical_qubits must be >= 1")

    # Each T gate is one logical time step of `rounds_per_tgate` (= d)
    # rounds. A Clifford-only circuit (tcount == 0) still occupies at least
    # one step, so it gets the minimum distance that protects its memory
    # instead of crashing (audit finding #5).
    logical_steps = logical_time_steps(tcount)

    # The fitted p_L(d) comes from Stim memory experiments with rounds = d,
    # i.e. it is the failure probability of ONE patch over ONE logical step.
    # During each step every one of the num_logical_qubits patches runs QEC,
    # so the circuit fails if any patch fails at any step: the budget has to
    # be shared among num_logical_qubits * logical_steps patch-steps, not
    # only among the T gates (audit finding #1; the previous version divided
    # by tcount alone and overstated the reliability of its own answer).
    p_logical_target_per_gate = error_budget / (num_logical_qubits * logical_steps)

    d = solve_min_distance(p_phys, p_logical_target_per_gate, fit)
    infeasible_reason = None
    if d is None:
        if p_phys >= fit["p_th"]:
            infeasible_reason = (f"p_fisico ({p_phys*100:.3f}%) esta en o por encima del "
                                  f"umbral ajustado ({fit['p_th']*100:.3f}%) -- ningun codigo "
                                  f"de superficie de este tipo ayuda a esta tasa de error fisico")
        else:
            infeasible_reason = "no se alcanzo el objetivo de error logico ni con d=51"
        return {"feasible": False, "reason": infeasible_reason, "fit_used": fit}

    if rounds_per_tgate is None:
        rounds_per_tgate = d  # regla de bolsillo estandar (lattice surgery), declarada arriba

    physical_qubits_per_lq = physical_qubits_per_logical(d)
    total_physical_qubits = num_logical_qubits * physical_qubits_per_lq

    cycle_time_s = 6 * two_qubit_gate_time_s  # 6 pasos por ronda de estabilizadores (estandar)
    total_rounds = logical_steps * rounds_per_tgate
    runtime_s = total_rounds * cycle_time_s

    achieved_p_logical = predicted_logical_error_rate(d, p_phys, fit)

    result = {
        "feasible": True,
        "codeDistance": d,
        "physicalQubitsPerLogicalQubit": physical_qubits_per_lq,
        "physicalQubits": total_physical_qubits,
        "runtime_s": runtime_s,
        "achieved_p_logical_per_gate": achieved_p_logical,
        "p_logical_target_per_gate": p_logical_target_per_gate,
        "cycle_time_s": cycle_time_s,
        "total_qec_rounds": total_rounds,
        "fit_used": {"A": fit["A"], "p_th": fit["p_th"], "p_th_stderr": fit["p_th_stderr"]},
        "assumptions": [
            "compuertas T consumidas secuencialmente (sin paralelizar fabricas de estados magicos, "
            "a diferencia de Litinski 2019 -- esto da una cota SUPERIOR del tiempo real, no una estimacion optimizada)",
            "error budget shared equally among num_logical_qubits x max(tcount, 1) "
            "patch-steps (each patch runs d QEC rounds per T gate)",
            f"tiempo de ciclo = 6 x tiempo de compuerta de 2 qubits ({two_qubit_gate_time_s*1e9:.0f} ns)",
        ],
    }
    if verbose:
        print(f"  [estimador propio] d={d}  qubits_fisicos={total_physical_qubits:,}  "
              f"runtime={runtime_s:.3e}s  p_logico_por_compuerta={achieved_p_logical:.2e}")
    return result


def litinski_15to1_factory_qubits(d: int) -> int:
    """
    Costo de UNA fabrica de destilacion de magic states 15-a-1, tomado
    LITERALMENTE de Litinski 2019 ("A Game of Surface Codes: Large-Scale
    Quantum Computing with Lattice Surgery", arXiv:1808.02892), NO
    reajustado ni reestimado por nosotros -- a diferencia de
    physical_qubits_per_logical(), que si viene de nuestra propia
    simulacion Stim.

    Citas textuales del paper (verificadas via fetch directo del PDF,
    Fig. 17c y Seccion 4.1):
      - "the 15-to-1 protocol of Fig. 15 is implemented in 11*Delta using
        11 tiles"
      - "each tile corresponds to d^2 physical data qubits" en general,
        pero el ejemplo numerico concreto de la Seccion 4.1 usa
        2*d^2 qubits por tile (routing/ancilla incluidos) y confirma:
        d=13 -> "11 tiles x 2d^2 = 11 x 338 = 3,718 physical data qubits"
        (11*2*13**2 = 3718, verificado aritmeticamente).

    Por lo tanto: qubits_por_fabrica(d) = 11 * 2 * d^2 = 22*d^2.

    NOTA: esto NO es un ajuste propio (a diferencia del resto de este
    modulo) -- es la formula del paper, citada, no derivada de nuestros
    datos. Se usa aqui solo para dar una PRIMERA cota (no optimizada) del
    costo de fabricas, ausente hasta ahora en este estimador.
    """
    return 11 * 2 * d * d


def estimate_open_ftqc_with_factories(num_logical_qubits: int, tcount: int,
                                       p_phys: float, error_budget: float = 1e-2,
                                       two_qubit_gate_time_s: float = 50e-9,
                                       rounds_per_tgate: int = None,
                                       num_factories: int = 1,
                                       verbose: bool = False) -> dict:
    """
    Extension de estimate_open_ftqc() que SUMA el costo de fabricas de
    destilacion 15-a-1 (formula citada de Litinski 2019, ver
    litinski_15to1_factory_qubits arriba) al costo de qubits de datos ya
    calculado por estimate_open_ftqc().

    Limitacion declarada, a proposito: el modelo de TIEMPO no cambia
    respecto a estimate_open_ftqc() (sigue asumiendo consumo secuencial
    de compuertas T, sin paralelizar fabricas -- ver docstring de esa
    funcion). Por eso aqui se usa num_factories=1 por default: es la
    UNICA opcion consistente con "sin paralelizar fabricas" que ya estaba
    declarada. Usar num_factories>1 aqui SIN tambien corregir el modelo de
    tiempo (que no hacemos) subestimaria el numero de fabricas realmente
    necesarias para ese num_factories de forma inconsistente -- por eso
    esa combinacion no se expone como un caso "recomendado", solo como
    una perilla auditable para quien quiera explorarla sabiendo esta
    limitacion.

    Motivacion (por que esto importa para el reuso de qubits): a
    diferencia de physical_qubits_per_logical(d)*num_logical_qubits (que
    SI se reduce si num_logical_qubits se sustituye por
    max_concurrent_qubits), el costo de fabricas NO depende de
    num_logical_qubits en absoluto -- es un costo FIJO una vez fijado d.
    Esto significa que el ahorro porcentual de reusar qubits, medido
    sobre el total (datos + fabricas), es necesariamente MENOR que el
    ahorro medido solo sobre qubits de datos -- se cuantifica exactamente
    mas abajo con shor_like_sequential.qasm.
    """
    base = estimate_open_ftqc(num_logical_qubits, tcount, p_phys, error_budget,
                               two_qubit_gate_time_s, rounds_per_tgate, verbose=False)
    if not base["feasible"]:
        return base

    d = base["codeDistance"]
    qubits_per_factory = litinski_15to1_factory_qubits(d)
    factory_qubits_total = num_factories * qubits_per_factory

    result = dict(base)
    result["codeDistanceUsedForFactory"] = d
    result["qubitsPerFactory_litinski2019"] = qubits_per_factory
    result["numFactories"] = num_factories
    result["physicalQubitsForFactories"] = factory_qubits_total
    result["physicalQubitsForData"] = base["physicalQubits"]
    result["physicalQubits"] = base["physicalQubits"] + factory_qubits_total
    result["assumptions"] = base["assumptions"] + [
        "costo de fabricas = num_factories * 11 * 2 * d^2 qubits, formula "
        "CITADA de Litinski 2019 (Fig. 17c / Sec. 4.1), no reajustada por "
        "nosotros; num_factories=1 por default, consistente con el supuesto "
        "de no paralelizar fabricas ya declarado en el modelo de tiempo.",
        "el costo de fabricas NO depende de num_logical_qubits -- es "
        "constante una vez fijado d, a diferencia del costo de qubits de "
        "datos (que si se beneficia de la sustitucion por "
        "max_concurrent_qubits para modelar reuso temporal)",
    ]
    if verbose:
        print(f"  [con fabricas] d={d}  qubits_datos={base['physicalQubits']:,}  "
              f"qubits_fabricas({num_factories}x)={factory_qubits_total:,}  "
              f"TOTAL={result['physicalQubits']:,}")
    return result


def uncertainty_band(num_logical_qubits: int, tcount: int, p_phys: float,
                      error_budget: float = 1e-2, n_sigma: float = 1.0) -> dict:
    """
    Repite la estimacion con p_th +/- n_sigma * p_th_stderr (la incertidumbre
    REAL del ajuste a nuestros propios datos, no un numero inventado) para
    dar un rango de runtime/qubits, no un solo numero puntual -- esto es lo
    que permite despues distinguir una ventaja de hardware real de una
    diferencia que podria explicarse por la incertidumbre del propio ajuste.
    """
    fit = _load_fit()
    band = {}
    for label, p_th_shift in [("central", 0.0), ("optimista", -n_sigma), ("pesimista", +n_sigma)]:
        shifted_fit = dict(fit)
        shifted_fit["p_th"] = fit["p_th"] + p_th_shift * fit["p_th_stderr"]
        steps = logical_time_steps(tcount)
        d = solve_min_distance(p_phys, error_budget / (num_logical_qubits * steps), shifted_fit)
        if d is None:
            band[label] = {"feasible": False}
            continue
        pq = num_logical_qubits * physical_qubits_per_logical(d)
        rounds = steps * d
        runtime = rounds * 6 * 50e-9
        band[label] = {"feasible": True, "codeDistance": d, "physicalQubits": pq, "runtime_s": runtime}
    return band


if __name__ == "__main__":
    fit = _load_fit()
    print(f"Modelo ajustado: A={fit['A']:.4f}  p_th={fit['p_th']*100:.3f}% +/- {fit['p_th_stderr']*100:.3f}pp\n")

    casos = [
        ("QFT10 (ejemplo ilustrativo)", 10, 168, 1e-3),
        ("Sumador8 (ejemplo ilustrativo)", 9, 60, 1e-3),
    ]
    for label, nq, tc, p in casos:
        print(f"{label}: n_qubits_logicos={nq}, tcount={tc}, p_fisico={p}")
        r = estimate_open_ftqc(nq, tc, p, verbose=True)
        if r["feasible"]:
            band = uncertainty_band(nq, tc, p)
            print(f"    banda de incertidumbre (+/-1 sigma en p_th del ajuste):")
            for k, v in band.items():
                if v.get("feasible"):
                    print(f"      {k:10s}: d={v['codeDistance']}  qubits={v['physicalQubits']:,}  runtime={v['runtime_s']:.3e}s")
                else:
                    print(f"      {k:10s}: infactible")
        print()
