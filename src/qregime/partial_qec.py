"""
partial_qec_sweep.py -- primer intento de llenar la BRECHA que
`regime_decision.py` reporta explicitamente (ni QEM ni FTQC-completo
alcanzan la restriccion, para circuitos como `shor_like_sequential.qasm`)
en vez de solo señalarla.

LA IDEA, explicada simple: el estimador FTQC de este proyecto
(`open_resource_estimator.py`) SOLO sabe resolver "dame la distancia d
minima que hace el error logico casi cero" (`solve_min_distance()`, con
un objetivo de error muy estricto repartido entre tcount compuertas).
Eso es codigo de superficie COMPLETO -- el extremo caro. El otro extremo
que ya existe (QEM) es "sin codigo en absoluto" -- cero overhead, pero
sin proteccion.

Lo que falta -- QEC PARCIAL -- es el punto intermedio: un codigo de
distancia PEQUEÑA que no llega a suprimir el error a niveles de
tolerancia a fallas completa, pero SI reduce el error logico por debajo
del error fisico crudo, a un costo de qubits mucho menor que la
distancia completa. Las piezas para calcularlo YA EXISTEN en el
proyecto -- `physical_qubits_per_logical(d)` y
`predicted_logical_error_rate(d, p_phys, fit)`, ambas ya validadas
contra Stim+PyMatching -- solo nunca se habian usado para BARRER d desde
abajo en vez de resolver para un solo d objetivo.

METODO: para cada d IMPAR de 3 a d_max, calcular:
  costo(d)        = qubits de datos (formula ya validada) + qubits de
                     fabrica de Litinski AL MISMO d (declarado: usar el
                     mismo d para datos y fabrica es una simplificacion;
                     en principio podrian ser distintos, ver limitacion
                     abajo)
  p_logico(d)     = tasa de error logico predicha por NUESTRO ajuste
                     propio (Stim+PyMatching), no un numero de un paper
  confiabilidad(d) = (1 - p_logico(d))^(n*tcount) -- probabilidad de que
                     NINGUNO de los n parches logicos falle durante
                     ninguna de las tcount compuertas T, la misma simplificacion de "presupuesto de
                     error repartido por igual entre tcount compuertas"
                     que ya declara `open_resource_estimator.py`

Se reporta el d de MENOR COSTO (equivalente al d mas pequeño, porque el
costo es monotono creciente en d) que alcanza confiabilidad >= epsilon.
Si NINGUN d en el rango barrido lo logra, se reporta explicitamente que
ni siquiera QEC parcial cierra la brecha con este modelo -- otra vez, no
se fuerza una respuesta.

LIMITACION DECLARADA (importante, la misma honestidad que el resto del
proyecto): esto NO es un modelo de un esquema de QEC parcial real y
publicado (codigos concatenados pequeños, codigos de subsistema, qubits
bosonicos tipo cat/GKP) -- es una EXTRAPOLACION del mismo modelo de
codigo de superficie que ya se uso y valido para FTQC completo, evaluado
a distancias mas pequeñas de las que se disenio para operar. El ajuste
propio (A, p_th) se calibro con Stim+PyMatching en el mismo rango de d
que ya se usa para FTQC completo (ver `fit_own_threshold_model.py`); no
se ha verificado por separado que la formula siga siendo precisa a
distancias pequeñas (d=3,5) fuera de ese rango de calibracion -- se
reporta el resultado igual, pero con esta advertencia explicita, en vez
de aparentar mas certeza de la que hay.
"""
from .resources import (
    physical_qubits_per_logical, predicted_logical_error_rate,
    litinski_15to1_factory_qubits, _load_fit, DEFAULT_NOISE_MODEL,
)


def sweep_partial_qec(num_logical_qubits: int, tcount: int, p_phys: float,
                       reliability_target: float, physical_qubit_budget: int,
                       d_max: int = 51, verbose: bool = True,
                       noise_model: str = DEFAULT_NOISE_MODEL) -> dict:
    fit = _load_fit(noise_model)
    if p_phys >= fit["p_th"]:
        return {"feasible": False, "reason": f"p_fisico ({p_phys}) >= umbral ajustado "
                f"({fit['p_th']:.4f}) -- ningun d ayuda, ni QEC parcial ni completo"}

    rows = []
    chosen = None
    d = 3
    while d <= d_max:
        data_qubits = num_logical_qubits * physical_qubits_per_logical(d)
        factory_qubits = litinski_15to1_factory_qubits(d)
        cost = data_qubits + factory_qubits
        p_logical = predicted_logical_error_rate(d, p_phys, fit)
        # todos los parches fallan independientemente en cada T (ver resources.py)
        reliability = (1 - p_logical) ** (num_logical_qubits * tcount)
        rows.append({"d": d, "cost_qubits": cost, "p_logical_per_gate": p_logical,
                      "reliability": reliability})
        if chosen is None and reliability >= reliability_target and cost <= physical_qubit_budget:
            chosen = rows[-1]
        d += 2

    if verbose:
        print(f"  Barrido de distancia (n={num_logical_qubits}, tcount={tcount}, p_phys={p_phys}):")
        for r in rows[:8]:
            print(f"    d={r['d']:3d}  costo={r['cost_qubits']:>8,}  "
                  f"p_logico/compuerta={r['p_logical_per_gate']:.2e}  "
                  f"confiabilidad~{r['reliability']*100:6.2f}%")
        if len(rows) > 8:
            print(f"    ... ({len(rows)-8} valores de d mas, hasta d={d_max})")

    if chosen:
        result = {"feasible": True, "chosen_d": chosen["d"], "cost_qubits": chosen["cost_qubits"],
                   "reliability": chosen["reliability"],
                   "reduction_vs_full_ftqc_note": "ver estimate_open_ftqc_with_factories() para el d de tolerancia a fallas completa con el mismo circuito"}
        if verbose:
            print(f"  -> QEC PARCIAL FACTIBLE: d={chosen['d']}, costo={chosen['cost_qubits']:,} qubits, "
                  f"confiabilidad~{chosen['reliability']*100:.2f}% (umbral={reliability_target*100:.0f}%, "
                  f"presupuesto={physical_qubit_budget:,})")
    else:
        best = max(rows, key=lambda r: r["reliability"]) if rows else None
        result = {"feasible": False,
                   "reason": "ningun d en el rango barrido alcanza el umbral de confiabilidad dentro del presupuesto",
                   "best_attempt": best}
        if verbose:
            print(f"  -> QEC PARCIAL NO CIERRA LA BRECHA con este modelo. Mejor intento: "
                  f"d={best['d']}, costo={best['cost_qubits']:,}, confiabilidad~{best['reliability']*100:.2f}%")
    return result


_CONFIDENCE_QUARTILE_LABELS = [
    (0.25, "confianza BAJA (cuartil 1, 0-25%) -- no usable como esta, ni siquiera de forma exploratoria"),
    (0.50, "confianza BAJA-MODERADA (cuartil 2, 25-50%) -- exploratorio, no tolerante a fallas"),
    (0.75, "confianza MODERADA (cuartil 3, 50-75%) -- usable con reservas para resultados no-criticos"),
    (1.01, "confianza ALTA (cuartil 4, 75-100%) -- cerca de tolerancia a fallas completa"),
]


def best_feasible_within_hard_budget(num_logical_qubits: int, tcount: int, p_phys: float,
                                      physical_qubit_budget: int, d_max: int = 51,
                                      verbose: bool = True,
                                      noise_model: str = DEFAULT_NOISE_MODEL) -> dict:
    """A diferencia de `sweep_partial_qec` (que exige un umbral de
    confiabilidad Y un presupuesto, y falla si ninguna combinacion
    cumple ambos), esta funcion responde una pregunta distinta y mas
    honesta cuando el presupuesto es duro e innegociable (p.ej. "estos
    son los qubits fisicos que hay, punto"): de TODOS los `d` que
    efectivamente caben en el presupuesto (sin exigir ningun umbral de
    confiabilidad de entrada), ¿cual es el de MEJOR confiabilidad
    posible? Y en vez de declarar binariamente "no alcanza", se reporta
    en que CUARTIL de confianza cae ese mejor punto realizable -- una
    respuesta cuantificada y honesta ("no es tolerante a fallas, pero
    esta en el cuartil X de confianza") en vez de un solo bit de
    pasa/no-pasa."""
    fit = _load_fit(noise_model)
    if p_phys >= fit["p_th"]:
        return {"realizable": False, "reason": f"p_fisico >= umbral ajustado ({fit['p_th']:.4f})"}

    best = None
    d = 3
    while d <= d_max:
        data_qubits = num_logical_qubits * physical_qubits_per_logical(d)
        factory_qubits = litinski_15to1_factory_qubits(d)
        cost = data_qubits + factory_qubits
        if cost > physical_qubit_budget:
            break  # costo monotono creciente en d -- una vez que se pasa, no vuelve a caber
        p_logical = predicted_logical_error_rate(d, p_phys, fit)
        # todos los parches fallan independientemente en cada T (ver resources.py)
        reliability = (1 - p_logical) ** (num_logical_qubits * tcount)
        best = {"d": d, "cost_qubits": cost, "reliability": reliability}
        d += 2

    if best is None:
        result = {"realizable": False,
                   "reason": f"ni siquiera d=3 (el minimo con codigo real) cabe en el presupuesto de "
                             f"{physical_qubit_budget:,} qubits -- el presupuesto es insuficiente para "
                             f"CUALQUIER codigo de superficie sobre {num_logical_qubits} qubits logicos, "
                             f"sin importar la distancia"}
        if verbose:
            print(f"  -> NO REALIZABLE: {result['reason']}")
        return result

    quartile_label = next(lbl for cutoff, lbl in _CONFIDENCE_QUARTILE_LABELS if best["reliability"] < cutoff)
    result = {
        "realizable": True, "chosen_d": best["d"], "cost_qubits": best["cost_qubits"],
        "reliability": best["reliability"], "confidence_quartile": quartile_label,
        "declared_honestly": "esto NO es tolerancia a fallas -- es el mejor punto que el presupuesto "
                              "permite, reportado con su nivel de confianza real, no redondeado hacia arriba",
    }
    if verbose:
        print(f"  Mejor d realizable dentro de {physical_qubit_budget:,} qubits: d={best['d']} "
              f"(costo={best['cost_qubits']:,}), confiabilidad~{best['reliability']*100:.2f}%")
        print(f"  -> {quartile_label}")
    return result


if __name__ == "__main__":
    print("Circuito insignia de este proyecto (la brecha que regime_decision.py reporto sin resolver):\n")
    r = sweep_partial_qec(num_logical_qubits=16, tcount=3930, p_phys=0.0032,
                           reliability_target=0.5, physical_qubit_budget=1121)
    print(f"\nResultado: {r}")
