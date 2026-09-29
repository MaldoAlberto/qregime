"""
QEC real (no estimacion de recursos): simulacion de un codigo de superficie
rotado con circuito real (Stim) y decodificador real (PyMatching, Minimum
Weight Perfect Matching sobre el grafo de deteccion), midiendo la tasa de
error LOGICO real via Monte Carlo, barriendo distancia de codigo (d) y tasa
de error fisico (p).

Por que esto cierra una brecha declarada del proyecto: hasta ahora, todo lo
relacionado a "FTQC" en esta sesion pasaba por Azure QRE como estimador de
recursos (qubits fisicos, runtime) a partir de un modelo de error fisico
asumido -- nunca se generaba ni se decodificaba un codigo de correccion de
errores real. Aqui SI se construye el circuito de sindromes (Stim), se
inyecta ruido de circuito realista (depolarizing + medicion con error), y
se decodifica con un algoritmo real (MWPM via PyMatching) para medir,
experimentalmente, la tasa de error logico -- no asumida de una tabla, sino
observada en simulacion.

Limitacion declarada: esto es simulacion clasica de un codigo estabilizador
(Stim es un simulador de circuitos Clifford, EXACTO para este tipo de
circuito, no una aproximacion) -- no es ejecutar en hardware real, y el
modelo de ruido es depolarizado uniforme, no el perfil de ruido especifico
de un backend real (a diferencia de pipeline_v2.py, que si usa error rates
reales de FakeTorino para la parte QEM). Combinar ambos (ruido real de
backend + decodificador real) queda como extension futura, no se hace aqui.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import stim
import pymatching


def build_surface_code_circuit(distance: int, rounds: int, p_physical: float):
    """
    Codigo de superficie rotado (memoria Z), usando el generador de tareas
    de Stim (stim.Circuit.generated), que construye el circuito de
    sindromes REAL (compuertas CX/CZ/H de un round de estabilizadores +
    mediciones), no una aproximacion analitica.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_physical,
        before_round_data_depolarization=p_physical,
        before_measure_flip_probability=p_physical,
        after_reset_flip_probability=p_physical,
    )
    return circuit


def logical_error_rate(distance: int, rounds: int, p_physical: float,
                        num_shots: int = 20000, seed: int = 0):
    """
    Monte Carlo real: muestrea `num_shots` corridas del circuito de
    sindromes con ruido, decodifica cada una con PyMatching (MWPM sobre el
    grafo de deteccion que Stim deriva automaticamente del circuito), y
    cuenta cuantas veces el decodificador se equivoca sobre el valor
    logico real -- esa fraccion ES la tasa de error logico medida.
    """
    circuit = build_surface_code_circuit(distance, rounds, p_physical)

    # el grafo de deteccion (que combinaciones de detectores corresponden
    # a que error, y su peso) se deriva EXACTAMENTE del circuito -- no se
    # aproxima a mano
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    predictions = matcher.decode_batch(detection_events)
    actual = observable_flips.reshape(-1) if observable_flips.ndim > 1 else observable_flips
    predicted = predictions.reshape(-1) if predictions.ndim > 1 else predictions

    num_errors = int(np.sum(predicted != actual))
    p_logical = num_errors / num_shots

    # error estandar binomial, para saber si la diferencia entre puntos es
    # real o ruido estadistico del propio Monte Carlo
    se = np.sqrt(p_logical * (1 - p_logical) / num_shots) if 0 < p_logical < 1 else 1.0 / num_shots

    return {
        "distance": distance, "rounds": rounds, "p_physical": p_physical,
        "num_shots": num_shots, "num_logical_errors": num_errors,
        "p_logical": p_logical, "p_logical_stderr": se,
    }


def run_threshold_sweep():
    """
    Barrido real: para cada distancia de codigo, la tasa de error logico
    debe CAER al aumentar d cuando p_fisico esta por debajo del umbral del
    codigo, y SUBIR (o quedar plana) por encima del umbral -- esta es la
    firma experimental clasica de un codigo de correccion de errores
    funcionando, y es lo que se mide aqui, no se asume.
    """
    distances = [3, 5, 7]
    p_values = [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05, 0.08, 0.10]
    results = []

    print(f"{'d':>3s} {'p_fisico':>10s} {'p_logico':>12s} {'stderr':>10s} {'errores/shots':>16s}")
    for d in distances:
        rounds = d  # convencion estandar: rounds = distance para memoria
        for p in p_values:
            r = logical_error_rate(d, rounds, p, num_shots=20000, seed=hash((d, p)) % (2**31))
            results.append(r)
            print(f"{d:3d} {p:10.4f} {r['p_logical']:12.5f} {r['p_logical_stderr']:10.5f} "
                  f"{r['num_logical_errors']:6d}/{r['num_shots']:6d}")

    with open("qec_surface_code_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # deteccion honesta del cruce (umbral aproximado): el p donde d=7 deja
    # de tener MENOR p_logico que d=3 (si nunca cruza en el rango barrido,
    # se declara explicitamente en vez de inventar un numero)
    by_p = {}
    for r in results:
        by_p.setdefault(r["p_physical"], {})[r["distance"]] = r["p_logical"]

    crossover_found = None
    for p in sorted(by_p.keys()):
        row = by_p[p]
        if 3 in row and 7 in row and row[7] >= row[3]:
            crossover_found = p
            break

    print(f"\n{len(results)} puntos simulados y guardados en qec_surface_code_results.json")
    if crossover_found is not None:
        print(f"Umbral aproximado observado: por encima de p~{crossover_found:.3f}, "
              f"aumentar distancia (d=7) deja de reducir el error logico frente a d=3.")
    else:
        print("En el rango de p barrido (0.001-0.10), aumentar distancia SIEMPRE redujo "
              "el error logico -- el umbral real del codigo esta fuera de este rango o "
              "no se alcanzo suficiente resolucion estadistica para verlo con 20000 shots.")

    return results, crossover_found


if __name__ == "__main__":
    run_threshold_sweep()
