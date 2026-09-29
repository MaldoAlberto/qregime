"""
Mapear el problema que codifica cada circuito QAOA de este proyecto a un
modelo de Ising (MaxCut sobre el mismo grafo aleatorio ya usado en
build_circuit_dataset.py) y buscar su estado fundamental (energia minima)
de tres formas distintas, para comparar honestamente:

  1. EXACTO (fuerza bruta, 2^n asignaciones -- factible para n<=16, que es
     el rango real de este proyecto): el minimo verdadero, sin aproximacion.
  2. Clasico aproximado (simulated annealing, un metodo de optimizacion
     combinatoria clasico estandar, sin nada cuantico).
  3. QAOA p=1 REAL, de dos formas:
     a. Con los parametros FIJOS que ya se usaron al construir el dataset
        (gamma=0.7, beta=0.4, sin optimizar) -- lo que el proyecto ya tenia.
     b. Con los parametros OPTIMIZADOS de verdad (scipy.optimize sobre el
        valor esperado real via Statevector) -- el mejor caso posible de
        QAOA p=1 en estas instancias.

Por que esto importa para la capa de decision: conecta el "lattice +
minima energia" que propuso el usuario con el problema que QAOA ya intenta
resolver por diseno. Si el annealing clasico iguala o supera a QAOA
optimizado en estas instancias (resultado esperado y bien documentado en
la literatura para QAOA p=1 en grafos aleatorios pequenos), eso es una
limitacion honesta que hay que reportar, no una razon para no reportarlo.

Limitacion declarada: fuerza bruta y annealing son sobre las MISMAS
instancias pequenas (n<=16) de este proyecto -- no se prueba en instancias
grandes donde la ventaja cuantica (si existe) tipicamente se buscaria.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import itertools
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def build_graph(n, density):
    return nx.gnp_random_graph(n, density, seed=hash((n, density)) % 1000)


def cut_value(bitstring, edges):
    """bitstring: tupla de 0/1. Cuenta cuantas aristas quedan CORTADAS
    (extremos en particiones distintas) -- el objetivo real de MaxCut."""
    return sum(1 for (u, v) in edges if bitstring[u] != bitstring[v])


def exact_ground_state_from_cutvals(cutvals):
    """Fuerza bruta vectorizada: el maximo sobre las 2^n asignaciones ya
    precomputadas -- exacto, sin aproximacion, solo mas rapido de calcular."""
    return int(np.max(cutvals))


def simulated_annealing_maxcut(n, edges, iters=1500, restarts=4, seed=0):
    """Metodo clasico estandar (sin nada cuantico): flips de un espin a la
    vez, aceptando peores soluciones con probabilidad Boltzmann decreciente
    (temperatura). Real, no un placeholder. Delta de corte calculado solo
    sobre los vecinos del qubit volteado (O(grado), no O(|E|) por paso)."""
    neighbors = {q: [] for q in range(n)}
    for (u, v) in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)

    rng = np.random.default_rng(seed)
    best_overall = -1
    for r in range(restarts):
        state = rng.integers(0, 2, size=n)
        current = sum(1 for (u, v) in edges if state[u] != state[v])
        best = current
        for t in range(iters):
            temp = max(0.01, 1.0 * (1 - t / iters))
            i = rng.integers(0, n)
            before = sum(1 for j in neighbors[i] if state[i] != state[j])
            state[i] ^= 1
            after = sum(1 for j in neighbors[i] if state[i] != state[j])
            delta = after - before
            if delta >= 0 or rng.random() < np.exp(delta / temp):
                current += delta
                best = max(best, current)
            else:
                state[i] ^= 1  # revertir
        best_overall = max(best_overall, best)
    return best_overall


def qaoa_p1_circuit(n, edges, gamma, beta):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for u, v in edges:
        qc.cx(u, v)
        qc.rz(2 * gamma, v)
        qc.cx(u, v)
    for q in range(n):
        qc.rx(2 * beta, q)
    return qc


def precompute_cutvals(n, edges):
    """Vectorizado con numpy (no loop de Python por estado): el valor de
    corte de CADA uno de los 2^n strings clasicos, calculado una sola vez
    por grafo -- esto es lo que hace la evaluacion de <C> barata despues,
    sin cambiar el resultado (sigue siendo el mismo valor esperado exacto)."""
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1)  # shape (2^n, n)
    cutvals = np.zeros(2 ** n, dtype=np.int32)
    for (u, v) in edges:
        cutvals += (bits[:, u] != bits[:, v]).astype(np.int32)
    return cutvals


def expected_cut_value_fast(n, gamma, beta, edges, cutvals):
    """Valor esperado REAL de la funcion de costo (MaxCut) bajo el estado
    QAOA p=1, via Statevector exacto (no shots, sin ruido de muestreo),
    vectorizado con el arreglo de cutvals precomputado."""
    qc = qaoa_p1_circuit(n, edges, gamma, beta)
    probs = Statevector(qc).probabilities()
    return float(np.dot(probs, cutvals))


def optimize_qaoa_p1(n, edges, cutvals, restarts=4, seed=0):
    rng = np.random.default_rng(seed)
    best_val, best_params = -1.0, (0.0, 0.0)
    for _ in range(restarts):
        g0, b0 = rng.uniform(0, np.pi, 2)
        res = minimize(lambda x: -expected_cut_value_fast(n, x[0], x[1], edges, cutvals),
                        x0=[g0, b0], method="Nelder-Mead",
                        options={"xatol": 1e-2, "fatol": 1e-2, "maxiter": 40})
        val = -res.fun
        if val > best_val:
            best_val, best_params = val, tuple(res.x)
    return best_val, best_params


def run():
    configs = [(n, d) for n in [6, 8, 10, 12, 14, 16] for d in [0.2, 0.4, 0.6, 0.8]]
    rows = []
    print(f"{'n':>3s} {'dens':>5s} {'|E|':>4s} {'maxcut_exacto':>14s} {'annealing':>10s} "
          f"{'QAOA_fijo(0.7,0.4)':>19s} {'QAOA_optimizado':>16s} {'ratio_opt/exacto':>17s}")
    for n, density in configs:
        g = build_graph(n, density)
        edges = list(g.edges())
        if len(edges) == 0:
            continue

        cutvals = precompute_cutvals(n, edges)
        exact = exact_ground_state_from_cutvals(cutvals)
        annealing = simulated_annealing_maxcut(n, edges, seed=hash((n, density, "sa")) % 2**31)
        fixed_val = expected_cut_value_fast(n, 0.7, 0.4, edges, cutvals)
        opt_val, opt_params = optimize_qaoa_p1(n, edges, cutvals, seed=hash((n, density, "qaoa")) % 2**31)
        ratio_opt = opt_val / exact if exact > 0 else float("nan")
        ratio_annealing = annealing / exact if exact > 0 else float("nan")

        rows.append({
            "n": n, "density": density, "num_edges": len(edges),
            "maxcut_exact": exact, "annealing_best": annealing,
            "qaoa_fixed_params_expected": fixed_val,
            "qaoa_optimized_expected": opt_val, "qaoa_optimized_params": opt_params,
            "ratio_annealing_over_exact": ratio_annealing,
            "ratio_qaoa_optimized_over_exact": ratio_opt,
        })
        print(f"{n:3d} {density:5.1f} {len(edges):4d} {exact:14d} {annealing:10d} "
              f"{fixed_val:19.2f} {opt_val:16.2f} {ratio_opt:17.3f}")

    with open("qaoa_ising_ground_state_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    n_ok = sum(1 for r in rows)
    avg_ratio_opt = np.mean([r["ratio_qaoa_optimized_over_exact"] for r in rows])
    avg_ratio_ann = np.mean([r["ratio_annealing_over_exact"] for r in rows])
    beats = sum(1 for r in rows if r["ratio_qaoa_optimized_over_exact"] > r["ratio_annealing_over_exact"])
    print(f"\nN={n_ok} instancias")
    print(f"Ratio promedio QAOA p=1 optimizado / optimo exacto: {avg_ratio_opt:.3f}")
    print(f"Ratio promedio simulated annealing / optimo exacto: {avg_ratio_ann:.3f}")
    print(f"QAOA p=1 optimizado supera a annealing clasico en {beats}/{n_ok} instancias")


if __name__ == "__main__":
    run()
