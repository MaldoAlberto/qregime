"""
Extension de qaoa_ising_ground_state_experiment.py a profundidad p=1,2,3:
¿el patron de "QAOA nunca supera al annealing clasico" (visto en p=1, 24/24
instancias) se mantiene, o QAOA empieza a acercarse/superar al annealing
clasico al aumentar la profundidad?

Se usa un subconjunto mas chico de grafos (9 en vez de 24) para mantener
el costo de optimizacion acotado: mas profundidad = mas parametros (2p) =
optimizacion mas cara sobre el mismo Statevector exacto.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qaoa_ising_ground_state_experiment import (
    build_graph, precompute_cutvals, exact_ground_state_from_cutvals,
    simulated_annealing_maxcut,
)


def qaoa_p_circuit(n, edges, params):
    """params: lista de (gamma, beta), una por capa -- p = len(params)."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for gamma, beta in params:
        for u, v in edges:
            qc.cx(u, v)
            qc.rz(2 * gamma, v)
            qc.cx(u, v)
        for q in range(n):
            qc.rx(2 * beta, q)
    return qc


def expected_cut_value_p(n, params, edges, cutvals):
    qc = qaoa_p_circuit(n, edges, params)
    probs = Statevector(qc).probabilities()
    return float(np.dot(probs, cutvals))


def optimize_qaoa_p(n, edges, cutvals, p, restarts=5, seed=0):
    rng = np.random.default_rng(seed)
    best_val, best_params = -1.0, None
    for _ in range(restarts):
        x0 = rng.uniform(0, np.pi, 2 * p)
        res = minimize(
            lambda x: -expected_cut_value_p(n, [(x[2*i], x[2*i+1]) for i in range(p)], edges, cutvals),
            x0=x0, method="Nelder-Mead",
            options={"xatol": 1e-2, "fatol": 1e-2, "maxiter": 60 * p})
        val = -res.fun
        if val > best_val:
            best_val, best_params = val, res.x.tolist()
    return best_val, best_params


def run():
    configs = [(n, d) for n in [6, 10, 14] for d in [0.2, 0.5, 0.8]]
    depths = [1, 2, 3]
    rows = []

    print(f"{'n':>3s} {'dens':>5s} {'p':>2s} {'exacto':>7s} {'annealing':>10s} "
          f"{'QAOA_opt':>10s} {'ratio_qaoa':>11s}")
    for n, density in configs:
        g = build_graph(n, density)
        edges = list(g.edges())
        if len(edges) == 0:
            continue
        cutvals = precompute_cutvals(n, edges)
        exact = exact_ground_state_from_cutvals(cutvals)
        annealing = simulated_annealing_maxcut(n, edges, seed=hash((n, density, "sa")) % 2**31)

        for p in depths:
            opt_val, opt_params = optimize_qaoa_p(n, edges, cutvals, p,
                                                   restarts=5, seed=hash((n, density, p)) % 2**31)
            ratio = opt_val / exact if exact > 0 else float("nan")
            rows.append({"n": n, "density": density, "p": p, "maxcut_exact": exact,
                         "annealing_best": annealing, "qaoa_p_optimized_expected": opt_val,
                         "ratio_qaoa_over_exact": ratio,
                         "ratio_annealing_over_exact": annealing / exact if exact > 0 else float("nan")})
            print(f"{n:3d} {density:5.1f} {p:2d} {exact:7d} {annealing:10d} {opt_val:10.2f} {ratio:11.3f}")

    with open("qaoa_depth_sweep_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    for p in depths:
        sub = [r for r in rows if r["p"] == p]
        avg_ratio = np.mean([r["ratio_qaoa_over_exact"] for r in sub])
        beats = sum(1 for r in sub if r["ratio_qaoa_over_exact"] > r["ratio_annealing_over_exact"])
        print(f"\np={p}: ratio promedio QAOA/exacto = {avg_ratio:.3f}, "
              f"supera a annealing clasico en {beats}/{len(sub)} instancias")

    return rows


if __name__ == "__main__":
    run()
