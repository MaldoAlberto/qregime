"""
Comparativa: sistema de decision PREVIO (barato, chequeo directo de
fusion de gadgets de fase, SIN reestructurar el diagrama) vs sistema
POSTERIOR (full_reduce completo, que ademas usa "pivoting" para CREAR
fusiones que no existian en el diagrama original).

Motivacion (pregunta del usuario): en vez de aproximar con ML si ZX-calculus
va a ayudar (lo que fallo en generalizar en QAOA/VQE-ansatz, ver Parte 9),
usar directamente el criterio analitico real que la literatura ya describe
(Kissinger-van de Wetering 2019): dos gadgets de fase se fusionan si actuan
sobre el MISMO conjunto de vecinos. PyZX expone esto como un paso propio y
barato (`gadget_simp` / `merge_phase_gadgets_for_apply`), separado del
`full_reduce` completo (que ademas usa `pivot_gadget_simp` para reestructurar
el diagrama y crear fusiones nuevas que no existian de entrada -- la parte
cara/heuristica).

Pregunta que se responde empiricamente: el chequeo barato (¿hay gadgets ya
alineados?) ¿predice bien si vale la pena pagar el costo de full_reduce? ¿O
la reestructuracion por pivoting aporta una fraccion sustancial de la
ganancia total, de forma que un chequeo "previo" subestimaria sistematicamente
el beneficio real?
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import pyzx as zx
from qiskit import QuantumCircuit, transpile, qasm2

from qasm_hypergraph import resource_report

REAL_BASIS = ["cz", "rz", "sx", "x"]


def to_real_basis_pyzx(qc: QuantumCircuit):
    normalized = transpile(qc, basis_gates=REAL_BASIS, optimization_level=1, seed_transpiler=0)
    return zx.Circuit.from_qasm(qasm2.dumps(normalized))


def tcount_of_qasm_str(qasm_str, basis_normalize=True):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    if basis_normalize:
        c = to_real_basis_pyzx(qc)
    else:
        c = zx.Circuit.from_qasm(qasm_str)
    return resource_report(c)["tcount"]


def measure_pre_and_post(original_qasm_str):
    """
    original_qasm_str: el circuito YA en el basis nativo de PyZX (el que se
    le pasa a full_reduce en el resto de la sesion).
    """
    tcount_before = tcount_of_qasm_str(original_qasm_str)
    if tcount_before == 0:
        return None

    # --- sistema PREVIO: solo fusionar gadgets ya alineados, SIN pivoting ---
    t0 = time.perf_counter()
    g_pre = zx.Circuit.from_qasm(original_qasm_str).to_graph()
    zx.simplify.interior_clifford_simp(g_pre)   # limpieza Clifford basica (barata, garantizada)
    zx.simplify.gadget_simp(g_pre)              # SOLO fusion directa de gadgets alineados
    t_pre = time.perf_counter() - t0
    try:
        c_pre = zx.extract_circuit(g_pre.copy())
        tcount_pre = resource_report(zx.Circuit.from_qasm(c_pre.to_qasm()))["tcount"] \
            if False else tcount_of_qasm_str(c_pre.to_qasm(), basis_normalize=True)
    except Exception:
        tcount_pre = tcount_before  # extraccion fallo -> tratar como "sin ganancia utilizable"

    # --- sistema POSTERIOR: full_reduce completo (incluye pivoting) ---
    t0 = time.perf_counter()
    g_post = zx.Circuit.from_qasm(original_qasm_str).to_graph()
    zx.simplify.full_reduce(g_post, quiet=True)
    t_post = time.perf_counter() - t0
    c_post = zx.extract_circuit(g_post.copy())
    tcount_post = tcount_of_qasm_str(c_post.to_qasm(), basis_normalize=True)

    gain_pre = 100.0 * (tcount_before - tcount_pre) / tcount_before
    gain_post = 100.0 * (tcount_before - tcount_post) / tcount_before

    return {
        "tcount_before": tcount_before,
        "tcount_pre": tcount_pre, "tcount_post": tcount_post,
        "gain_pre_pct": gain_pre, "gain_post_pct": gain_post,
        "time_pre_s": t_pre, "time_post_s": t_post,
        "pre_predicts_post_well": abs(gain_pre - gain_post) < 5.0,
    }


if __name__ == "__main__":
    import sys
    from build_circuit_dataset import generate_circuits
    from qiskit import qasm2 as _qasm2

    circuits = generate_circuits()
    results = []
    print(f"{'circuito':26s} {'T antes':>8s} {'ganancia PREVIA':>16s} {'ganancia POSTERIOR':>19s} "
          f"{'t_previo':>9s} {'t_posterior':>11s} {'previo predice bien':>20s}")
    for name, qc in circuits:
        qasm_str = _qasm2.dumps(qc)
        r = measure_pre_and_post(qasm_str)
        if r is None:
            continue
        r["name"] = name
        results.append(r)
        print(f"{name:26s} {r['tcount_before']:8d} {r['gain_pre_pct']:15.1f}% {r['gain_post_pct']:18.1f}% "
              f"{r['time_pre_s']:8.3f}s {r['time_post_s']:10.3f}s {'SI' if r['pre_predicts_post_well'] else 'NO':>20s}")

    with open("pre_vs_post_zx_results.json", "w") as f:
        json.dump(results, f, indent=2)

    n = len(results)
    ok = sum(1 for r in results if r["pre_predicts_post_well"])
    speedup = sum(r["time_post_s"] for r in results) / max(sum(r["time_pre_s"] for r in results), 1e-9)
    print(f"\nN={n}  el chequeo previo predice bien (dif<5pp) en {ok}/{n} ({100*ok/n:.1f}%)")
    print(f"velocidad: el chequeo previo es {speedup:.1f}x mas rapido en promedio que full_reduce")
