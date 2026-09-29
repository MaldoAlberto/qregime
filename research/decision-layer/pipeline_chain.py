"""
Extension propuesta #10 de esta sesion: encadenar las tecnicas en vez de
evaluarlas por separado sobre el circuito completo. Primero se evalua si
circuit cutting real (qiskit-addon-cutting) es factible (overhead de
muestreo por debajo de un presupuesto declarado); si lo es, se particiona
el circuito en subcircuitos REALES (via `partition_problem`, no una
simulacion de la particion) y se corre technique_advisor + pipeline_v2 en
CADA subcircuito por separado, en vez de una sola vez sobre el circuito
entero.

Limitacion declarada y verificada empiricamente: `partition_problem` no
siempre produce mas de 1 fragmento aunque `find_cuts` haya encontrado
varios cortes -- se probo con Sumador8 (structuralmente en cadena, 2 cortes
reportados) y el particionador automatico lo dejo como 1 solo bloque, pero
con QAOA/farm (grafo denso, 46 cortes) SI produjo 3 fragmentos reales (9,
7 y 1 qubits). Esto se reporta tal cual -- no se fuerza una particion que
la libreria no encontro, y el caso de 1 solo fragmento simplemente reduce
al comportamiento actual (analizar el circuito completo), sin romper nada.
"""

import warnings
warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, qasm2
from qiskit_addon_cutting import find_cuts, OptimizationParameters, DeviceConstraints, partition_problem

from technique_advisor import profile_circuit, recommend_techniques


CUTTING_OVERHEAD_BUDGET = 1e6  # presupuesto declarado de sobrecosto de muestreo


def check_cutting_feasibility(qc: QuantumCircuit):
    opt = OptimizationParameters(seed=0)
    qps = max(2, qc.num_qubits // 2 + 1)
    constraints = DeviceConstraints(qubits_per_subcircuit=qps)
    try:
        cut_circuit, metadata = find_cuts(qc, opt, constraints)
    except Exception as e:
        return {"feasible": False, "reason": f"find_cuts fallo: {e}", "cut_circuit": None}

    overhead = metadata.get("sampling_overhead", float("inf"))
    num_cuts = len(metadata.get("cuts", []))
    feasible = overhead < CUTTING_OVERHEAD_BUDGET
    return {"feasible": feasible, "overhead": overhead, "num_cuts": num_cuts,
            "cut_circuit": cut_circuit,
            "reason": f"overhead={overhead:.2e} {'<' if feasible else '>='} presupuesto {CUTTING_OVERHEAD_BUDGET:.0e}"}


def run_chained_pipeline(qasm_path: str, verbose: bool = True):
    qc = QuantumCircuit.from_qasm_file(qasm_path)
    cut_check = check_cutting_feasibility(qc)

    if verbose:
        print(f"{'='*70}\nPaso 1: factibilidad de circuit cutting\n{'='*70}")
        print(f"  {cut_check['reason']}")

    subcircuits = {0: qc}  # por defecto: el circuito completo, sin particion
    if cut_check["feasible"]:
        try:
            partitioned = partition_problem(circuit=cut_check["cut_circuit"])
            if len(partitioned.subcircuits) > 1:
                subcircuits = dict(partitioned.subcircuits)
                if verbose:
                    print(f"  Particion real obtenida: {len(subcircuits)} subcircuitos "
                          f"({', '.join(str(sc.num_qubits) + 'q' for sc in subcircuits.values())})")
            elif verbose:
                print("  find_cuts encontro cortes, pero el particionador automatico "
                      "no separo el circuito en mas de 1 bloque -- se analiza completo")
        except Exception as e:
            if verbose:
                print(f"  partition_problem fallo ({e}) -- se analiza el circuito completo")

    results = {}
    for label, frag in subcircuits.items():
        if frag.num_qubits == 0 or len(frag.data) == 0:
            continue
        frag_qasm_path = f"/tmp/_chain_fragment_{label}.qasm"
        try:
            with open(frag_qasm_path, "w") as f:
                f.write(qasm2.dumps(frag))
        except Exception as e:
            if verbose:
                print(f"  fragmento {label}: no se pudo exportar a QASM ({e}), se omite")
            continue

        if verbose:
            print(f"\n{'='*70}\nPaso 2 -- fragmento {label} ({frag.num_qubits} qubits, {len(frag.data)} compuertas)\n{'='*70}")
        try:
            p, recs = recommend_techniques(frag_qasm_path)
        except Exception as e:
            if verbose:
                print(f"  perfil/recomendacion fallo en este fragmento ({e}) -- se omite")
            continue
        results[label] = {"profile": p, "recommendations": recs}
        if verbose:
            for tech, r in recs.items():
                marca = r["recomendada"]
                print(f"  [{marca!s:>11s}] {tech}")

    return {"cutting_check": {k: v for k, v in cut_check.items() if k != "cut_circuit"},
            "num_fragments_analyzed": len(results), "fragment_results": results}


if __name__ == "__main__":
    import sys
    qasm_path = sys.argv[1] if len(sys.argv) > 1 else "qaoa_farm.qasm"
    run_chained_pipeline(qasm_path)
