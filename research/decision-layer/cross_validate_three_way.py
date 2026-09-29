"""
Comparacion de TRES vias, sobre circuitos reales, para el mismo p_fisico:

  1. Azure QRE (pipeline_v2.estimate_with_azure_qre)      -- modelo analitico, caja semi-cerrada (vendor)
  2. Estimador propio (open_resource_estimator)            -- modelo analitico, 100% propio y auditable
  3. Stim + PyMatching (qec_surface_code_experiment)       -- medicion Monte Carlo real, a la distancia
                                                               que CADA estimador eligio

Si las tres coinciden en orden de magnitud, la confianza en el numero final
no depende de creer en ningun vendor en particular.
"""
import json

from qasm_hypergraph import load_qasm_circuit, resource_report
from pipeline_v2 import estimate_with_azure_qre
from open_resource_estimator import estimate_open_ftqc, uncertainty_band
from qec_surface_code_experiment import logical_error_rate


def cross_validate(qasm_path: str, p_phys: float = 1e-3, error_budget: float = 1e-2,
                    num_shots: int = 5000, verbose: bool = True):
    circuit = load_qasm_circuit(qasm_path)
    normalized = circuit.to_basic_gates()
    metrics = resource_report(normalized)
    nq, tcount = metrics["num_qubits"], metrics["tcount"]

    if verbose:
        print(f"\n{'='*78}\n{qasm_path}  (qubits={nq}, tcount={tcount}, p_fisico={p_phys})\n{'='*78}")

    out = {"qasm_path": qasm_path, "num_qubits": nq, "tcount": tcount, "p_phys": p_phys}

    if tcount == 0:
        if verbose:
            print("  tcount=0 -- este circuito no tiene margen no-Clifford, no aplica FTQC/QEC "
                  "(es Clifford puro, simulable clasicamente sin necesidad de codigo de correccion)")
        out["skipped"] = "tcount=0, no aplica"
        return out

    # 1. Azure QRE
    azure = estimate_with_azure_qre(num_logical_qubits=nq, tcount=tcount, p_phys=p_phys,
                                     error_budget=error_budget)
    d_azure = azure["logicalQubit"]["codeDistance"]
    q_azure = azure["physicalCounts"]["physicalQubits"]
    t_azure = azure["physicalCounts"]["runtime"] / 1e9
    if verbose:
        print(f"  [1] Azure QRE        : d={d_azure}  qubits={q_azure:,}  runtime={t_azure:.3e}s")

    # 2. Estimador propio
    own = estimate_open_ftqc(num_logical_qubits=nq, tcount=tcount, p_phys=p_phys,
                              error_budget=error_budget)
    if own["feasible"]:
        d_own = own["codeDistance"]
        q_own = own["physicalQubits"]
        t_own = own["runtime_s"]
        if verbose:
            print(f"  [2] Estimador propio : d={d_own}  qubits={q_own:,}  runtime={t_own:.3e}s")
    else:
        if verbose:
            print(f"  [2] Estimador propio : INFACTIBLE ({own['reason']})")
        d_own = None

    # 3. Medicion real Stim+PyMatching, a AMBAS distancias elegidas (si difieren)
    distances_to_check = sorted(set(d for d in [d_azure, d_own] if d is not None))
    measured = {}
    for d in distances_to_check:
        sim = logical_error_rate(distance=d, rounds=d, p_physical=p_phys,
                                  num_shots=num_shots, seed=hash((qasm_path, d)) % 2**31)
        measured[d] = sim
        if verbose:
            print(f"  [3] Stim+PyMatching a d={d} ({num_shots} shots): "
                  f"p_logico_medido={sim['p_logical']:.2e} +/- {sim['p_logical_stderr']:.2e}")

    # orden de magnitud: comparar qubits y runtime entre [1] y [2]
    ratio_qubits = (q_azure / q_own) if d_own is not None else None
    ratio_runtime = (t_azure / t_own) if d_own is not None else None
    if verbose and d_own is not None:
        print(f"\n  razon Azure/propio -> qubits: {ratio_qubits:.2f}x   runtime: {ratio_runtime:.2f}x")
        same_order = (0.1 <= ratio_qubits <= 10) and (0.1 <= ratio_runtime <= 10)
        print(f"  {'MISMO orden de magnitud' if same_order else 'ORDENES DE MAGNITUD DISTINTOS'} "
              f"entre los dos estimadores analiticos")

    out.update({
        "azure": {"d": d_azure, "qubits": q_azure, "runtime_s": t_azure},
        "own": {"d": d_own, "qubits": own.get("physicalQubits"), "runtime_s": own.get("runtime_s"),
                "feasible": own["feasible"]},
        "measured_at_each_d": {str(k): v for k, v in measured.items()},
        "ratio_qubits_azure_over_own": ratio_qubits,
        "ratio_runtime_azure_over_own": ratio_runtime,
    })
    return out


if __name__ == "__main__":
    circuitos = ["qft10.qasm", "adder8.qasm", "qaoa_farm.qasm", "random_circuit8.qasm"]
    results = []
    for c in circuitos:
        results.append(cross_validate(c, p_phys=1e-3))
    json.dump(results, open("cross_validate_three_way_results.json", "w"), indent=2, default=str)
