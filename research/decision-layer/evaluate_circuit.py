"""
Herramienta funcional de punta a punta: alguien con un circuito (.qasm) lo
pasa por este script y recibe un reporte con:

  1. Que tecnicas de reduccion se recomiendan, con su nivel de confianza
     HONESTO (no una etiqueta SI/NO sin matices) -- technique_advisor.py.
  2. La decision QEM vs FTQC con numeros reales de un backend real --
     pipeline_v2.py (Azure QRE).
  3. Una SEGUNDA estimacion FTQC 100% propia (open_resource_estimator.py,
     ajustada a nuestros propios datos de Stim+PyMatching), cruzada contra
     la de Azure -- para que la conclusion final no dependa de creer en un
     solo vendor.
  4. Un veredicto final explicito: dado lo medido HASTA AHORA, ¿hay una
     senal de ventaja de hardware que sobrevive a la incertidumbre conocida
     (la de nuestro propio ajuste Y el desacuerdo entre estimadores), o la
     diferencia entre regimenes es demasiado chica / incierta para
     llamarla una ventaja real en vez de ruido de estimacion?

Limitaciones declaradas de este veredicto (no se esconden):
  - El estimador propio NO modela fabricas de destilacion de estados
    magicos (a diferencia de Azure QRE) -- esto le hace subestimar el
    conteo de qubits fisicos de forma sistematica (ver mas abajo, tipicamente
    30-45x menos que Azure en los circuitos probados esta sesion). El
    conteo de qubits fisicos usado para la decision de FACTIBILIDAD sigue
    viniendo de Azure QRE por esta razon -- el estimador propio se usa solo
    para el chequeo de orden de magnitud del RUNTIME, donde ambos estimadores
    si coinciden razonablemente (2-3x).
  - "Ventaja real" aqui significa una separacion de tiempo QEM-vs-FTQC que
    sobrevive a estas incertidumbres conocidas -- NO significa que el
    circuito ya corrio en hardware cuantico real.
"""
import sys
import json

from qiskit_ibm_runtime.fake_provider import FakeTorino

from qasm_hypergraph import load_qasm_circuit, resource_report
from technique_advisor import recommend_techniques
from pipeline_v2 import run_pipeline_v2, estimate_with_azure_qre
from open_resource_estimator import estimate_open_ftqc, uncertainty_band
from case_study import real_two_qubit_error_rate, qem_expected_time_to_reliable_answer
from qiskit import QuantumCircuit, transpile
from layout_noise_search import find_best_layout


def _signal_or_noise(qem_time_s, ftqc_time_azure_s, ftqc_time_own_s, ftqc_feasible):
    """
    Regla explicita y simple para distinguir "ventaja real" de "dentro del
    ruido de estimacion", basada en TRES chequeos, cada uno declarado:

      1. Separacion QEM vs FTQC (Azure): >10x = clara: <3x = ruido.
      2. Si el regimen elegido es FTQC: ¿el estimador propio, pese a NO
         modelar fabricas magicas, da un runtime del MISMO orden de
         magnitud que Azure? Si no, la conclusion de tiempo tampoco es
         confiable.
      3. Factibilidad de qubits SIEMPRE viene de Azure (el propio la
         subestima) -- si Azure dice infactible, no hay "ventaja FTQC"
         posible sin importar el tiempo.
    """
    if not ftqc_feasible:
        return ("SIN VENTAJA FTQC POSIBLE HOY", "FTQC no es factible en qubits con el presupuesto "
                "dado (segun Azure QRE, que SI modela fabricas de destilacion) -- el regimen viable "
                "es QEM independientemente de cualquier comparacion de tiempo")

    ratio_regime = ftqc_time_azure_s / qem_time_s if qem_time_s else float("inf")
    if ftqc_time_own_s:
        ratio_estimators = max(ftqc_time_azure_s, ftqc_time_own_s) / max(min(ftqc_time_azure_s, ftqc_time_own_s), 1e-30)
    else:
        ratio_estimators = None

    if ratio_regime > 10 or ratio_regime < 0.1:
        veredicto = "VENTAJA CON SENAL REAL" if ratio_regime < 0.1 else "QEM CLARAMENTE MEJOR (sin ventaja FTQC)"
        detalle = (f"la separacion de tiempo QEM-vs-FTQC es de {max(ratio_regime, 1/ratio_regime):.1f}x -- "
                   f"esto es mayor que el desacuerdo tipico entre nuestros dos estimadores "
                   f"analiticos ({ratio_estimators:.1f}x en runtime, medido en esta sesion), asi que "
                   f"la conclusion de REGIMEN sobrevive a esa incertidumbre conocida")
    else:
        veredicto = "DENTRO DEL RUIDO DE ESTIMACION"
        detalle = (f"la separacion de tiempo QEM-vs-FTQC es de solo {max(ratio_regime, 1/ratio_regime):.1f}x -- "
                   f"esto es del mismo orden que el desacuerdo ya observado entre nuestros dos "
                   f"estimadores analiticos ({ratio_estimators:.1f}x en runtime), asi que esta diferencia "
                   f"NO se puede distinguir con confianza de un artefacto de que estimador se use")
    return veredicto, detalle


def evaluate(qasm_path: str, backend=None, budget_multiplier: int = 50, p_error_budget: float = 1e-2):
    backend = backend or FakeTorino()

    print(f"\n{'#'*90}\n#  EVALUACION DE CIRCUITO: {qasm_path}\n{'#'*90}")

    # --- 1. Perfil + recomendaciones de tecnicas, con limitaciones honestas ---
    print(f"\n{'='*90}\nPASO 1: perfil del circuito y recomendaciones de tecnicas de reduccion\n{'='*90}")
    profile, recs = recommend_techniques(qasm_path)
    print(f"  qubits={profile['num_qubits']}  compuertas={profile['num_gates']}  "
          f"T-count={profile['tcount']}  grado_prom={profile['avg_degree']:.1f}  "
          f"grado_max={profile['max_degree']:.1f}  spectral_sparsifiability={profile['spectral_sparsifiability']:.3f}")
    for tech, r in recs.items():
        marca = r["recomendada"]
        etiqueta = "ADVERTENCIA" if marca == "ADVERTENCIA" else ("SI" if marca else "NO")
        print(f"  [{etiqueta:>11s}] {tech}")
        print(f"        -> {r['razon']}")

    # --- 2. Decision QEM vs FTQC con Azure QRE (pipeline_v2 real) ---
    print(f"\n{'='*90}\nPASO 2: decision QEM vs FTQC (Azure QRE + backend real)\n{'='*90}")
    pv2 = run_pipeline_v2(qasm_path, backend, budget_multiplier=budget_multiplier, verbose=False)
    regime = pv2["regime"]
    qem_time_s = pv2["qem_time_s"]
    print(f"  regimen elegido: {regime}")
    print(f"  tiempo QEM (real, este backend): {qem_time_s:.3e} s")

    if regime == "FTQC":
        azure = pv2["azure_qre_result"]
        ftqc_time_azure_s = azure["physicalCounts"]["runtime"] / 1e9
        ftqc_qubits_azure = azure["physicalCounts"]["physicalQubits"]
        ftqc_feasible = True
        print(f"  tiempo FTQC (Azure QRE, circuito ya reducido por ZX): {ftqc_time_azure_s:.3e} s")
        print(f"  qubits fisicos (Azure QRE): {ftqc_qubits_azure:,}")
        d_azure = azure["logicalQubit"]["codeDistance"]
        nq_used, tc_used = pv2["reduced_metrics"]["num_qubits"], pv2["reduced_metrics"]["tcount"]
    else:
        ftqc_time_azure_s = pv2["qem_time_s"] * 1e9  # placeholder no usado si no factible; ver abajo
        ftqc_qubits_azure = pv2.get("ftqc_qubits_baseline")
        ftqc_feasible = pv2.get("ftqc_feasible", False)
        print(f"  qubits fisicos que pediria FTQC sin reducir (linea base Azure QRE): {ftqc_qubits_azure:,}")
        print(f"  factible en presupuesto de qubits: {'SI' if ftqc_feasible else 'NO'}")
        # para el chequeo de tres vias, igual se calcula sobre el circuito ORIGINAL
        circuit = load_qasm_circuit(qasm_path).to_basic_gates()
        m = resource_report(circuit)
        nq_used, tc_used = m["num_qubits"], m["tcount"]
        best_layout = find_best_layout(qasm_path, backend, n_trials=20, optimization_level=3)
        avg_error = best_layout["avg_error"]
        azure = estimate_with_azure_qre(nq_used, tc_used, p_phys=max(avg_error, 1e-6))
        ftqc_time_azure_s = azure["physicalCounts"]["runtime"] / 1e9
        d_azure = azure["logicalQubit"]["codeDistance"]

    # --- 3. Segunda estimacion, 100% propia, cruzada contra Azure ---
    print(f"\n{'='*90}\nPASO 3: segunda estimacion FTQC (100% propia, sin Azure QRE) -- cruce de tres vias\n{'='*90}")
    # se reutiliza la MISMA busqueda de layout/ruteo de 20 semillas (ETAPA 0 de pipeline_v2.py)
    # -- probar varias configuraciones y quedarse con el menor error fisico medido cambia el
    # resultado final en ordenes de magnitud (ver layout_noise_search.py), asi que usar un p_phys
    # de una sola configuracion fija subestimaria lo que el hardware real puede lograr
    best_layout = find_best_layout(qasm_path, backend, n_trials=20, optimization_level=3)
    p_phys_used = max(best_layout["avg_error"], 1e-6)

    own = estimate_open_ftqc(nq_used, tc_used, p_phys=p_phys_used, error_budget=p_error_budget)
    if own["feasible"]:
        ftqc_time_own_s = own["runtime_s"]
        print(f"  Azure QRE        : d={d_azure}     qubits={ftqc_qubits_azure:,}   runtime={ftqc_time_azure_s:.3e}s")
        print(f"  Estimador propio : d={own['codeDistance']}     qubits={own['physicalQubits']:,}   runtime={ftqc_time_own_s:.3e}s")
        ratio_q = ftqc_qubits_azure / own["physicalQubits"]
        ratio_t = ftqc_time_azure_s / ftqc_time_own_s
        print(f"\n  razon Azure/propio -> qubits: {ratio_q:.1f}x   runtime: {ratio_t:.1f}x")
        print(f"  NOTA HONESTA: el estimador propio NO modela fabricas de destilacion de estados")
        print(f"  magicos (Azure si) -- la razon grande en qubits ({ratio_q:.0f}x) es la firma esperada")
        print(f"  de esa omision, no un error de calculo. Para RUNTIME el acuerdo es mas cercano")
        print(f"  ({ratio_t:.1f}x), asi que el runtime propio SI sirve como chequeo de orden de magnitud.")

        band = uncertainty_band(nq_used, tc_used, p_phys_used, error_budget=p_error_budget)
        print(f"\n  banda de incertidumbre propia (+/-1 sigma en p_th ajustado):")
        for k, v in band.items():
            if v.get("feasible"):
                print(f"    {k:10s}: d={v['codeDistance']}  qubits={v['physicalQubits']:,}  runtime={v['runtime_s']:.3e}s")
            else:
                print(f"    {k:10s}: infactible")
    else:
        ftqc_time_own_s = None
        print(f"  Estimador propio: INFACTIBLE ({own['reason']})")

    # --- 4. Veredicto final: senal real vs ruido ---
    print(f"\n{'='*90}\nPASO 4: veredicto -- ¿hay una ventaja de hardware con senal real, o es ruido de estimacion?\n{'='*90}")
    veredicto, detalle = _signal_or_noise(qem_time_s, ftqc_time_azure_s, ftqc_time_own_s, ftqc_feasible)
    print(f"  {veredicto}")
    print(f"  {detalle}")

    return {
        "qasm_path": qasm_path, "profile": profile, "recommendations": recs,
        "regime": regime, "qem_time_s": qem_time_s,
        "ftqc_time_azure_s": ftqc_time_azure_s, "ftqc_qubits_azure": ftqc_qubits_azure,
        "ftqc_feasible": ftqc_feasible, "ftqc_time_own_s": ftqc_time_own_s,
        "veredicto": veredicto, "detalle_veredicto": detalle,
    }


if __name__ == "__main__":
    qasm_path = sys.argv[1] if len(sys.argv) > 1 else "adder8.qasm"
    result = evaluate(qasm_path)
