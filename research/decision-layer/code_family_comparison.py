"""
Compara nuestro pipeline (que asume codigo de superficie en TODO el
proyecto -- Stim, Azure QRE con qecScheme="surface_code", y nuestro propio
open_resource_estimator.py) contra un dato real, medido en hardware, de
OTRA familia de codigo: el codigo Steane [[7,1,3]] sobre iones atrapados.

Fuente: Perlin, He, Alexiades Armenakas, Andres-Martinez, Hao, Herman, Jin,
Mayer, Self, Amaro, Ryan-Anderson, Shaydulin, "Fault-tolerant execution of
error-corrected quantum algorithms", arXiv:2603.04584 (2026). Ejecutaron
QAOA real (no simulado) en Quantinuum H2/Helios con codigo Steane, llegando
a 12 qubits logicos / 97 qubits fisicos / 2132 compuertas fisicas de 2
qubits, con una compuerta T logica tolerante a fallos de infidelidad
medida ~2.6(4)e-3.

IMPORTANTE, declarado honestamente: Azure QRE (la libreria local que
usamos en este proyecto) SOLO ofrece qecScheme in {"surface_code",
"floquet_code"} -- no tiene una opcion Steane. No se construye aqui un
estimador completo para codigos concatenados tipo Steane (eso requeriria
una curva de supresion de error por nivel de concatenacion, y el paper
solo da UN punto medido, no una curva) -- este script usa ese unico punto
real como ANCLA de comparacion, no como un modelo generalizable.

El T-count de su circuito QAOA de 12 qubits logicos no esta reportado en
el abstract -- se extrapola aqui desde la razon T-count/compuertas-de-2q
medida en NUESTRO propio circuito de referencia QAOA (qaoa_farm.qasm:
tcount=56, 2q=79 -> razon 0.709), aplicada a sus 2132 compuertas de 2
qubits reportadas. Esto se declara como una EXTRAPOLACION, no un dato de
ellos -- el numero real podria ser distinto.
"""
from pipeline_v2 import estimate_with_azure_qre
from open_resource_estimator import estimate_open_ftqc

# --- dato real, citado, no inventado ---
STEANE_DEMO = {
    "source": "Perlin et al., arXiv:2603.04584 (2026)",
    "code": "[[7,1,3]] Steane, iones atrapados (Quantinuum H2/Helios)",
    "logical_qubits": 12,
    "physical_qubits": 97,
    "two_qubit_gates": 2132,
    "logical_T_gate_infidelity": 2.6e-3,
    "outcome": "desempeno mejor-que-aleatorio mantenido (no 'break-even' total a esta escala)",
}

# --- extrapolacion declarada del T-count, desde nuestro propio dato medido ---
OUR_QAOA_REF_RATIO_TCOUNT_OVER_2Q = 56 / 79  # qaoa_farm.qasm, medido esta sesion
EXTRAPOLATED_TCOUNT = round(OUR_QAOA_REF_RATIO_TCOUNT_OVER_2Q * STEANE_DEMO["two_qubit_gates"])


def compare(p_phys_grid=(1e-3, 1e-4), verbose=True):
    if verbose:
        print(f"{'='*90}\nCOMPARACION DE FAMILIA DE CODIGO: superficie (este proyecto) vs Steane (dato real, {STEANE_DEMO['source']})\n{'='*90}")
        print(f"  Steane, REAL, medido en hardware: {STEANE_DEMO['logical_qubits']} qubits logicos, "
              f"{STEANE_DEMO['physical_qubits']} qubits fisicos ({STEANE_DEMO['physical_qubits']/STEANE_DEMO['logical_qubits']:.1f}x), "
              f"{STEANE_DEMO['two_qubit_gates']} compuertas de 2 qubits, infidelidad T logica={STEANE_DEMO['logical_T_gate_infidelity']:.1e}")
        print(f"  T-count EXTRAPOLADO (no reportado por ellos) desde nuestra razon T-count/2q "
              f"medida en QAOA/farm: {EXTRAPOLATED_TCOUNT}")
        print()

    results = []
    for p in p_phys_grid:
        azure = estimate_with_azure_qre(num_logical_qubits=STEANE_DEMO["logical_qubits"],
                                          tcount=EXTRAPOLATED_TCOUNT, p_phys=p)
        own = estimate_open_ftqc(num_logical_qubits=STEANE_DEMO["logical_qubits"],
                                   tcount=EXTRAPOLATED_TCOUNT, p_phys=p)
        ratio_azure = azure["physicalCounts"]["physicalQubits"] / STEANE_DEMO["physical_qubits"]
        row = {"p_phys": p, "azure_qubits": azure["physicalCounts"]["physicalQubits"],
               "azure_d": azure["logicalQubit"]["codeDistance"],
               "ratio_azure_over_steane_real": ratio_azure}
        if own["feasible"]:
            row["own_qubits"] = own["physicalQubits"]
            row["own_d"] = own["codeDistance"]
            row["ratio_own_over_steane_real"] = own["physicalQubits"] / STEANE_DEMO["physical_qubits"]
        results.append(row)

        if verbose:
            print(f"  p_fisico={p:.0e}  (asumiendo codigo de SUPERFICIE, para un circuito de "
                  f"12 qubits logicos comparable):")
            print(f"    Azure QRE       : d={row['azure_d']}  qubits={row['azure_qubits']:,}  "
                  f"({ratio_azure:.0f}x mas que el Steane real medido)")
            if own["feasible"]:
                print(f"    Estimador propio: d={row['own_d']}  qubits={row['own_qubits']:,}  "
                      f"({row['ratio_own_over_steane_real']:.0f}x mas que el Steane real medido, "
                      f"pese a NO modelar fabricas de destilacion)")
            print()

    if verbose:
        print("CONCLUSION HONESTA: incluso con nuestro estimador mas conservador (que "
              "subestima qubits al no modelar fabricas), un circuito de este tamano "
              "necesitaria un orden de magnitud mas de qubits fisicos bajo el supuesto "
              "de codigo de superficie que lo que Quantinuum demostro con codigo Steane "
              "en iones atrapados. Esto NO significa que el codigo de superficie sea peor "
              "en general -- el Steane requiere conectividad todos-con-todos (viable en "
              "iones atrapados, no en qubits superconductores de conectividad local, que "
              "es donde Azure QRE y nuestro pipeline asumen que se corre) -- significa que "
              "la eleccion de FAMILIA DE CODIGO, no solo de tecnica de reduccion, es una "
              "variable que este proyecto no ha explorado todavia, y que cambia las "
              "conclusiones del barrido de cruce FTQC de forma sustancial.")

    return results


if __name__ == "__main__":
    compare()
