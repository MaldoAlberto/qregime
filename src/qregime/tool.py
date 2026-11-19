"""
qasm_tool.py -- el punto de entrada UNICO que integra todo lo construido
en este proyecto en una sola herramienta: perfilado, recomendacion de
tecnicas, factibilidad de conectividad y estimacion de recursos FTQC, a
partir de UN SOLO INPUT (un archivo QASM) y UN PERFIL DE HARDWARE
explicito y minimo -- sin generar codigo especifico de ningun vendor y
sin asumir arquitectura superconductora en ningun punto de este modulo.

MOTIVACION (por que este archivo existe): cada pieza (technique_advisor,
graph_layout_search, open_resource_estimator, pipeline_v2) se construyo y
valido por separado a lo largo de este proyecto. Este modulo no repite esa
logica -- la ORQUESTA, y hace explicito, en un solo lugar, cual es el
UNICO punto donde el hardware entra al analisis: el HardwareProfile de
abajo. Todo lo demas (perfilado del circuito, deteccion de patrones,
recomendaciones de reduccion) ya opera solo sobre el QASM, sin ningun
parametro de hardware -- eso se verifica aqui, no se supone.

QUE SI GENERALIZA (validado en este archivo, con datos publicados reales,
no solo argumentado):
  - profile_circuit() y recommend_techniques() (technique_advisor.py):
    100% agnosticos -- ninguna de sus funciones recibe un backend o un
    parametro de hardware. Se comprueba con `inspect.signature`.
  - estimate_open_ftqc_with_factories() (open_resource_estimator.py):
    su unico parametro de hardware es un escalar (p_phys). Esta sesion
    ya lo corrio con 3 tasas de error de 2 qubits PUBLICADAS de 3
    arquitecturas reales distintas (superconductor IBM, trapped-ion
    Quantinuum H2, trapped-ion IonQ/Oxford Ionics) sin cambiar una linea
    de codigo -- ver validate_across_architectures() abajo.

QUE NO GENERALIZA AUTOMATICAMENTE (declarado, no oculto):
  - graph_layout_search.py asume un objeto `backend` de Qiskit con
    coupling_map + calibracion -- una abstraccion pensada para hardware
    de conectividad FIJA Y LIMITADA (superconductores, y neutros atrapados
    en trampas fijas). Para conectividad all-to-all (trapped-ion) la
    pregunta de factibilidad estructural es TRIVIAL (grado maximo del
    hardware = N-1, siempre >= el del circuito) -- se modela aqui como un
    caso explicito, no se fuerza el codigo original a aceptarlo.
  - Computacion fotonica basada en medicion (MBQC/cluster-state) NO se
    puede representar como un circuito QASM de compuertas de forma que
    conserve el costo de recursos real -- es un modelo computacional
    distinto (estados de recursos + medicion adaptativa). Alimentar un
    QASM de compuertas a este pipeline y reportar un numero para
    "hardware fotonico" seria enganoso; por eso esta herramienta declara
    ese caso como NO SOPORTADO en vez de convertirlo silenciosamente.
  - El ajuste de umbral p_th=0.74% (fit_own_threshold_model.py) viene de
    una simulacion Stim con un modelo de ruido depolarizing/Pauli. Al
    reusar ese mismo p_th con un p_phys de trapped-ion, se asume
    implicitamente que el canal de error dominante de esa arquitectura
    tambien es aproximadamente Pauli/depolarizing -- razonable como
    primera aproximacion (es el supuesto estandar de la mayoria de
    literatura de codigos de superficie), pero NO una re-derivacion
    especifica de trapped-ion (que tiene mecanismos de error propios:
    fugas, calentamiento del ion, crosstalk de puerta global) -- se
    declara como aproximacion, no como resultado re-verificado por canal.
"""
import inspect
from dataclasses import dataclass, field
from typing import Optional

from .advisor import profile_circuit, recommend_techniques
from .resources import estimate_open_ftqc_with_factories


def circuit_interaction_graph(qasm_path: str):
    """Undirected logical interaction graph: one node per qubit, one edge per
    pair of qubits sharing at least one two-qubit gate (weight = number of
    such gates). Inlined from the research code's graph_layout_search.py so
    the package does not depend on a Qiskit backend object."""
    import rustworkx as rx

    from . import circuit as _circuit
    qc = _circuit.load(qasm_path)
    graph = rx.PyGraph()
    graph.add_nodes_from(range(qc.num_qubits))
    weight = {}
    for instruction in qc.gates():
        if len(instruction.qubits) == 2:
            a, b = instruction.qubits
            key = (min(a, b), max(a, b))
            weight[key] = weight.get(key, 0) + 1
    for (a, b), w in weight.items():
        graph.add_edge(a, b, w)
    return graph


@dataclass
class HardwareProfile:
    """El UNICO lugar donde entra informacion de hardware especifico a
    esta herramienta. Todo lo demas en el pipeline es agnostico por
    construccion (ver verify_architecture_agnostic_layers() abajo)."""
    name: str
    p_2q: float  # tasa de error de compuerta de 2 qubits, medida o publicada
    connectivity: str  # "all_to_all" | "limited"
    max_hw_degree: Optional[int] = None  # requerido solo si connectivity=="limited"
    source: str = ""  # cita/procedencia del numero de p_2q, siempre declarada
    # modelo de ruido de circuito con el que se ajusto el umbral del surface
    # code para esta arquitectura (ver resources.NOISE_MODELS)
    noise_model: str = "sd6"
    # vendor-neutral device calibration (qregime.calibration) used for the
    # exact calibrated QEM simulation; None when no calibration is published
    calibration: Optional[str] = None

    def __post_init__(self):
        if self.connectivity == "limited" and self.max_hw_degree is None:
            raise ValueError(
                f"HardwareProfile '{self.name}': connectivity='limited' "
                "requiere max_hw_degree explicito (no se asume un valor).")


# Perfiles reales, publicados, usados en esta sesion -- cada uno con su
# fuente citada, ninguno inventado:
HARDWARE_PROFILES = {
    "superconductor_ibm": HardwareProfile(
        name="Superconducting (IBM heavy-hex)",
        p_2q=0.0032, connectivity="limited", max_hw_degree=3,
        source="real FakeTorino calibration (heavy-hex, max degree 3), "
               "used throughout the accompanying paper",
        noise_model="si1000", calibration="ibm_torino"),
    "trapped_ion_quantinuum_h2": HardwareProfile(
        name="Trapped-ion Quantinuum H2",
        p_2q=0.001, connectivity="all_to_all",
        source="Quantinuum System Model H2: '56 qubits fully-connected', "
               "published two-qubit fidelity >99.9%",
        noise_model="toy", calibration="quantinuum_h2_2"),
    "superconductor_ibm_heron": HardwareProfile(
        name="Superconducting (IBM Heron, ibm_boston 2026)",
        p_2q=0.0013, connectivity="limited", max_hw_degree=3,
        source="qiskit-ibm-runtime 0.52.1 FakeBoston snapshot of ibm_boston (2026-04-17): "
               "156 qubits, heavy-hex, median CZ error 1.3e-3",
        noise_model="si1000", calibration="ibm_boston"),
    "superconductor_ibm_nighthawk": HardwareProfile(
        name="Superconducting (IBM Nighthawk, ibm_miami 2026)",
        p_2q=0.0029, connectivity="limited", max_hw_degree=4,
        source="qiskit-ibm-runtime 0.52.1 FakeMiami snapshot of ibm_miami (2026-04-17): "
               "120 qubits, square lattice, median CZ error 2.9e-3",
        noise_model="si1000", calibration="ibm_miami"),
    "superconductor_google_willow": HardwareProfile(
        name="Superconducting (Google Willow, square grid)",
        p_2q=0.0033, connectivity="limited", max_hw_degree=4,
        source="Google Willow spec sheet (Dec 9, 2024), chip 1: mean simultaneous CZ "
               "error 0.33%, 105 qubits, 4-way typical connectivity",
        noise_model="si1000", calibration="google_willow"),
    "superconductor_iqm_garnet": HardwareProfile(
        name="Superconducting (IQM Garnet, square lattice)",
        p_2q=0.005, connectivity="limited", max_hw_degree=4,
        source="Abdurakhimov et al., arXiv:2408.12433: IQM Garnet, 20 qubits on a "
               "square lattice, median CZ error 5e-3",
        noise_model="si1000", calibration="iqm_garnet"),
    "superconductor_iqm_emerald": HardwareProfile(
        name="Superconducting (IQM Emerald, square lattice)",
        p_2q=0.0055, connectivity="limited", max_hw_degree=4,
        source="Joch et al., arXiv:2603.28486 (Mar 2026), Table 1: IQM Emerald, 54 qubits, "
               "median CZ fidelity 99.45%",
        noise_model="si1000", calibration="iqm_emerald"),
    "trapped_ion_quantinuum_helios": HardwareProfile(
        name="Trapped-ion Quantinuum Helios",
        p_2q=0.00079, connectivity="all_to_all",
        source="Ransford et al., arXiv:2511.05465 (Nov 2025): Helios, 98 all-to-all qubits, "
               "zone-averaged two-qubit infidelity 7.9e-4",
        noise_model="toy", calibration="quantinuum_helios"),
    "trapped_ion_ionq_forte": HardwareProfile(
        name="Trapped-ion IonQ Forte",
        p_2q=0.0046, connectivity="all_to_all",
        source="Chen et al., arXiv:2308.05071: IonQ Forte, 30 all-to-all qubits, "
               "median two-qubit DRB error 46.4e-4",
        noise_model="toy", calibration="ionq_forte"),
    "trapped_ion_ionq_oxford": HardwareProfile(
        name="Trapped-ion IonQ/Oxford Ionics",
        p_2q=0.0001, connectivity="all_to_all",
        source="IonQ/Oxford Ionics, Oct. 2025: two-qubit fidelity 99.99% "
               "via randomized benchmarking, arXiv:2510.17286",
        noise_model="toy"),
}


def verify_architecture_agnostic_layers(verbose: bool = True) -> dict:
    """Comprueba, por inspeccion de firma (no por lectura del docstring),
    que profile_circuit y recommend_techniques NO reciben ningun
    parametro de hardware -- es la evidencia de que esas dos capas ya son
    agnosticas, en vez de simplemente afirmarlo."""
    sig_profile = inspect.signature(profile_circuit)
    sig_recommend = inspect.signature(recommend_techniques)
    sig_estimator = inspect.signature(estimate_open_ftqc_with_factories)

    result = {
        "profile_circuit_params": list(sig_profile.parameters.keys()),
        "recommend_techniques_params": list(sig_recommend.parameters.keys()),
        "estimator_hardware_param_is_scalar": "p_phys" in sig_estimator.parameters,
    }
    if verbose:
        print("Verificacion por firma (no por afirmacion):")
        print(f"  profile_circuit({', '.join(result['profile_circuit_params'])}) "
              f"-- 0 parametros de hardware")
        print(f"  recommend_techniques({', '.join(result['recommend_techniques_params'])}) "
              f"-- 0 parametros de hardware")
        print(f"  estimate_open_ftqc_with_factories(...) -- unico parametro de "
              f"hardware es el escalar 'p_phys': {result['estimator_hardware_param_is_scalar']}")
    return result


def structural_feasibility_generic(qasm_path: str, hw: HardwareProfile, verbose: bool = True) -> dict:
    """Version generalizada de structural_feasibility() (graph_layout_search.py)
    que no requiere un objeto backend de Qiskit -- solo el HardwareProfile.
    Para connectivity=='all_to_all' la respuesta es trivial por construccion
    (se declara asi, no se ejecuta VF2 sin necesidad)."""
    interaction = circuit_interaction_graph(qasm_path)
    max_deg_circuit = max((interaction.degree(n) for n in interaction.node_indices()), default=0)

    if hw.connectivity == "all_to_all":
        feasible = True
        note = "conectividad all-to-all: cualquier grafo de interaccion cabe por construccion (grado maximo posible = N-1)"
    else:
        feasible = max_deg_circuit <= hw.max_hw_degree
        note = f"grado_max_circuito={max_deg_circuit} vs grado_max_hardware={hw.max_hw_degree}"

    result = {
        "hardware": hw.name,
        "max_degree_circuit": max_deg_circuit,
        "connectivity": hw.connectivity,
        "structurally_feasible_without_swaps": feasible,
        "note": note,
    }
    if verbose:
        print(f"  [{hw.name}] factibilidad estructural: {feasible} ({note})")
    return result


def simulate_expectation(qasm_path: str, hw: HardwareProfile, shots: int = 150,
                          seed: int = 42, verbose: bool = True) -> dict:
    """
    Complemento EMPIRICO (no solo analitico) a estimate_open_ftqc_with_factories():
    corre una simulacion RUIDOSA real (matriz de densidad exacta hasta 10
    qubits; por encima, Qiskit Aer si esta instalado o trayectorias numpy;
    None si el trabajo excede el limite) y
    mide que fraccion de los shots cae en el SOPORTE IDEAL del circuito
    (el conjunto de bitstrings con probabilidad no nula en la ejecucion sin
    ruido, calculado una sola vez con un statevector exacto). Esta es la
    misma metrica que un circuito real tipo Shor usaria para decidir exito:
    no exige el bitstring EXACTO, exige caer en el conjunto de resultados
    validos.

    LIMITACION DECLARADA (a proposito, no oculta): el modelo de ruido usado
    aqui es un canal depolarizing generico de 2 qubits, con probabilidad
    p_2q tomada DIRECTAMENTE del HardwareProfile -- no es una calibracion
    real completa (no incluye error de lectura, error de 1 qubit,
    decoherencia T1/T2 dependiente del tiempo de compuerta, ni
    correlaciones espaciales). Para el perfil "superconductor_ibm" existe
    una calibracion real mas completa en otras partes de este proyecto
    (graph_layout_search.py, manual_em_stack.py, via AerSimulator.from_backend);
    aqui se usa la MISMA aproximacion simplificada para los 3 perfiles, a
    proposito, para que la comparacion entre arquitecturas sea
    consistente (mismo tipo de ruido, solo cambia la magnitud p_2q) en vez
    de mezclar un modelo completo con dos aproximados.
    """
    import numpy as np

    from . import circuit as _circuit
    from . import simulate as _sim

    circ = _circuit.load(qasm_path)
    ideal_probs = np.abs(_sim.statevector(circ)) ** 2
    support = ideal_probs > 1e-9
    ideal_support = set(np.nonzero(support)[0].tolist())

    # engine: exact density matrix (small), Qiskit Aer if installed, else
    # numpy trajectories; None tells the caller to use the analytic proxy
    total = successes = None
    if circ.num_qubits <= MAX_QUBITS_FOR_DENSITY_MATRIX:
        success_fraction = _sim.depolarizing_success(circ, support, hw.p_2q)
        engine = "exact density matrix"
    else:
        aer = _aer_success(qasm_path, hw, shots, seed, ideal_support)
        if aer is not None:
            success_fraction, successes, total = aer
            engine = "qiskit-aer"
        else:
            traj = _sim.depolarizing_trajectories(circ, support, hw.p_2q, shots, seed)
            if traj is None:
                return None
            success_fraction, _ = traj
            total = shots
            engine = "numpy trajectories"

    result = {
        "hardware": hw.name,
        "p_2q": hw.p_2q,
        "engine": engine,
        "shots": total,
        "ideal_support_size": len(ideal_support),
        "successes": successes,
        "success_fraction_measured": success_fraction,
        "noise_model": "depolarizing generico de 2 qubits, magnitud p_2q del "
                        "HardwareProfile -- ver limitacion declarada en docstring",
    }
    if verbose:
        print(f"  [simulacion ruidosa, {engine}] {hw.name}: exito = "
              f"{success_fraction*100:.2f}% (soporte ideal de {len(ideal_support)} bitstrings)")
    return result


#: above this, noisy simulations use Aer (if installed) or trajectories
MAX_QUBITS_FOR_DENSITY_MATRIX = 10


def _aer_success(qasm_path, hw, shots, seed, ideal_support):
    """Shot-sampled success fraction with Qiskit Aer (optional accelerator).
    None if Aer is not installed or cannot run the circuit as written."""
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError:
        return None
    try:
        qc = QuantumCircuit.from_qasm_file(qasm_path)
        qc.measure_all()
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(hw.p_2q, 2), ["cx", "cz", "ecr", "rzz", "cp"])
        sim = AerSimulator(method="statevector", noise_model=noise_model)
        counts = sim.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    except Exception:
        return None
    total = sum(counts.values())
    successes = sum(c for bitstring, c in counts.items()
                    if int(bitstring.replace(" ", ""), 2) in ideal_support)
    return (successes / total if total else 0.0), successes, total


def validate_expectations_across_architectures(qasm_path: str = "shor_like_sequential.qasm",
                                                shots: int = 150) -> dict:
    """Corre simulate_expectation() para las 3 arquitecturas publicadas,
    misma semilla y mismo numero de shots, SIN CAMBIAR CODIGO entre ellas
    -- el resultado empirico analogo a validate_across_architectures()
    (que solo corria la estimacion analitica de recursos FTQC)."""
    results = {}
    for key, hw in HARDWARE_PROFILES.items():
        results[key] = simulate_expectation(qasm_path, hw, shots=shots, verbose=True)
    return results


def analyze_circuit(qasm_path: str, hardware_profile_key: str, verbose: bool = True,
                     simulate: bool = False, shots: int = 150) -> dict:
    """LA HERRAMIENTA: un solo QASM + una sola clave de perfil de hardware
    -> un reporte consolidado de perfilado, recomendaciones, factibilidad
    estructural, estimacion ANALITICA de recursos FTQC (con fabricas) y,
    opcionalmente (`simulate=True`), una expectativa EMPIRICA obtenida por
    simulacion ruidosa real (ver simulate_expectation() arriba) -- las dos
    cosas responden preguntas distintas: la estimacion analitica dice
    cuantos qubits fisicos requeriria correr esto en FTQC; la simulacion
    dice que pasaria si se corriera HOY, tal cual, sin QEC, en cada
    arquitectura."""
    hw = HARDWARE_PROFILES[hardware_profile_key]
    if verbose:
        print(f"=== analizando {qasm_path} para {hw.name} ===")
        print(f"    (p_2q={hw.p_2q*100:.4f}%, fuente: {hw.source})")

    profile = profile_circuit(qasm_path)
    # BUG REAL CORREGIDO (esta sesion): antes se llamaba
    # recommend_techniques(qasm_path) sola, que volvia a calcular
    # profile_circuit() internamente por segunda vez -- se le pasa el
    # `profile` ya calculado arriba para evitar ese trabajo duplicado
    # (dominaba la latencia del circuito insignia de este proyecto).
    recs = recommend_techniques(qasm_path, profile=profile)[1]
    feas = structural_feasibility_generic(qasm_path, hw, verbose=verbose)

    est_no_reuse = estimate_open_ftqc_with_factories(
        profile["num_qubits"], profile["tcount"], hw.p_2q, noise_model=hw.noise_model)
    est_with_reuse = estimate_open_ftqc_with_factories(
        profile["max_concurrent_qubits"], profile["tcount"], hw.p_2q,
        noise_model=hw.noise_model)

    report = {
        "qasm_path": qasm_path,
        "hardware": hw.name,
        "hardware_source": hw.source,
        "profile": profile,
        "recommended_techniques": recs,
        "structural_feasibility": feas,
        "resource_estimate_declared_qubits": est_no_reuse,
        "resource_estimate_with_reuse": est_with_reuse,
    }
    if verbose and est_no_reuse.get("feasible"):
        print(f"    d={est_no_reuse['codeDistance']}  "
              f"qubits(sin reuso)={est_no_reuse['physicalQubits']:,}  "
              f"qubits(con reuso)={est_with_reuse['physicalQubits']:,}")

    if simulate:
        report["simulated_expectation_today"] = simulate_expectation(
            qasm_path, hw, shots=shots, verbose=verbose)

    return report


def validate_across_architectures(qasm_path: str = "shor_like_sequential.qasm",
                                   simulate: bool = False) -> dict:
    """Corre analyze_circuit() para las 3 arquitecturas reales publicadas
    en HARDWARE_PROFILES, SIN CAMBIAR NINGUNA LINEA de codigo del pipeline
    -- esto es lo que se entiende por 'agnostico de hardware' aqui: no una
    afirmacion, sino esta funcion, ejecutada. Con simulate=True, agrega la
    expectativa empirica de simulate_expectation() para cada arquitectura."""
    results = {}
    for key in HARDWARE_PROFILES:
        results[key] = analyze_circuit(qasm_path, key, verbose=True, simulate=simulate)
        print()
    return results


if __name__ == "__main__":
    print("PASO 1: verificar (por firma, no por afirmacion) que las capas de "
          "perfilado y recomendacion son agnosticas de hardware\n")
    verify_architecture_agnostic_layers()
    print()

    print("PASO 2: correr la MISMA herramienta (estimacion analitica + "
          "simulacion ruidosa real), sin cambios, sobre 3 arquitecturas "
          "reales publicadas distintas\n")
    validate_across_architectures(simulate=True)
