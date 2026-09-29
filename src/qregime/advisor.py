"""
Capa de PRE-decision: antes de correr cualquier tecnica de reduccion (ZX,
teletransportacion, AND-gate, QCP, descomputo diferido), medir metricas
baratas del circuito (compuertas, qubits, grado de interaccion, ancillas
candidatas) y usar los UMBRALES que esta misma sesion midio -- no numeros
inventados -- para predecir cuales tecnicas vale la pena intentar.

Esto es una heuristica calibrada con una muestra pequena (4 familias de
circuito + 9 instancias QOBLIB de un benchmark previo) -- se declara asi,
no como un clasificador general validado. El objetivo es automatizar la
Parte 4.6 del reporte (que tecnica sirve para que tipo de circuito) en vez
de dejarla como una tabla cualitativa.
"""

import warnings
warnings.filterwarnings("ignore")

import collections

import networkx as nx
import pyzx as zx
from qiskit import QuantumCircuit as _QiskitCircuit
from qiskit.quantum_info import Statevector as _Statevector, partial_trace as _partial_trace, entropy as _entropy

from .hypergraph import CircuitHypergraph, resource_report
from .qcp_tracker import run_tracker
from .sparsifiability import spectral_sparsifiability


_PARAMETRIZED_TWO_QUBIT_GATES = {"rzz", "crz", "cp", "rxx", "ryy", "cu1", "cry", "crx"}
# `rz` (1 qubit) se incluye TAMBIEN: verificado directamente que
# `qaoa_farm.qasm`/`qaoa_small_test.qasm` no usan ninguna compuerta de la
# lista anterior -- su operador de costo esta escrito ya descompuesto como
# CX; RZ(2*gamma); CX (la forma estandar de implementar exp(-i*gamma*ZZ)),
# asi que el angulo gamma repetido vive en la compuerta `rz` intermedia,
# no en una compuerta de 2 qubits con nombre propio. Se verifico que esto
# es seguro de agregar: ninguno de los otros 7 circuitos de referencia usa
# `rz` en absoluto (adder8, qft10, random8, vqe_ansatz, grover), asi que no
# se introduce ningun nuevo riesgo de falso positivo por este cambio.
_PARAMETRIZED_SINGLE_QUBIT_GATES_IN_ZZ_DECOMPOSITION = {"rz"}


_NATIVE_DIAGONAL_TWO_QUBIT_GATES = {"rzz", "cp", "crz", "cu1"}
# Estas compuertas SON ya, atomicamente, exp(-i*theta*Z_aZ_b) (o una
# fase equivalente) -- no necesitan el sandwich CX;RZ;CX porque no estan
# descompuestas; se cuentan como motivo diagonal directamente, sin
# importar theta.


def _zz_cost_operator_motif_fraction(qasm_path: str) -> float:
    """GENERALIZACION (v3, esta sesion) de lo que originalmente se llamo
    la firma "QAOA": el mecanismo real que hace que ZX-calculus no reduzca
    nada no es "es QAOA" -- es que el circuito implementa una unica CAPA
    HAMILTONIANA DIAGONAL EN Z (una suma de terminos Z_aZ_b que conmutan
    entre si, del tipo exp(-i*theta*Z_aZ_b) por arista). Esto se sigue de
    la teoria de polinomios de fase detras de ZX-calculus (Kissinger &
    van de Wetering): el poder real de `full_reduce()` para bajar T-count
    viene de fusionar spiders A TRAVES de bases no diagonales (aristas
    Hadamard); si TODA la parte no-Clifford del circuito ya es una capa
    diagonal ya fusionada al maximo por construccion, no hay nada que
    fusionar mas -- es un resultado estructural, no solo empirico. Esto
    aplica a QAOA (el caso que motivo la regla original) pero TAMBIEN a
    cualquier otro ansatz de una sola capa Hamiltoniana diagonal: VQE de
    Ising, QUBO, otros solvers de tipo "costo diagonal + mezclador".

    Se detectan DOS formas equivalentes de escribir ese termino, para no
    quedar atado a una decomposicion especifica:
      1. el sandwich CX(a,b); RZ(theta); CX(a,b) (la forma "decompuesta",
         que es la que usan los generadores de circuito de este proyecto
         y la que motivo el hallazgo original del falso positivo/negativo)
      2. una compuerta diagonal NATIVA de 2 qubits (`rzz`, `cp`, `crz`,
         `cu1`) aplicada directamente sobre esa arista -- la forma en que
         un backend con compuerta rzz nativa (p.ej. trapped-ion) o un
         transpilador que no descompone escribiria lo mismo.
    Ambas formas son angulo-agnosticas por construccion (theta puede ser
    distinto por arista, como en QAOA con pesos de arista, y se sigue
    detectando).

    Limitacion declarada: el escaneo del sandwich CX;RZ;CX sigue siendo
    de TRIPLETES CONTIGUOS (no un analisis de conmutacion completo), asi
    que un circuito reordenado por un transpilador podria evadir esa
    rama de deteccion -- se declara como limite conocido, no se resuelve
    aqui.

    Devuelve la fraccion de aristas del grafo de interaccion que SI
    exhiben alguna de las dos formas (0.0 si no hay ninguna)."""
    qc = _QiskitCircuit.from_qasm_file(qasm_path)
    data = qc.data
    motif_edges = set()
    total_two_qubit_edges = set()
    for i in range(len(data)):
        instr = data[i]
        if len(instr.qubits) != 2:
            continue
        a = qc.find_bit(instr.qubits[0]).index
        b = qc.find_bit(instr.qubits[1]).index
        edge = (min(a, b), max(a, b))

        if instr.operation.name in _NATIVE_DIAGONAL_TWO_QUBIT_GATES:
            total_two_qubit_edges.add(edge)
            motif_edges.add(edge)
            continue

        if instr.operation.name != "cx":
            total_two_qubit_edges.add(edge)
            continue
        total_two_qubit_edges.add(edge)
        # buscar el sandwich CX(a,b) ... RZ ... CX(a,b) en tripletes
        # contiguos (permitiendo compuertas de 1 qubit de por medio)
        j = i + 1
        saw_rz = False
        while j < len(data) and j < i + 4:
            nxt = data[j]
            nxt_qubits = {qc.find_bit(q).index for q in nxt.qubits}
            if len(nxt.qubits) == 1 and nxt_qubits <= {a, b}:
                if nxt.operation.name == "rz":
                    saw_rz = True
                j += 1
                continue
            if (nxt.operation.name == "cx" and len(nxt.qubits) == 2
                    and {qc.find_bit(q).index for q in nxt.qubits} == {a, b}):
                if saw_rz:
                    motif_edges.add(edge)
                break
            break
    if not total_two_qubit_edges:
        return 0.0
    return len(motif_edges) / len(total_two_qubit_edges)


def _qaoa_repeated_angle_signature(qasm_path: str) -> bool:
    """Resuelve la causa raiz del falso positivo reportado sobre
    `shor_like_sequential.qasm`: el patron `qaoa_like_pattern` original
    solo miraba forma del grafo (grado maximo, dispersion espectral), y
    NUNCA verificaba el mecanismo real que distingue a un operador de
    costo QAOA -- que sus compuertas de 2 qubits son rotaciones
    PARAMETRIZADAS (exp(-i*gamma*Z_aZ_b)) con el MISMO angulo gamma
    repetido en todas las aristas del grafo. Verificado directamente:
    `shor_like_sequential.qasm` esta compuesto solo de compuertas `u` (1
    qubit) y `cx` (2 qubits, FIJA, sin parametro) -- cero compuertas de
    2 qubits parametrizadas, así que no puede tener esta firma sin
    importar que tan denso o dispersamente conectado este su grafo de
    interaccion. `qaoa_farm.qasm` y `qaoa_small_test.qasm`, en cambio, sí
    tienen exactamente 1 angulo distinto repetido en el 100% de sus
    compuertas parametrizadas de 2 qubits -- la firma real.
    Devuelve False (no hay firma QAOA posible) si el circuito no usa
    NINGUNA compuerta de 2 qubits parametrizada; si usa alguna, True solo
    si un unico angulo domina (>=30% de esas instancias)."""
    qc = _QiskitCircuit.from_qasm_file(qasm_path)
    relevant_gate_names = _PARAMETRIZED_TWO_QUBIT_GATES | _PARAMETRIZED_SINGLE_QUBIT_GATES_IN_ZZ_DECOMPOSITION
    angles = []
    for instr in qc.data:
        if instr.operation.name in relevant_gate_names and instr.operation.params:
            angles.append(round(float(instr.operation.params[0]), 6))
    if not angles:
        return False
    most_common_count = collections.Counter(angles).most_common(1)[0][1]
    return (most_common_count / len(angles)) >= 0.3


_MAX_QUBITS_FOR_EXACT_ENTANGLEMENT_PROBE = 20
# CORREGIDO (esta sesion, tras medir latencia real tratando de arreglar
# los 3 bugs de escala encontrados en la prueba de generalizacion a gran
# escala): el limite anterior (24) resulto ser SOLO sobre qubits, y NO
# acotaba el tiempo real -- medido directamente, `Statevector.from_
# instruction` + `partial_trace` completa en 6-12s a 20 qubits (aceptable
# para uso en linea) pero pasa a colgarse mas de 40s a 22-24 qubits
# incluso para circuitos con pocas compuertas (p.ej. GHZ_n24, solo 24
# compuertas, igual colgo) -- la memoria (2^24 amplitudes ~256MB, mas los
# intermedios de partial_trace) parece ser el cuello de botella real, no
# el numero de compuertas. Se baja el limite a 20 qubits, el mayor valor
# confirmado empiricamente rapido, en vez de dejar un limite que en teoria
# "degrada con gracia" pero en la practica cuelga el pipeline completo
# antes de llegar a esa degradacion.


def _final_state_entanglement_fraction(qasm_path: str):
    """Version barata, de UN SOLO PUNTO (solo el estado final, no la curva
    completa de crecimiento por profundidad) de la metrica implementada y
    validada en `entanglement_growth.py` -- ver ese archivo para la
    motivacion completa (respuesta directa al punto debil encontrado al
    comparar contra Tindall et al. 2023 sobre el experimento de "utilidad
    cuantica" de IBM: nuestros propios estimadores de recursos, igual que
    los de Azure, nunca preguntan si el circuito tiene estructura
    clasicamente explotable via redes tensoriales / bond dimension bajo).

    Calcula la entropia de entrelazamiento bipartita (von Neumann, en bits)
    del ESTADO FINAL del circuito, sobre el corte mas simple posible
    (primera mitad de qubits vs segunda mitad, por indice de registro), y
    la reporta como fraccion del maximo teorico posible para ese tamano de
    corte (min(n_a, n_b) bits).

    MISMA limitacion declarada que en `entanglement_growth.py`: esta
    biparticion no es necesariamente el corte optimo para detectar
    compresibilidad tipo-arbol (un corte informado por la geometria real
    del grafo de interaccion podria ser mas representativo); esto es una
    PRIMERA APROXIMACION barata para uso rutinario en el pipeline, no un
    analisis riguroso de bond dimension de una MPS optimizada (eso
    requeriria una libreria de redes tensoriales que este proyecto no
    tiene instalada).

    Devuelve None (degradacion con gracia, no un error) para circuitos por
    encima de `_MAX_QUBITS_FOR_EXACT_ENTANGLEMENT_PROBE` qubits, donde el
    statevector exacto deja de ser practico."""
    qc = _QiskitCircuit.from_qasm_file(qasm_path)
    n = qc.num_qubits
    if n < 2 or n > _MAX_QUBITS_FOR_EXACT_ENTANGLEMENT_PROBE:
        return None
    n_a = n // 2
    max_entropy_bits = min(n_a, n - n_a)
    if max_entropy_bits == 0:
        return None
    sv = _Statevector.from_instruction(qc)
    rho_a = _partial_trace(sv, list(range(n_a, n)))
    s_bits = float(_entropy(rho_a, base=2))
    return s_bits / max_entropy_bits


def profile_circuit(qasm_path: str, n_pcm_probe: int = 4):
    circuit = zx.Circuit.from_qasm_file(qasm_path).to_basic_gates()
    hg = CircuitHypergraph(circuit)
    metrics = resource_report(circuit)
    ig = hg.interaction_graph()

    degrees = [d for _, d in ig.degree()]
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0

    # heuristica barata de "ancilla candidata a descartable": un qubit cuyo
    # ULTIMO uso en el circuito ocurre antes de la mitad del circuito (se
    # queda "quieto" el resto del tiempo) -- proxy, no una prueba exacta de
    # que vuelve a |0>, se declara asi
    first_use, last_use = {}, {}
    for he in hg.hyperedges:
        for q in he["qubits"]:
            if q not in first_use:
                first_use[q] = he["index"]
            last_use[q] = he["index"]
    midpoint = len(hg.hyperedges) / 2
    ancilla_candidates = sum(1 for q, idx in last_use.items() if idx < midpoint)

    # METRICA NUEVA (motivada por una pregunta real del usuario: "?se pueden
    # reusar qubits fisicos, apoyados por mediciones intermedias?"): en vez
    # del proxy crudo de arriba (antes/despues del punto medio), se calcula
    # el numero MINIMO de "ranuras" logicas simultaneas que el circuito
    # realmente necesita, tratando cada qubit como un intervalo de actividad
    # [primer_uso, ultimo_uso] y encontrando el maximo de intervalos
    # superpuestos en cualquier punto -- exactamente el problema clasico de
    # asignacion de registros via coloreo de grafo de intervalos (si se
    # permite medir+reiniciar un qubit logico y reusar su ranura fisica
    # para otro qubit logico que aun no empieza). Verificado sobre los 9
    # circuitos de referencia: la mayoria no tiene margen (concurrentes ==
    # declarados), pero `shor_like_sequential.qasm` si (16 declarados, 11
    # concurrentes reales, 31% de margen) y `vqe_ansatz.qasm` aun mas (8
    # declarados, 3 concurrentes, 62% de margen).
    events = []
    for q in first_use:
        events.append((first_use[q], 1))
        events.append((last_use[q] + 0.5, -1))
    events.sort()
    concurrent, max_concurrent_qubits = 0, 0
    for _, delta in events:
        concurrent += delta
        max_concurrent_qubits = max(max_concurrent_qubits, concurrent)

    # oportunidad de rastreo QCP-acotado a presupuesto modesto (n_pcm_probe)
    # con entrada en superposicion genuina -- proxy de "que tan globalmente
    # entrelazante es el circuito por diseno"
    # BUG REAL CORREGIDO (esta sesion): se pasa el `circuit` YA PARSEADO
    # (linea 211 de esta funcion) en vez de dejar que run_tracker() vuelva
    # a parsear el mismo archivo QASM desde cero -- para el circuito
    # insignia de este proyecto esto evita pagar el costo dominante del
    # pipeline (el parser de pyzx) dos veces por la misma llamada.
    qcp = run_tracker(qasm_path, n_pcm=n_pcm_probe, superpose_inputs=True, circuit=circuit)

    # dispersion espectral del grafo de interaccion (Spielman-Srivastava):
    # el MEJOR predictor honesto encontrado para "¿ayuda ZX-calculus?",
    # 70.4% de precision leave-one-family-out sobre las 98 instancias
    # completas del dataset de este proyecto (ver benchmarks/README.md) --
    # reemplaza la regla de arbol de decision anterior (43-47% LOFO)
    sparsifiability = spectral_sparsifiability(ig, tol=0.20, max_frac=0.9)

    return {
        "num_qubits": metrics["num_qubits"],
        "num_gates": metrics["num_gates"],
        "tcount": metrics["tcount"],
        "two_qubit_count": metrics["two_qubit_count"],
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "ancilla_candidates": ancilla_candidates,
        "max_concurrent_qubits": max_concurrent_qubits,
        "qubit_reuse_potential_fraction": 1 - (max_concurrent_qubits / metrics["num_qubits"]) if metrics["num_qubits"] else 0.0,
        "qcp_known_fraction_at_probe": qcp["known_fraction"],
        "spectral_sparsifiability": sparsifiability,
        "qaoa_repeated_angle_signature": _qaoa_repeated_angle_signature(qasm_path),
        "zz_cost_operator_motif_fraction": _zz_cost_operator_motif_fraction(qasm_path),
        "final_state_entanglement_fraction": _final_state_entanglement_fraction(qasm_path),
    }


def recommend_techniques(qasm_path: str, profile: dict = None):
    """
    BUG REAL CORREGIDO (esta sesion, encontrado midiendo latencia
    operacional): esta funcion siempre volvia a llamar
    `profile_circuit(qasm_path)` internamente, incluso cuando quien la
    llama YA habia calculado ese mismo perfil un momento antes (el patron
    usado en casi todos los scripts de este proyecto: `p =
    profile_circuit(...)` seguido de `recs = recommend_techniques(...)`
    sobre el MISMO archivo) -- duplicando, para el circuito insignia, mas
    de 15 segundos de trabajo (parseo QASM + rastreo QCP) por nada. Ahora
    se acepta el resultado ya calculado de `profile_circuit()` de forma
    opcional via el parametro `profile`; si no se da uno, se preserva el
    comportamiento anterior (llamarla internamente) para no romper codigo
    existente que ya invoca `recommend_techniques(qasm_path)` sola.
    Umbrales calibrados con los datos de ESTA sesion (no genericos):
      - teletransportacion RZZ: ayuda con grado promedio bajo (<15; el
        unico caso negativo real, MANN-a9, tenia ~40); en 25/27 corridas
        con grado de un digito hubo mejora.
      - ZX-calculus: la regla original (tcount/num_gates > 10%) se calibro
        con solo 4 circuitos y se probo despues, en un dataset real de 62
        circuitos (build_circuit_dataset.py: QFT n=3-12, sumador Cuccaro
        n=2-8, QAOA n=6-16 x densidad 0.2-0.8, aleatorios n=4-16 x
        profundidad 8-24), que acertaba solo 56.5% de las veces -- con
        27 falsos positivos, TODOS de la familia QAOA (T-count fraccional
        alto pero ganancia real de ZX-calculus = 0.0% en las 16 instancias
        probadas, sin excepcion). Un arbol de decision (profundidad 2,
        sklearn) entrenado sobre las mismas features con validacion cruzada
        de 5 pliegues llega a 85.1% +/- 9.8%, y una regresion logistica a
        88.8% +/- 3.5% -- ambos metodos CLASICOS de ML, no una red neuronal
        (con 62 muestras, un modelo de deep learning no tiene datos ni
        para no sobreajustar, no se intento). El arbol aprendio una segunda
        condicion (grado maximo del grafo de interaccion) que la regla a
        mano nunca tuvo, y que es exactamente lo que distingue a QAOA
        (T-count fraccional alto Y grado maximo alto -> NO ayuda) del resto
        (T-count fraccional alto y grado maximo moderado -> SI ayuda).
      - AND-gate + descomputo por medicion: requiere ancillas candidatas
        reales (>0 por esta heuristica) Y que esas ancillas no formen parte
        del registro de salida (no verificable automaticamente aqui -- se
        marca como "revisar manualmente", el mismo limite honesto que ya
        encontramos con el sumador Cuccaro).
      - QCP acotado / medicion intermedia general: solo si el circuito
        muestra fraccion de conocimiento > 0 a un presupuesto modesto
        (n_pcm=4) -- si da 0%, el circuito entrelaza globalmente por diseno
        y no hay nada que rastrear parcialmente.
    """
    p = profile if profile is not None else profile_circuit(qasm_path)
    recs = {}

    recs["teletransportacion_RZZ"] = {
        "recomendada": p["avg_degree"] < 15,
        "razon": f"grado promedio del grafo de interaccion = {p['avg_degree']:.1f} "
                 f"({'< 15 -> dentro del rango donde 25/27 casos mejoraron' if p['avg_degree'] < 15 else '>= 15 -> cerca del unico caso negativo real (MANN-a9, ~40)'})",
    }

    tcount_fraction = p["tcount"] / p["num_gates"] if p["num_gates"] else 0

    # CORRECCION IMPORTANTE (esta sesion): la version anterior de esta
    # regla (un arbol de decision de profundidad 2 sobre tcount_frac +
    # grado_maximo) se reporto con 82.9% de precision leave-one-family-out
    # usando spectral_sparsifiability. Al re-verificar HOY con el dataset
    # completo de 98 circuitos y el MISMO protocolo honesto (LOFO), la
    # cifra correcta es 70.4% -- y ese 70.4% resulta ser EXACTAMENTE igual
    # a la precision de un baseline trivial ("asumir que SI ayuda, salvo
    # que tcount_frac==0"). Es decir: NINGUNA de las features probadas
    # esta sesion (spectral_sparsifiability, grado maximo, costo de
    # contraccion tensorial, razon de angulos distintos, combinaciones de
    # las anteriores) supera de forma honesta a simplemente asumir que SI
    # ayuda por defecto. Se corrige aqui en vez de dejar la cifra inflada.
    #
    # Regla desplegada, con esto en mente: el UNICO filtro que es exacto
    # (no probabilistico) es tcount_frac==0 -- si no hay T-gates
    # fraccionales, no hay margen para reducir, punto. Fuera de eso, se
    # asume SI por defecto (grupo mayoritario real: 68% de las instancias
    # no triviales de este proyecto se benefician), y se emite una
    # ADVERTENCIA de baja confianza especificamente para el patron QAOA
    # (grado maximo alto Y muchas rotaciones de dos qubits con el MISMO
    # angulo repetido -- la firma de un operador de costo QAOA), el unico
    # caso donde 0/24 instancias probadas se beneficiaron pese a que todo
    # lo demas asuma que si.
    # CORRECCION v1 (esta sesion, tras encontrar un falso positivo real en
    # shor_like_sequential.qasm): se agrego la firma de angulo repetido
    # como condicion adicional. CORRECCION v2 (misma sesion, tras probar
    # el caso obvio que faltaba -- QAOA con pesos de arista real): la
    # firma de angulo UNICO fallaba ahi (falso negativo, verificado con
    # full_reduce real: 0.0% igual que sin pesos). Se reemplaza por
    # `zz_cost_operator_motif_fraction`, que es angulo-agnostico (detecta
    # el motivo estructural CX;RZ;CX sin importar el valor de theta) --
    # ver `_zz_cost_operator_motif_fraction` arriba para el detalle y su
    # limitacion declarada (escaneo de tripletes contiguos, no un analisis
    # de conmutacion completo).
    # CORRECCION (esta sesion, tras probar contra un circuito REAL derivado
    # del experimento de utilidad cuantica de IBM, Kim et al. 2023, Nature):
    # las compuertas de grado_maximo>=4.5 y spectral_sparsifiability>=0.08
    # eran heuristicas de FORMA de grafo, heredadas de cuando el detector
    # solo miraba "se parece a QAOA denso" -- pero el mecanismo real
    # (capa diagonal en Z, fundamentado en teoria de polinomios de fase,
    # ver docstring arriba) NO depende de que el grafo sea denso. Un
    # circuito heavy-hex disperso (grado maximo 3, el de IBM) tiene
    # zz_cost_operator_motif_fraction=1.0 y 0% de reduccion real medida
    # (verificado con full_reduce() directamente), pero las compuertas de
    # grado/dispersion antiguas lo dejaban pasar sin advertencia -- un
    # falso negativo real, no hipotetico. El motif_fraction YA ES la
    # señal suficiente y angulo-agnostica; se elimina el resto del
    # filtro compuesto.
    qaoa_like_pattern = p.get("zz_cost_operator_motif_fraction", 0) >= 0.8

    if tcount_fraction == 0:
        zx_ok, motivo = False, "T-count fraccional = 0 -- no hay margen no-Clifford que reducir (regla exacta, no probabilistica)"
    elif qaoa_like_pattern:
        zx_ok, motivo = ("ADVERTENCIA", "grado maximo alto + dispersion espectral moderada-alta -- patron similar a QAOA "
                          "(ganancia real medida = 0% en 24/24 instancias QAOA probadas). Confianza BAJA: "
                          "ninguna regla probada esta sesion distingue esto de forma confiable, se recomienda "
                          "verificar con full_reduce() directamente antes de decidir")
    else:
        zx_ok, motivo = True, ("valor por defecto (68% de las instancias no triviales de este proyecto se "
                                "benefician) -- NINGUNA feature probada esta sesion supera de forma honesta "
                                "este baseline; no se reporta mas confianza de la que hay")

    recs["zx_calculus"] = {
        "recomendada": zx_ok,
        "razon": f"{motivo} (tcount_frac={tcount_fraction*100:.1f}%, grado_max={p['max_degree']:.1f}, "
                 f"spectral_sparsifiability={p.get('spectral_sparsifiability', float('nan')):.3f}); "
                 f"SIEMPRE sube el 2q-count si se aplica -- validar con decide_regime() antes de usar en la ruta QEM",
    }

    recs["and_gate_descomputo_medicion"] = {
        "recomendada": p["ancilla_candidates"] > 0,
        "razon": f"{p['ancilla_candidates']} qubit(s) candidato(s) a ancilla temporal (heuristica: "
                 f"ultimo uso antes de la mitad del circuito) -- "
                 f"{'requiere verificar a mano que de verdad se descartan a |0>, no que se reutilizan (caso Cuccaro)' if p['ancilla_candidates'] > 0 else 'no hay candidatos -- probablemente no aplica'}",
    }

    recs["qcp_medicion_intermedia_general"] = {
        "recomendada": p["qcp_known_fraction_at_probe"] > 0,
        "razon": f"fraccion rastreable a presupuesto modesto (n_pcm=4) = {p['qcp_known_fraction_at_probe']*100:.0f}% "
                 f"({'hay subconjuntos desacoplados que explotar' if p['qcp_known_fraction_at_probe'] > 0 else 'el circuito entrelaza el registro completo por diseno -- sin oportunidad parcial'})",
    }

    # ADVERTENCIA NUEVA (esta sesion, motivada directamente por Tindall et
    # al. 2023 vs. Kim et al. 2023): a diferencia de las recomendaciones de
    # arriba (que dicen que TECNICA de reduccion usar), esta senal no
    # recomienda una tecnica -- adviert que el circuito, TAL COMO ESTA,
    # podria no necesitar QEC/QEM en absoluto, porque podria ser simulable
    # clasicamente via redes tensoriales (el mismo atajo que Tindall uso
    # contra el circuito heavy-hex de IBM, que ninguno de nuestros
    # estimadores -- ni el propio ni el de Azure -- pregunta jamas). Se
    # declara explicitamente de BAJA CONFIANZA: es una biparticion ingenua
    # (mitad/mitad por indice), no el corte optimo, y no es una prueba de
    # bond dimension real. Umbral (50%) elegido por ser el punto medio
    # declarado en `entanglement_growth.py`, no calibrado con un dataset
    # etiquetado -- se re-evaluara si aparece evidencia de que otro umbral
    # separa mejor casos clasicamente simulables de casos genuinamente
    # dificiles.
    frac = p.get("final_state_entanglement_fraction")
    if frac is None:
        atajo_clasico_msg = (f"no evaluado (circuito de {p['num_qubits']} qubits, por encima del limite "
                              f"practico de {_MAX_QUBITS_FOR_EXACT_ENTANGLEMENT_PROBE} qubits para statevector "
                              f"exacto, o menos de 2 qubits) -- sin senal")
        atajo_clasico_flag = None
    elif frac < 0.5:
        atajo_clasico_msg = (f"entropia de entrelazamiento bipartita final = {frac*100:.1f}% del maximo teorico "
                              f"(corte ingenuo mitad/mitad) -- POSIBLE atajo clasico (compresion tipo red "
                              f"tensorial/bond dimension bajo), el mismo mecanismo que Tindall et al. exploto "
                              f"contra el circuito heavy-hex de Kim et al. 2023. NO VERIFICADO: solo una senal "
                              f"de confianza BAJA (biparticion ingenua, no el corte optimo) -- se recomienda "
                              f"verificar con una libreria de redes tensoriales real antes de asumir que el "
                              f"circuito requiere QEC/FTQC")
        atajo_clasico_flag = "ADVERTENCIA"
    else:
        atajo_clasico_msg = (f"entropia de entrelazamiento bipartita final = {frac*100:.1f}% del maximo teorico "
                              f"-- consistente con ley de volumen (dificil de comprimir clasicamente via redes "
                              f"tensoriales de bond dimension acotada), aunque tampoco esto es una prueba formal")
        atajo_clasico_flag = False

    recs["posible_atajo_clasico_bond_dimension"] = {
        "recomendada": atajo_clasico_flag,
        "razon": atajo_clasico_msg,
    }

    return p, recs


def print_report(qasm_path, label):
    p, recs = recommend_techniques(qasm_path)
    print(f"\n{'='*90}\n{label}  ({qasm_path})\n{'='*90}")
    print(f"  qubits={p['num_qubits']}  compuertas={p['num_gates']}  T-count={p['tcount']}  "
          f"2q-count={p['two_qubit_count']}  grado_prom={p['avg_degree']:.1f}  grado_max={p['max_degree']}  "
          f"ancillas_candidatas={p['ancilla_candidates']}")
    for tech, r in recs.items():
        marca = "SI" if r["recomendada"] else "NO"
        print(f"  [{marca:>2s}] {tech:35s}  {r['razon']}")


if __name__ == "__main__":
    circuitos = {
        "QFT10 (fase controlada)": "qft10.qasm",
        "Sumador8 (Clifford+T, Cuccaro)": "adder8.qasm",
        "QAOA/farm (grafo denso)": "qaoa_farm.qasm",
        "random8 (sin estructura)": "random_circuit8.qasm",
    }
    for label, path in circuitos.items():
        print_report(path, label)
