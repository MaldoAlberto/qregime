"""
treewidth_simulability.py -- certificado de simulabilidad clasica FUNDAMENTADO
EN TEORIA (no un proxy heuristico como `entanglement_growth.py` /
`final_state_entanglement_fraction`), motivado por el mismo punto debil que
esos dos: nuestros estimadores de recursos nunca preguntan si el circuito
tiene estructura clasicamente explotable, el mismo mecanismo que Tindall et
al. exploto contra el circuito heavy-hex de Kim et al. 2023.

DIFERENCIA CLAVE frente a la metrica de entropia bipartita ya implementada:
la entropia de un corte ingenuo (mitad/mitad) es un PROXY empirico -- un
valor bajo es evidencia sugestiva de compresibilidad, pero no una garantia
formal (el corte podria simplemente no ser el que revela la estructura
tipo-arbol), y un valor alto no prueba dureza clasica.

Este modulo usa en cambio un resultado FORMAL y ya publicado, con un
ALCANCE que se declara con precision (evitar sobre-reclamar es tan
importante aqui como el resultado mismo):

  Teorema (Markov & Shi, SIAM J. Comput. 2008, "Simulating Quantum
  Computation by Contracting Tensor Networks"): si el grafo de un circuito
  cuantico (vertices = compuertas/tensores, no solo qubits) tiene treewidth
  <= w, existe una estrategia de contraccion que lo simula en tiempo
  poly(tamano del circuito) * 2^O(w).

  ALCANCE EXACTO de lo que se calcula aqui (para no sobre-reclamar): el
  treewidth que este modulo calcula es del GRAFO DE INTERACCION ESTATICO
  (qubits como vertices, una arista por cada PAR de qubits que comparte
  alguna compuerta de 2 qubits en algun momento -- la topologia,
  colapsada en el tiempo, no el grafo temporal completo que usa el
  teorema original). Esto es exacto y suficiente SOLO para acotar el
  costo de una unica capa (profundidad O(1)) sobre esa topologia -- por
  ejemplo, para comparar que tan favorable es la topologia de un
  hardware (heavy-hex vs. all-to-all) independientemente de cuantos pasos
  de Trotter se corran despues. NO es, por si solo, una cota valida del
  costo de simular el circuito COMPLETO cuando la profundidad crece: la
  fisica real (y la razon de que Tindall necesitara belief propagation, no
  solo mirar la topologia) es que el entrelazamiento puede crecer con la
  profundidad incluso sobre una topologia de treewidth bajo, hasta
  saturar el limite de ley de volumen -- exactamente lo que
  `entanglement_growth.py` / `final_state_entanglement_fraction` miden.
  Por eso este certificado es COMPLEMENTARIO, no un reemplazo: treewidth
  estatico bajo es una condicion NECESARIA (estructural, topologica,
  demostrada) para que exista un atajo tensorial barato; que la entropia
  de entrelazamiento no sature ese techo al aumentar la profundidad es la
  evidencia adicional (empirica, no una prueba formal completa) de que el
  atajo se mantiene. Ver la seccion "Teoria nueva" del paper para el
  argumento completo con ambas piezas combinadas.

Por que esto es mas fuerte epistemicamente que la entropia bipartita: el
treewidth EXACTO es NP-dificil de calcular, asi que aqui se usan dos
heuristicas de aproximacion ya en networkx (`treewidth_min_degree`,
`treewidth_min_fill_in`), que devuelven cada una una DESCOMPOSICION EN
ARBOL VALIDA de algun ancho w_heuristico >= w_optimo. Esto importa en una
sola direccion, pero de forma rigurosa: SI una heuristica encuentra una
descomposicion de ancho w, el Teorema de Markov-Shi garantiza formalmente
que existe una estrategia de simulacion de costo ~2^w -- una garantia
real, no un proxy. Lo que NO se puede concluir es lo contrario: si ambas
heuristicas devuelven un ancho alto, eso NO prueba que el circuito sea
dificil (el treewidth optimo real podria ser mucho menor; las heuristicas
solo acotan por arriba). Esta asimetria se declara explicitamente en el
codigo y en el reporte: un treewidth bajo ENCONTRADO es una prueba
positiva utilizable hoy mismo; un treewidth alto encontrado es solo
"no se encontro un atajo con estas dos heuristicas baratas", no una prueba
de dureza.

LIMITACION DECLARADA (la misma honestidad que el resto del proyecto):
"grafo de interaccion" (que arista existe, no cuantas veces ni en que
orden) es una abstraccion de Markov-Shi para la COTA DE COSTO DE
CONTRACCION; no captura la estructura temporal del circuito (que
compuertas conmutan, orden real de aplicacion), asi que la cota de
2^treewidth es conservadora (una cota superior valida, pero podria no ser
ajustada). Tampoco se calcula aqui el treewidth EXACTO (NP-dificil): se
reporta el MEJOR (menor) ancho encontrado por las dos heuristicas
disponibles, un limite superior valido del treewidth real.
"""
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree, treewidth_min_fill_in

from .hypergraph import CircuitHypergraph
import pyzx as zx


def treewidth_simulability_certificate(qasm_path: str, verbose: bool = True) -> dict:
    circuit = zx.Circuit.from_qasm_file(qasm_path).to_basic_gates()
    hg = CircuitHypergraph(circuit)
    ig = hg.interaction_graph()
    n = ig.number_of_nodes()

    # dos heuristicas de aproximacion (cada una da una descomposicion en
    # arbol VALIDA, por lo que cada ancho devuelto es una cota superior
    # legitima del treewidth real -- se toma el mejor (menor) de las dos)
    w_min_degree, _ = treewidth_min_degree(ig)
    w_min_fill_in, _ = treewidth_min_fill_in(ig)
    best_w = min(w_min_degree, w_min_fill_in)

    # cota de Markov & Shi 2008, ALCANCE DECLARADO ARRIBA: costo de
    # contraccion de UNA CAPA (profundidad O(1)) sobre esta topologia
    # ~ 2^w, independiente del numero de qubits -- se reporta 2^w como el
    # "bond dimension" de esa cota estructural (no una medicion; una cota
    # formal, pero solo por-capa, no del circuito completo a profundidad
    # arbitraria)
    bond_dimension_bound = 2 ** best_w

    # umbral declarado para "condicion estructural favorable": un ancho
    # <= 20 da una cota por-capa <= 2^20 (~1M) -- no calibrado contra un
    # dataset etiquetado, es un umbral de orden de magnitud declarado,
    # igual que el 50% de la metrica de entropia
    certificate_positive = best_w <= 20

    result = {
        "qasm_path": qasm_path,
        "num_qubits_in_interaction_graph": n,
        "treewidth_upper_bound_min_degree_heuristic": w_min_degree,
        "treewidth_upper_bound_min_fill_in_heuristic": w_min_fill_in,
        "best_treewidth_upper_bound": best_w,
        "bond_dimension_bound_2^w": bond_dimension_bound,
        "classical_simulation_certificate": (
            "CONDICION ESTRUCTURAL FAVORABLE (demostrada por capa, Markov & Shi 2008): "
            f"la topologia admite una estrategia de contraccion de costo ~2^{best_w} = "
            f"{bond_dimension_bound:,} por capa de profundidad O(1), independiente del "
            "numero de qubits -- NECESARIA pero no suficiente por si sola para circuitos "
            "profundos: combinar con `final_state_entanglement_fraction` para ver si el "
            "entrelazamiento real satura este techo al crecer la profundidad" if certificate_positive else
            "SIN CONDICION FAVORABLE ENCONTRADA con estas heuristicas (no es una prueba de "
            "dureza -- el treewidth real podria ser menor; solo dice que estas dos "
            "heuristicas baratas no hallaron una descomposicion de ancho <=20)"
        ),
    }
    if verbose:
        print(f"  {qasm_path}: treewidth estatico (mejor de 2 heuristicas) = {best_w}  "
              f"(min_degree={w_min_degree}, min_fill_in={w_min_fill_in})  "
              f"-> cota por-capa = 2^{best_w} = {bond_dimension_bound:,}  "
              f"-> {'condicion estructural favorable' if certificate_positive else 'sin condicion favorable (no concluyente)'}")
    return result


if __name__ == "__main__":
    circuits = [
        "adder8.qasm", "grover.qasm", "qft10.qasm", "qft6.qasm",
        "random_circuit8.qasm", "shor_like_sequential.qasm",
        "kicked_ising_analog_16q.qasm", "kicked_ising_analog_16q_deep.qasm",
        "qaoa_farm.qasm", "qaoa_small_test.qasm", "vqe_ansatz.qasm",
    ]
    print("Condicion estructural (topologica, por-capa) de simulabilidad clasica via\n"
          "treewidth del grafo de interaccion estatico (Markov & Shi 2008) -- NECESARIA\n"
          "pero no suficiente por si sola para circuitos profundos; complementa (no\n"
          "reemplaza) la entropia de entrelazamiento bipartita ya implementada:\n")
    results = {}
    for c in circuits:
        try:
            r = treewidth_simulability_certificate(c)
            results[c] = r
        except FileNotFoundError:
            print(f"  {c}: no encontrado, se omite")

    print("\n\nResumen ordenado por treewidth (menor = condicion estructural mas favorable):")
    for c, r in sorted(results.items(), key=lambda kv: kv[1]["best_treewidth_upper_bound"]):
        w = r["best_treewidth_upper_bound"]
        flag = "condicion favorable" if w <= 20 else "no concluyente"
        print(f"  {c:40s} treewidth<={w:3d}  2^w={2**w:>15,}  {flag}")
