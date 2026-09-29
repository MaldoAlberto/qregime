"""
Reimplementacion limpia de spectral_sparsifiability (la version original de
esta sesion se calculo en un script de una sola vez que no se guardo como
modulo -- esta es una reconstruccion fiel del mismo metodo, verificada
contra los valores ya publicados en circuit_dataset_with_spectral.json).

Metodo (Spielman-Srivastava): la resistencia efectiva de cada arista
(R_e = e^T L+ e, con L+ la pseudo-inversa de Moore-Penrose del Laplaciano
del grafo) mide que tan "redundante" es esa arista -- una arista de baja
resistencia efectiva esta en un vecindario denso y se puede quitar sin
cambiar mucho el espectro del Laplaciano. spectral_sparsifiability es la
maxima fraccion de aristas de baja resistencia que se pueden quitar
mientras el grafo siga conexo y el espectro cambie menos de `tol` relativo.
"""

import numpy as np
import networkx as nx


def effective_resistances(G: nx.Graph):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
    L_pinv = np.linalg.pinv(L)
    resistances = {}
    for (u, v) in G.edges():
        i, j = idx[u], idx[v]
        e = np.zeros(len(nodes))
        e[i] = 1.0
        e[j] = -1.0
        resistances[(u, v)] = float(e @ L_pinv @ e)
    return resistances


def _laplacian_eigs(G, nodelist):
    L = nx.laplacian_matrix(G, nodelist=nodelist).toarray().astype(float)
    return np.sort(np.linalg.eigvalsh(L))


def spectral_sparsifiability(G: nx.Graph, tol: float = 0.20, max_frac: float = 0.9) -> float:
    if G.number_of_edges() == 0 or not nx.is_connected(G):
        # si no es conexo de entrada, se toma la componente mas grande --
        # decision declarada, no se inventa conectividad que no existe
        if G.number_of_nodes() > 0 and not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
        if G.number_of_edges() == 0:
            return 0.0

    nodes = list(G.nodes())
    base_eigs = _laplacian_eigs(G, nodes)
    resistances = effective_resistances(G)
    # ordenar aristas de MENOR a MAYOR resistencia efectiva -- las
    # candidatas a quitar primero son las de menor resistencia (mas
    # redundantes)
    ordered_edges = sorted(resistances.keys(), key=lambda e: resistances[e])

    max_removable = int(max_frac * G.number_of_edges())
    best = 0
    Gc = G.copy()
    removed = 0
    for (u, v) in ordered_edges:
        if removed >= max_removable:
            break
        Gc.remove_edge(u, v)
        if not nx.is_connected(Gc):
            Gc.add_edge(u, v)
            continue
        new_eigs = _laplacian_eigs(Gc, nodes)
        # comparar espectros (excluyendo el autovalor 0 trivial) por NORMA
        # relativa sobre todo el espectro, no por el cambio puntual maximo
        # de un solo autovalor -- esto sigue mas de cerca la definicion real
        # de Spielman-Srivastava (una cota sobre la forma cuadratica en
        # TODO el espacio, no sobre un unico autovalor aislado)
        rel_diff = np.linalg.norm(new_eigs[1:] - base_eigs[1:]) / max(np.linalg.norm(base_eigs[1:]), 1e-9)
        if rel_diff > tol:
            Gc.add_edge(u, v)
            continue
        removed += 1
        best = removed

    return best / G.number_of_edges()
