"""
La otra mitad de la propuesta del usuario (lattice + minima energia):
representar cada circuito como una RED TENSORIAL real (un tensor por
compuerta, un indice por segmento de cable entre compuertas consecutivas
sobre el mismo qubit) y usar un optimizador real de orden de contraccion
(cotengra) para medir que tan barato es simular el circuito clasicamente.

Por que esto es literalmente "buscar la minima energia" sobre una
estructura tipo lattice: encontrar el orden de contraccion optimo de una
red tensorial es un problema NP-duro de particionado de hipergrafos, y
cotengra lo resuelve por defecto con SIMULATED ANNEALING (el mismo
algoritmo de optimizacion combinatoria que se uso en el experimento de
Ising/MaxCut de este proyecto) -- aqui el "costo" que se minimiza es el
numero de operaciones de punto flotante (flops) o el ancho de contraccion
(el tamano del tensor intermedio mas grande, analogo directo al treewidth
del grafo de interaccion).

Hipotesis a probar: si un circuito es "tipo lattice" (grafo de interaccion
disperso, poco denso -- lo que spectral_sparsifiability ya mide), deberia
ser barato de contraer (ancho de contraccion bajo). Si eso se confirma,
spectral_sparsifiability y el ancho de contraccion real deberian estar
correlacionados -- lo cual seria evidencia de que ambos miden la MISMA
propiedad estructural subyacente, solo por vias distintas (algebra
espectral vs. optimizacion combinatoria de contraccion).

Limitacion declarada: el costo de contraccion depende del ORDEN en que
cotengra decide contraer, que es un HEURISTICO (annealing con repeticiones
acotadas), no el optimo global garantizado -- se reporta el mejor
encontrado en `max_repeats` intentos, no el verdadero minimo NP-duro.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import cotengra as ctg
from qiskit import QuantumCircuit, transpile

REAL_BASIS = ["cz", "rz", "sx", "x"]


def to_real_basis(qc: QuantumCircuit):
    return transpile(qc, basis_gates=REAL_BASIS, optimization_level=1, seed_transpiler=0)


def circuit_to_tensor_network(qc: QuantumCircuit):
    """
    Construye la estructura de indices de la red tensorial (SIN valores
    numericos -- cotengra solo necesita la estructura de conexion para
    buscar el orden de contraccion, no los tensores reales, exactamente
    igual a como opt_einsum/cotengra se usan en produccion para PLANEAR
    una contraccion antes de ejecutarla).

    Cada compuerta de 1 o 2 qubits (post-transpile a REAL_BASIS, que solo
    tiene compuertas de 1-2 qubits) se vuelve un tensor cuyos indices de
    entrada son el "cable" actual de cada qubit que toca, y cuyos indices
    de salida son cables NUEVOS (la compuerta re-etiqueta el cable).

    Entrada y salida se FIJAN (|0...0> de entrada, un bitstring fijo de
    salida) dandoles dimension 1 a esos indices de frontera -- esto mide
    el costo de UNA amplitud de transicion, la metrica estandar usada en
    la literatura de "quantum advantage"/supremacia para caracterizar que
    tan dificil es simular clasicamente un circuito (Boixo et al. 2018,
    y los benchmarks de Google Sycamore usan exactamente esta cantidad).
    """
    n = qc.num_qubits
    counters = [0] * n
    inputs = []
    initial_idx = [f"q{q}_0" for q in range(n)]

    for instr in qc.data:
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        idx_group = []
        for q in qubits:
            in_idx = f"q{q}_{counters[q]}"
            counters[q] += 1
            out_idx = f"q{q}_{counters[q]}"
            idx_group.extend([in_idx, out_idx])
        inputs.append(tuple(idx_group))

    final_idx = [f"q{q}_{counters[q]}" for q in range(n)]

    size_dict = {}
    for group in inputs:
        for idx in group:
            size_dict[idx] = 2
    # fijar frontera (entrada |0...0>, salida a un bitstring fijo) a
    # dimension 1 -- una amplitud de transicion especifica, no el estado
    # completo
    for idx in initial_idx + final_idx:
        size_dict[idx] = 1

    output = ()  # escalar: una sola amplitud
    return inputs, output, size_dict


def measure_contraction_cost(qc: QuantumCircuit, max_repeats=24, seed=0):
    real_qc = to_real_basis(qc)
    inputs, output, size_dict = circuit_to_tensor_network(real_qc)

    opt = ctg.HyperOptimizer(minimize="flops", max_repeats=max_repeats,
                              progbar=False, parallel=False)
    tree = opt.search(inputs, output, size_dict)

    return {
        "num_qubits": real_qc.num_qubits,
        "num_tensors": len(inputs),
        "log2_flops": float(np.log2(max(tree.contraction_cost(), 1))),
        "contraction_width": float(tree.contraction_width()),  # log2 del tensor intermedio mas grande
    }


def run():
    import sys
    sys.path.insert(0, ".")
    from build_circuit_dataset import generate_circuits as gen1
    from build_circuit_dataset_v2 import generate_circuits as gen2
    import re

    # una instancia representativa por familia, de tamano moderado y
    # comparable (no las 98 -- esto es una prueba de concepto de la
    # metrica, no un barrido exhaustivo; se declara asi)
    targets = {
        "QFT": "QFT_n8", "Adder": "Adder_n5",
        "QAOA": "QAOA_n8_d0.4", "Random": "Random_n8_depth16",
        "GHZ": "GHZ_n8", "VQEansatz": "VQEansatz_n8_r2",
        "Grover": "Grover_n5", "ToffoliChain": "ToffoliChain_n8",
        "QPE": "QPE_ncount5",
    }

    circuits = dict(gen1())
    circuits.update(dict(gen2()))

    rows = []
    print(f"{'familia':14s} {'circuito':20s} {'qubits':>7s} {'tensores':>9s} "
          f"{'log2(flops)':>12s} {'ancho_contraccion':>18s}")
    for fam, name in targets.items():
        if name not in circuits:
            print(f"{fam:14s} {name:20s}  SALTADO (no encontrado en el dataset)")
            continue
        qc = circuits[name]
        try:
            r = measure_contraction_cost(qc, max_repeats=24, seed=hash(name) % 2**31)
            r["family"] = fam
            r["name"] = name
            rows.append(r)
            print(f"{fam:14s} {name:20s} {r['num_qubits']:7d} {r['num_tensors']:9d} "
                  f"{r['log2_flops']:12.2f} {r['contraction_width']:18.2f}")
        except Exception as e:
            print(f"{fam:14s} {name:20s}  ERROR ({type(e).__name__}: {e})")

    with open("tensor_network_contraction_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} circuitos analizados, guardados en tensor_network_contraction_results.json")
    return rows


if __name__ == "__main__":
    run()
