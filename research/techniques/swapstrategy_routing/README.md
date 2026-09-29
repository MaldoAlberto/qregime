# SwapStrategy vs SabreSwap (ruteo)

Codigo: `swapstrategy_density_sweep.py`. Compara SWAP-count real segun
densidad del grafo de interaccion.

**Resultado**: SwapStrategy es constante (~120 SWAPs) en todo el barrido
(densidad 0.15-1.00); SabreSwap va de 18 a 274 SWAPs y luego BAJA a 132 en
densidad=1.0 -- una anomalia real, reproducida en dos corridas
independientes con grafos distintos, y todavia sin explicacion.

**Dos bugs de conteo reales, corregidos en esta sesion**: basis_gates
forzado y decompose(reps>=1) sintetizaban SWAP->3xCX antes de contar,
escondiendo el conteo real de SWAPs de ambos metodos por separado.
