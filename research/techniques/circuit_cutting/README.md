# Circuit cutting (particionado real)

Codigo: `circuit_cutting_experiment.py`, usando `qiskit_addon_cutting.find_cuts`
real (no simulado). Resultados medidos: Sumador8 = 2 cortes / 256x overhead
de muestreo QPD; QAOA/farm = 18-44 cortes / 1e17-1e43x overhead segun el
tamano del subcircuito permitido.

**Nota de implementacion**: el parametro `qubits_per_subcircuit` afecta
criticamente el resultado -- un valor muy restrictivo da muchos mas cortes
de los estructuralmente necesarios (bug real encontrado y corregido en esta
sesion).
