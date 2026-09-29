# QCP acotado + extension RQAOA-style

Codigo: `qcp_bounded_tracker.py` (rastreo de estado bajo presupuesto de
qubits `n_pcm`) y `rqaoa_style_circuit_reduction.py` (extension: en vez de
abandonar el rastreo cuando un grupo excede el presupuesto, fija el qubit
mas confiable -- el analogo directo de RQAOA -- y cuenta cuantas "rondas"
hacen falta).

**Metrica nueva sin validar externamente**: numero de rondas por familia
(20-33 para Sumador8/QAOA, bajando con presupuesto mayor). Verificada solo
con un caso de sanidad (GHZ-3), no contra ningun trabajo publicado
equivalente -- se reporta como exploratoria.
