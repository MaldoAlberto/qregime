# AND-gate + descomputo por medicion

Codigo: `verify_and_uncompute.py`. Requiere qubits ancilla candidatos
reales (heuristica: ultimo uso antes de la mitad del circuito).

**Limite declarado**: la heuristica no verifica automaticamente que la
ancilla realmente vuelva a |0> en vez de reutilizarse -- el caso del
sumador de Cuccaro mostro esto explicitamente y requiere revision manual.
