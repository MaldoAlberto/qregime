# ZX-calculus (reduccion de T-count)

Codigo: `zx_reduction.py` (wrapper de `pyzx.full_reduce`), `build_circuit_dataset.py`
+ `build_circuit_dataset_v2.py` (98 circuitos reales, 9 familias), `pre_vs_post_zx_check.py`
(chequeo barato de gadgets alineados vs `full_reduce` completo).

**Resultado principal**: ganancia real medida por familia en `../../benchmarks/README.md`.
**Corregido tras re-verificacion**: ninguna feature probada (incluyendo
`spectral_sparsifiability`) supera de forma honesta (leave-one-family-out,
98 instancias) al baseline trivial de asumir "SI ayuda" por defecto
(70.4% ambos -- una version anterior de este documento citaba 82.9% para
spectral_sparsifiability, cifra que no se reprodujo en una re-verificacion
limpia y se corrige aqui). Ver `../../benchmarks/README.md` para la tabla
completa de metodos probados.

Falla completamente en QAOA (0% de ganancia real en las 24 instancias
probadas, sin excepcion, explicado por el teorema de fusion de gadgets de
fase de Kissinger & van de Wetering, arXiv:1903.10477 -- los gadgets de
QAOA no comparten soporte exacto de qubits). Ningun metodo probado esta
sesion (incluyendo costo de contraccion de red tensorial, ver
`../tensor_network_simulability/`) distingue este caso de forma confiable
-- se maneja en `../../decision-layer/technique_advisor.py` como una
advertencia de baja confianza, no como una clasificacion binaria.

`phase_polynomial_parity_check.py` **NO se usa en produccion**: es una
reimplementacion propia del algebra GF(2) de polinomios de fase que, al
verificarse con un chequeo de equivalencia unitaria real, resulto
incorrecta (diferencia maxima 1.35 entre el circuito original y el
"optimizado") por compuertas Clifford en base X intercaladas. Se deja en
el repositorio como registro honesto de un intento que no funciono, no
como una herramienta recomendada -- usar `full_reduce()` de pyzx
directamente en su lugar.
