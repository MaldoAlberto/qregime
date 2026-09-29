# Teletransportacion de compuertas RZZ

Codigo: `case_study.py`. Mide el tiempo esperado a respuesta confiable en
QEM (shot_time x 1/fidelidad^2) contra un backend real, usando tasas de
error de 2 qubits reales del circuito transpileado.

**Umbral calibrado con datos de esta sesion**: grado promedio del grafo de
interaccion < 15 -> ayuda en 25/27 corridas probadas; el unico caso
negativo real conocido (MANN-a9) tiene grado ~40. No es un umbral generico
de la literatura, se declara asi.
