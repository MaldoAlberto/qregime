# Guia de Validacion de Tecnicas: QEM, QEC y FTQC

Este repositorio es un estudio experimental (no un producto terminado) que
mide, con codigo real y datos reales, en que punto estan hoy siete tecnicas
de reduccion/optimizacion de circuitos cuanticos, y como se pueden combinar
con una capa de decision QEM-vs-FTQC bajo presupuesto de qubits.

La inspiracion es el trabajo de benchmarking abierto de Unitary Foundation
(Metriq): un lugar donde se puede ver, por tecnica y por tipo de problema,
que tan bien funciona un metodo -- pero aqui los numeros vienen de
experimentos propios, ejecutados y verificados en esta sesion, no de
resultados recopilados de la comunidad. Es un punto de partida, no un
reemplazo de Metriq.

## Como esta organizado

- **`techniques/`** -- una carpeta por tecnica de reduccion/optimizacion
  (ZX-calculus, teletransportacion RZZ, AND-gate + descomputo por medicion,
  QCP acotado + extension RQAOA-style, circuit cutting, reproduccion de
  correlacion RQAOA, ruteo SwapStrategy vs SabreSwap). Cada una con su
  codigo y su propio README de alcance y limites.
- **`qec/`** -- simulacion real de un codigo de superficie rotado (Stim +
  PyMatching), con la curva de umbral medida por Monte Carlo. Esto es
  DISTINTO de usar Azure QRE como estimador de recursos: aqui se genera y
  se decodifica un codigo de correccion de errores de verdad.
- **`decision-layer/`** -- `technique_advisor.py` (recomienda que tecnica
  usar segun el perfil del circuito) y `pipeline_v2.py` (decide QEM vs
  FTQC con tiempo real de un backend y factibilidad de qubits via Azure
  QRE).
- **`benchmarks/`** -- la guia de validacion misma: tabla y heatmap de
  tecnica x familia de circuito x metrica medida x confianza, con el
  metodo de prueba honesto (leave-one-family-out, no CV con fuga).
- **`papers/`** -- las referencias publicadas contra las que se contrasto
  cada hallazgo, y que confirman o no confirman.
- **`educational/`** -- el paper (PDF/LaTeX) y un diagrama interactivo del
  pipeline completo.

## Que declarar honestamente sobre el alcance

- Las cifras de precision reportadas son **leave-one-family-out**, el
  unico metodo de validacion que se considero confiable en este proyecto
  (un CV aleatorio inicial dio cifras infladas por fuga de datos entre
  variantes de la misma familia -- corregido y documentado, no escondido).
- **Correccion importante**: una version anterior de este repositorio
  reportaba 82.9% de precision LOFO para `spectral_sparsifiability`. Una
  re-verificacion limpia con el dataset completo (98 instancias) dio
  70.4% -- identico a un baseline trivial ("asumir que SI ayuda por
  defecto"). Ninguna feature probada esta sesion (spectral, costo de
  contraccion tensorial, razon de angulos, combinaciones) supera ese
  baseline de forma honesta. Ver `benchmarks/README.md` para la tabla
  completa.
- Hay **un caso estructural sin resolver**: circuitos tipo QAOA (rotaciones
  ZZ independientes sobre un grafo de problema) dan 0% de ganancia real de
  ZX-calculus en las 24 instancias probadas, y ningun predictor probado
  (regla a mano, Random Forest, dispersion espectral, costo de contraccion
  tensorial) lo distingue bien. Ademas, QAOA p=1/2/3 optimizado NUNCA
  supera a un simulated annealing clasico simple en los 27 grafos
  probados (`techniques/qaoa_ising_ground_state/`). Se documenta como
  limite conocido, no se esconde.
- La pieza de QEC es **simulacion clasica exacta de un codigo
  estabilizador** (Stim + PyMatching), con un modelo de ruido depolarizado
  uniforme -- ya conectada a `decision-layer/pipeline_v2.py` como
  validacion cruzada del modelo analitico de Azure QRE (ver
  `decision-layer/README.md`), aunque con resolucion estadistica limitada
  para tasas de error logico muy bajas.
