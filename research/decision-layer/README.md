# Capa de decision: technique_advisor + pipeline_v2 (decide_regime)

- **`technique_advisor.py`**: perfila un circuito (grado del grafo de
  interaccion, tcount_frac, ancillas candidatas, fraccion QCP rastreable,
  spectral_sparsifiability) y recomienda SI/NO/ADVERTENCIA por tecnica.
  Regla de ZX-calculus vigente (actualizada y corregida esta sesion):
  filtro exacto tcount_frac==0 -> NO; patron QAOA (grado maximo alto +
  dispersion espectral moderada-alta) -> ADVERTENCIA de baja confianza;
  cualquier otro caso -> SI por defecto. Precision honesta (LOFO, 98
  instancias): 70.4%, identica al baseline trivial -- ver
  `../benchmarks/README.md` para la correccion completa (una version
  anterior de este documento citaba 82.9%, que no se reprodujo).
- **`pipeline_v2.py`**: decide QEM vs FTQC comparando tiempo real de QEM
  (shot_time x 1/fidelidad^2, con tasas de error reales de un backend) vs
  runtime + factibilidad de qubits de Azure QRE (FTQC). Ya usa calibracion
  de hardware real -- no es una brecha pendiente. **Nuevo: ETAPA 0**, ver
  `layout_noise_search.py` mas abajo.
- **`ftqc_advisor.py` / `ftqc_resource_estimator.py`**: version anterior /
  auxiliar, mantenida por referencia.

**Brecha declarada y ya resuelta en `../qec/`**: hasta ahora, todo lo de
FTQC pasaba por Azure QRE como estimador de recursos, nunca por una
simulacion real de un codigo de correccion de errores. Eso ahora existe en
`../qec/qec_surface_code_experiment.py`, y ya esta conectado a
`pipeline_v2.py` (`cross_validate_logical_error_rate`).

## Nuevo: estimador FTQC 100% propio, sin depender de Azure QRE ni de NVIDIA

Pregunta que motivo esto: ¿que podemos hacer para crear y evaluar tecnicas
sin depender de las "expectativas" (estimaciones) de un vendor -- Azure QRE
es un modelo analitico de caja semi-cerrada, no auditable en detalle?

- **`fit_own_threshold_model.py`**: ajusta el modelo ESTANDAR y publico de
  umbral de codigos de superficie (Fowler et al. 2012, `p_L(d) = A *
  (p_fisico/p_th)^((d+1)/2)`) directamente a nuestros propios 27 puntos
  medidos con Stim+PyMatching (`../qec/qec_surface_code_results.json`) --
  la FORMULA es publica, las CONSTANTES (A, p_th) son nuestras, no
  copiadas de ningun paper ni de ningun vendor. Hallazgo honesto durante el
  ajuste: un primer intento sobre los 27 puntos completos dio R^2=0.28
  (malo) porque para p_fisico>=2% el error logico satura en ~0.50 sin
  importar la distancia (regimen de "moneda al aire", fuera del alcance del
  modelo). Restringiendo el ajuste a p_fisico<=1% (15 de 27 puntos, con
  peso por el error estandar binomial de cada punto), el ajuste da
  **p_th=0.740% +/- 0.009pp, A=0.0348, R^2=0.964** -- propio, reproducible,
  con su propia incertidumbre declarada.
- **`open_resource_estimator.py`**: usa ese ajuste + el conteo EXACTO de
  qubits fisicos por qubit logico que nuestra propia simulacion de Stim ya
  usa (formula empirica `2d^2+3d-1`, verificada contra
  `stim.Circuit.generated(...).num_qubits` para d=3,5,7,9,11) + un modelo
  de tiempo de ejecucion simplificado y declarado (compuertas T consumidas
  secuencialmente, sin optimizar fabricas de estados magicos como si hace
  Litinski 2019). Cero llamadas a Azure QRE o a ningun servicio externo.
- **`cross_validate_three_way.py`**: compara, sobre los 4 circuitos de
  referencia, TRES vias -- Azure QRE / nuestro estimador propio / medicion
  Monte Carlo real a la distancia que cada uno eligio. **Hallazgo real,
  reportado sin suavizar**: el runtime coincide razonablemente entre Azure
  y el estimador propio (1.1x-2.1x en los 4 circuitos probados), pero el
  conteo de QUBITS FISICOS difiere fuertemente (24x-46x, Azure siempre
  mayor). La causa mas probable, declarada explicitamente: el estimador
  propio NO modela fabricas de destilacion de estados magicos (que Azure
  QRE si incluye), asi que subestima qubits de forma sistematica -- el
  estimador propio sirve como chequeo de orden de magnitud de TIEMPO, no
  como reemplazo del conteo de qubits de Azure, y la decision de
  factibilidad de qubits en `evaluate_circuit.py` sigue viniendo de Azure
  QRE por esta razon, declarada, no escondida.
- **`evaluate_circuit.py`**: la herramienta de punta a punta -- alguien con
  un circuito `.qasm` lo pasa por aqui y recibe (1) que tecnicas de
  reduccion se recomiendan con su confianza honesta, (2) la decision QEM vs
  FTQC con Azure QRE, (3) el cruce de tres vias del estimador propio como
  chequeo independiente, y (4) un veredicto explicito de **"ventaja con
  senal real" vs "dentro del ruido de estimacion"**, con una regla simple y
  declarada: si la separacion QEM-vs-FTQC (>10x) es mayor que el desacuerdo
  ya medido entre estimadores (tipicamente 1-3x en runtime), la conclusion
  de regimen sobrevive a esa incertidumbre; si no, se marca honestamente
  como ruido, no como hallazgo. Probado en los 4 circuitos de referencia
  con `p_fisico` real medido de FakeTorino: en los 4 casos, Azure QRE
  reporta que FTQC **no es factible en qubits** con un presupuesto de
  `50 x qubits_del_backend` -- el veredicto en los 4 es "SIN VENTAJA FTQC
  POSIBLE HOY", no por diferencia de tiempo sino por qubits fisicos
  insuficientes con el hardware de referencia usado.

## Nuevo: ¿en que punto SI se puede aplicar FTQC? (`ftqc_crossover_sweep.py`)

Los 4 circuitos de referencia (T-count 56-135) nunca ganan con FTQC, ni
siquiera con hardware perfecto -- pero eso no responde la pregunta real,
que es EN QUE PUNTO deja de ser asi. Se corrio un barrido de tres variables
sobre los mismos 4 arquetipos reales, escalados (mismo tipo de circuito,
T-count real x10/x100/x1000, proxy honesto de "una instancia mas grande
del mismo algoritmo"): calidad de hardware (p_fisico de 1e-3 a 1e-6),
presupuesto de qubits fisicos (1,000 a 1,000,000), y escala del problema.

**Hallazgo principal, no intuitivo**: el cruce hacia una ventaja FTQC clara
(factible + >10x mas rapido que QEM) aparece de forma consistente entre
T-count ~1,000 y ~15,000 -- y en TODOS los arquetipos probados, ya ocurre
al p_fisico de HOY (<=0.1%, el mismo que medimos en FakeTorino), no en una
mejora de hardware hipotetica futura. Lo unico que falta en ese punto es
presupuesto de qubits (tipicamente 75,000-260,000 qubits fisicos). Es decir:
**a la calidad de hardware que ya existe, el cuello de botella para aplicar
FTQC es de ESCALA DE INTEGRACION (numero de qubits fisicos), no de calidad
de compuerta** -- una conclusion mas especifica y accionable que "depende
del hardware". Verificado explicitamente en el caso limite: a la escala de
nuestros circuitos reales de prueba, incluso con p_fisico=1e-6 y presupuesto
de 1,000,000 de qubits, FTQC es factible pero **8x MAS LENTO** que QEM
(QFT10: 1.74e-4s FTQC vs 2.08e-5s QEM) -- el overhead fijo de FTQC (rondas
del codigo) simplemente no se amortiza en un circuito tan chico, sin
importar que tan bueno sea el hardware.

Ver `ftqc_crossover_heatmap.png`: mapa de calor infactible / factible-sin-
ventaja / ventaja-clara, por arquetipo, escala del problema y presupuesto
de qubits, con p_fisico fijo al valor real medido hoy.

## Nuevo: la eleccion de FAMILIA DE CODIGO tambien importa (`code_family_comparison.py`)

Todo lo anterior (Stim, Azure QRE, nuestro estimador propio) asume codigo
de superficie -- la unica opcion realista para qubits superconductores de
conectividad local, y una de solo dos que la libreria de Azure QRE ofrece
(`surface_code`, `floquet_code`; verificado en el codigo -- no hay opcion
Steane). Un paper real y reciente (Perlin et al., "Fault-tolerant execution
of error-corrected quantum algorithms", arXiv:2603.04584, 2026) ejecuto
QAOA de verdad (no simulado) en hardware de iones atrapados (Quantinuum
H2/Helios) usando el codigo Steane [[7,1,3]], que si tiene conectividad
todos-con-todos disponible en esa arquitectura: **12 qubits logicos con
solo 97 qubits fisicos (8.1x)**, con una compuerta T logica de infidelidad
medida 2.6(4)e-3.

Se comparo ese dato REAL contra lo que nuestro pipeline (codigo de
superficie) estimaria para un circuito comparable (12 qubits logicos, con
un T-count EXTRAPOLADO -- no reportado por ellos -- desde la razon
T-count/compuertas-de-2q de nuestro propio circuito QAOA de referencia,
declarado como extrapolacion, no como su dato real):

| p_fisico | Azure QRE (superficie) | Estimador propio (superficie) | Steane real (Quantinuum) |
| --- | --- | --- | --- |
| 1e-3 | 79,750 qubits (d=11) | 2,256 qubits (d=9) | **97 qubits** |
| 1e-4 | 14,750 qubits (d=5)  | 312 qubits (d=3)   | **97 qubits** |

**Conclusion honesta**: incluso con nuestro estimador MAS CONSERVADOR
(que subestima qubits al no modelar fabricas de destilacion), un circuito
de este tamano necesitaria de 3x a 822x mas qubits fisicos bajo el
supuesto de codigo de superficie que lo demostrado con Steane en iones
atrapados. Esto NO dice que el codigo de superficie sea "peor" en
general -- el Steane requiere conectividad que los qubits superconductores
(donde Azure QRE y este pipeline asumen que se corre) no tienen -- dice que
**la eleccion de FAMILIA DE CODIGO, no solo la tecnica de reduccion de
circuitos, es una variable que este proyecto no habia explorado y que
cambia sustancialmente las conclusiones del barrido de cruce FTQC**. Se
declara aqui como una limitacion de alcance real (todo el proyecto asume
codigos de superficie) y como una extension natural: construir el mismo
tipo de estimador propio y auditable que ya se hizo para superficie, pero
para codigos concatenados tipo Steane, en cuanto haya mas de un punto
medido publicado para ajustar una curva real (hoy solo hay uno).

## Nuevo: ETAPA 0 -- buscar el layout/ruteo de MENOR ruido antes de estimar (`layout_noise_search.py`)

Probando un circuito real mas grande (16 qubits, T-count~3930, tipo Shor)
se encontro que el error fisico efectivo depende MUCHO de que enlaces del
backend termina usando el circuito despues de transpilar: en FakeTorino,
la razon entre el peor y el mejor enlace calibrado de 2 qubits es de
**669x** (0.15% a 100% de error, sobre 300 enlaces). Con la configuracion
de transpilador que este pipeline usaba antes (una sola semilla,
`optimization_level=1`), el mismo circuito daba p_fisico=0.517% -- muy
cerca de nuestro umbral ajustado (0.740%), lo que dispara la distancia de
codigo necesaria: **4,425,210 qubits fisicos** segun Azure QRE.

**Primer intento, que fallo honestamente**: elegir a mano la region de 16
qubits con menor error de enlace PROMEDIO (se probaron 40 regiones
conectadas al azar, con una dispersion de 0.40% a 21.7% de error interno
promedio) y forzarla como `initial_layout`. Resultado: **peor**, no mejor
(2,071,530 qubits) -- una region bien calibrada en promedio puede tener
una topologia que no encaja con el grafo de interaccion del circuito,
forzando mas SWAPs, y esos SWAPs adicionales pesan mas que la ganancia de
calibracion. La calibracion de una region NO predice el error real una
vez que se rutea el circuito ahi.

**Lo que si funciona**: `layout_noise_search.py` prueba, con el
transpilador REAL (`optimization_level=3`, que ya activa SabreLayout con
refinamiento iterativo), N semillas distintas (20 por defecto, ~7 segundos
de costo total) y mide el error fisico efectivo de CADA una con
`real_two_qubit_error_rate` (la misma funcion que el resto del proyecto ya
usa) -- quedandose con la mejor. Para el circuito de prueba, esto bajo el
error fisico de 0.517% a 0.320%, y combinado con la reduccion real de
T-count de ZX-calculus (39.2%, ver mas abajo), el resultado final fue
**470,970 qubits fisicos -- una reduccion de 9.4x** sin ninguna tecnica de
reduccion de circuito nueva, solo probando varias configuraciones reales
de layout+ruteo y quedandose con la mejor medida.

Ya integrado como **ETAPA 0** de `pipeline_v2.run_pipeline_v2()` (antes de
perfilar el circuito) y en `evaluate_circuit.py` (reemplaza la
transpilacion de una sola configuracion fija que se usaba antes para medir
p_fisico, tanto para Azure QRE como para el estimador propio).

## Nuevo: `technique_advisor` dio un falso positivo real (declarado, no corregido todavia)

Sobre el mismo circuito tipo Shor (grado maximo=10, dispersion
espectral=0.082, ambos sobre el umbral de la regla), la regla de
ZX-calculus emitio ADVERTENCIA de "patron similar a QAOA". Se corrio
`full_reduce()` real para verificar (como la propia advertencia
recomienda) -- **el T-count SI bajo, 39.2%** (3930->2388), muy lejos del
0% que sugeria la advertencia. Es un falso positivo real de la heuristica
de deteccion de patron QAOA (grado alto + dispersion espectral moderada
tambien aparece en circuitos que no son QAOA), documentado aqui en vez de
ajustar la regla sin mas evidencia -- se necesitan mas instancias de este
tipo de circuito para saber si esto es un caso aislado o un patron.

## Nuevo: circuit cutting SI puede fallar por overflow numerico, no solo por overhead alto

Corriendo `find_cuts` sobre el mismo circuito tipo Shor (grado promedio
9.1 sobre 16 qubits, bastante denso), el buscador de cortes literalmente
desbordo un flotante calculando el overhead de muestreo (`OverflowError:
cannot convert float infinity to integer`) -- consistente con lo ya visto
en QAOA/dense (overhead de 10^17-10^44x), pero esta vez el overhead fue
tan extremo que ni siquiera se pudo representar como numero. Se reporta
como el mismo hallazgo cualitativo (circuitos densos no son separables)
con una manifestacion tecnica distinta.

## Nuevo: se probo la estrategia de layout del notebook QGSS 2025 (lab-2) contra nuestros circuitos

A raiz de revisar github.com/qiskit-community/qgss-2025 (lab-2, que busca layouts
EXACTOS via isomorfismo de subgrafos + error de calibracion, antes de barrer
semillas del transpilador), se implemento esa misma estrategia de forma
generalizada en `graph_layout_search.py` y se probo contra los 9 circuitos de
referencia del proyecto. Resultado honesto:

- Esa estrategia SOLO es aplicable cuando el grado maximo del grafo de
  interaccion del circuito es <= grado maximo del mapa de acoplamiento fisico
  (3, en heavy-hex). De nuestros 9 circuitos, **8 de 9 son estructuralmente
  infactibles** para esa estrategia -- incluyendo nuestro propio benchmark QAOA
  (grado maximo 10). Esto demuestra, con una prueba estructural exacta (VF2),
  por que la busqueda conjunta layout+ruteo (ETAPA 0) es necesaria: no es que
  el layout este mal elegido, es que el circuito no cabe sin SWAPs sin importar
  el layout.
- En el unico circuito donde SI aplica (`vqe_ansatz.qasm`, grado 2), se corrio
  la comparacion directa: los candidatos VF2 dieron avg_error=0.00349, contra
  **0.00302 de nuestra busqueda de 20 semillas** -- la nuestra sigue ganando,
  porque el ranking por error de calibracion acumulado no captura errores de
  lectura ni de 1 qubit que si entran en la medicion real post-transpilacion.

## Nuevo: se probo el stack de EM del mismo notebook (twirling/DD/ZNE via IBM Runtime) -- no se puede validar sin hardware real

El notebook aplica `EstimatorOptions.resilience` (ZNE, TREX) y
`options.twirling` de `qiskit_ibm_runtime` contra hardware real en la nube.
Se probo llamar exactamente esas mismas opciones contra un FakeBackend local
(sin token, sin nube) y Qiskit advierte explicitamente: *"Options ... have no
effect in local testing mode"* -- el resultado devuelto es identico al caso
sin mitigacion. Verificado directamente, no asumido.

Se implemento entonces una version 100% propia y local de dos de esas
tecnicas en `manual_em_stack.py`: Pauli twirling (con la matriz de correccion
calculada por conjugacion exacta del gate, no copiada de una tabla) y
dynamical decoupling (con el pass real `PadDynamicalDecoupling` del
transpilador de Qiskit, que corre localmente). Al correrlo sobre
`grover.qasm` con ruido real derivado de FakeTorino:

- **El resultado SIN mitigacion fue el MEJOR de los tres** (TVD=0.4266 vs.
  distribucion ideal). Dynamical decoupling lo empeoro (TVD=0.4711), y
  twirling+DD lo empeoro mas (TVD=0.5077).
- Causa raiz verificada (no es un bug): `NoiseModel.from_backend(...)` de
  Qiskit Aer NO incluye ninguna instruccion `"delay"` en sus canales de
  ruido -- el tiempo ocioso no tiene ningun error asociado en este modelo,
  asi que DD (que existe para suprimir decoherencia durante tiempo ocioso)
  no puede mostrar ningun beneficio en esta simulacion: cada pulso extra
  solo suma su propio error de compuerta, sin ningun costo evitado.
- Ademas, el ruido derivado de calibraciones ya es de Pauli/despolarizacion
  por construccion (no coherente), asi que Pauli twirling -- que sirve para
  convertir ruido coherente en estocastico -- no tiene nada que convertir
  aqui.

**Conclusion honesta:** ni el stack de produccion de IBM Runtime ni nuestra
propia reimplementacion local de DD/twirling pueden validarse sin hardware
real -- ambos actuan sobre fisica (decoherencia temporal, ruido coherente)
que ninguna simulacion basada en calibraciones de backend representa. La
unica tecnica de este proyecto que SI se pudo validar honestamente de forma
100% local fue la busqueda de layout/ruteo (ETAPA 0), porque el error de
compuerta -- lo unico que el modelo de ruido simulado si penaliza -- es
exactamente lo que esa busqueda optimiza.

## Resuelto: el falso positivo de `technique_advisor` en el circuito Shor-like

El hallazgo anterior (ADVERTENCIA QAOA-like disparada por
`shor_like_sequential.qasm` pese a que `full_reduce()` real dio 39.2% de
reduccion) se dejo abierto por falta de datos. Se investigo la causa raiz
en vez de esperar mas instancias:

- La regla original (`max_degree>=4.5` y `spectral_sparsifiability>=0.08`)
  solo mira la FORMA del grafo de interaccion -- nunca verifica el
  mecanismo real que distingue a un operador de costo QAOA: que sus
  compuertas de 2 qubits son rotaciones PARAMETRIZADAS
  (`exp(-i*gamma*Z_aZ_b)`) con el MISMO angulo `gamma` repetido en todas
  las aristas.
- Verificado directamente: `shor_like_sequential.qasm` esta compuesto
  UNICAMENTE de compuertas `u` (1 qubit) y `cx` (2 qubits, sin parametro)
  -- cero compuertas parametrizadas. No puede tener la firma QAOA sin
  importar que tan denso sea su grafo.
- `qaoa_farm.qasm` y `qaoa_small_test.qasm`, en cambio, implementan su
  operador de costo como `CX; RZ(2*gamma); CX` (la forma estandar de
  `exp(-i*gamma*ZZ)`), con exactamente 1 angulo distinto repetido en el
  100% de esas instancias de `rz`.

Se agrego `_qaoa_repeated_angle_signature()` a `technique_advisor.py`
como condicion adicional (AND, no reemplazo) de la regla existente.
Resultado, verificado sobre los 9 circuitos de referencia: la ADVERTENCIA
ahora dispara SOLO en los 2 circuitos QAOA reales (donde full_reduce()
real sigue dando 0.0% de reduccion, confirmado de nuevo) y ya NO dispara
en `shor_like_sequential.qasm` (el falso positivo). El resto de circuitos
(adder8, qft6, qft10, random8, vqe_ansatz, grover) tampoco disparan,
como antes.

## Corregido de nuevo: el fix anterior tenia un falso negativo (QAOA con pesos de arista)

Al probar el caso obvio que faltaba -- QAOA con MaxCut PONDERADO (cada arista
usa `gamma * peso_arista`, un angulo distinto por arista; el caso real de
aplicacion, no el juguete con un solo angulo compartido) -- se encontro que
`_qaoa_repeated_angle_signature` (v1 del fix) fallaba: como cada arista tiene
un angulo distinto, ningun angulo domina, la firma da False, la ADVERTENCIA
no dispara -- pero `full_reduce()` real sigue dando **0.0%** de reduccion,
igual que en el QAOA sin pesos. Verificado con un circuito de 17 qubits
construido especificamente para esta prueba.

Se reemplazo la firma de "angulo unico repetido" (fragil ante pesos) por
`_zz_cost_operator_motif_fraction`: en vez de mirar el VALOR del angulo, se
busca el motivo ESTRUCTURAL invariante de un operador de costo diagonal en Z
-- el sandwich `CX(a,b); RZ(theta); CX(a,b)` que implementa
`exp(-i*theta*Z_aZ_b)`, sin importar el valor de `theta`. Esto es
angulo-agnostico por construccion: detecta QAOA con y sin pesos por igual, y
sigue sin detectar nada en `shor_like_sequential.qasm` (que no tiene ninguna
compuerta `rz`). Reverificado sobre los 9 circuitos + el caso ponderado: la
ADVERTENCIA dispara en los 3 casos QAOA (con y sin pesos) y en ningun otro.

Limitacion declarada del nuevo detector: es un escaneo de TRIPLETES
CONTIGUOS en el flujo de instrucciones, no un analisis de conmutacion
completo -- un circuito que un transpilador reordene (intercalando otras
compuertas que conmutan entre el CX de apertura y cierre de la misma
arista) podria evadir la deteccion. No se resuelve aqui; se declara como
limite conocido, igual que el resto de heuristicas de este proyecto.

## Cross-validacion: el pass nativo `VF2PostLayout` de Qiskit no mejora nuestra busqueda

Qiskit trae su propio pass de refinamiento post-ruteo, `VF2PostLayout`
(isomorfismo de subgrafos aplicado DESPUES de rutear, buscando una asignacion
fisica alternativa con mejor calibracion para la MISMA topologia ya usada).
Se corrio directamente sobre el mejor resultado de nuestra busqueda de 10
semillas para `shor_like_sequential.qasm`: devolvio
`VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND` -- no encontro ningun
mapeo alternativo mejor. Esto es una validacion cruzada util (no solo una
prueba mas): confirma que nuestra busqueda por semillas ya converge a un
punto que ni siquiera el pass oficial de Qiskit para este problema exacto
puede superar, en vez de sugerir que nos faltaba una mejora facil y obvia
que ya existia en la libreria.

## Generalizado (v3): de "patron QAOA" a "capa Hamiltoniana diagonal en Z" -- con fundamento teorico, no solo empirico

Se generalizo `_zz_cost_operator_motif_fraction` mas alla de QAOA, apoyandose
en la teoria de polinomios de fase detras de ZX-calculus (Kissinger & van de
Wetering): el poder de `full_reduce()` para bajar T-count viene de fusionar
spiders A TRAVES de bases no diagonales (aristas Hadamard). Si TODA la parte
no-Clifford de un circuito ya es una unica capa diagonal en Z (una suma de
terminos `exp(-i*theta*Z_aZ_b)` que ya conmutan y estan fusionados al maximo
por construccion), no queda nada que fusionar -- esto es un resultado
ESTRUCTURAL, no solo el patron empirico "QAOA da 0%" encontrado antes. Aplica
igual a VQE de Ising, QUBO, y cualquier otro ansatz de una sola capa de costo
diagonal + mezclador, no solo a QAOA.

Se agrego deteccion de la forma NATIVA de esa capa (compuertas `rzz`, `cp`,
`crz`, `cu1` aplicadas directamente, sin la decomposicion `CX;RZ;CX`) ademas
del sandwich ya detectado. Se construyo un VQE-Ising de prueba (17 qubits,
`rzz` nativo, angulos distintos por arista, sin relacion con QAOA en su
generador) especificamente para validar esta generalizacion:

- `full_reduce()` real: **0.0%** de reduccion -- confirma la prediccion
  teorica antes de siquiera correr el detector.
- El detector generalizado: `motif=1.00`, ADVERTENCIA dispara correctamente.

Reverificado sobre los 9 circuitos de referencia + los 2 casos sinteticos
(QAOA ponderado, VQE-Ising nativo): la ADVERTENCIA dispara en los 4 casos de
capa diagonal (con y sin pesos, con y sin decomposicion) y en ningun otro.
`grover.qasm` paso de motif=0.00 a motif=0.40 (su oraculo usa algunas
compuertas de fase controladas nativamente) pero se queda bajo el umbral de
0.8 -- no dispara advertencia, consistente con que Grover si se beneficia de
sintesis/reduccion real (no es una capa diagonal completa, solo tiene
ALGUNOS terminos diagonales dentro de una estructura mas rica).

## Nuevo experimento: la reduccion de circuito como VARIABLE DE DECISION explicita (no solo un paso previo)

No se encontro ningun trabajo publicado que formule la reduccion de
circuitos (ZX-calculus, teletransportacion, etc.) como palanca de DECISION
de regimen QEM-vs-FTQC -- la literatura de reduccion optimiza recursos por
si misma, y los estimadores de recursos (Azure QRE) no incorporan reduccion
de circuito antes de estimar. Se implemento `reduction_sensitivity_sweep.py`
para responder la pregunta directamente: para `shor_like_sequential.qasm`
(el unico circuito de este proyecto que hoy no cae claramente en ningun
regimen), ¿CUANTA reduccion de T-count haria falta para cambiar el
veredicto?

Resultado (barrido de 0% a 90% de reduccion hipotetica de T-count, con
p_phys y num_qubits fijos en sus valores reales medidos):

**SIN CRUCE.** El veredicto se mantiene "SIN VENTAJA FTQC POSIBLE HOY" en
TODO el rango, incluyendo una reduccion hipotetica del 90% (mucho mas
agresiva que el 39.2% real medido con ZX-calculus). La razon, cuantificada:
el conteo de qubits fisicos que exige Azure QRE para este circuito
(706,250 sin reducir, 294,010 con 90% de reduccion hipotetica) esta
dominado por el numero de qubits LOGICOS y la distancia de codigo que
exige `p_phys` medido (0.32%) -- NO por el T-count. Reducir T-count baja el
RUNTIME estimado (de 3.9e-2s a 3.4e-3s) pero apenas mueve el conteo de
qubits necesario. Para que este circuito fuera factible al presupuesto
usado (50x los qubits del backend = 6,650), haria falta un multiplicador
de presupuesto de ~2,211x incluso con 90% de reduccion -- casi el mismo
orden de magnitud que sin reducir (~5,310x). Es decir: **para este
circuito, la reduccion de T-count no es la palanca que resuelve la
factibilidad -- el cuello de botella es el numero de qubits fisicos
disponibles, no la estructura del circuito.** Esto reconfirma, desde un
angulo distinto y especifico a un circuito real (no un barrido de
arquetipos escalados), el hallazgo de `ftqc_crossover_sweep.py`: escalar
el presupuesto de qubits, no mejorar la calidad del circuito o del
hardware, es la restriccion que de verdad importa aqui.

## Pregunta del usuario: "¿se pueden reusar qubits fisicos, apoyados por mediciones intermedias?"

Respuesta corta: la posibilidad existe y se puede MEDIR en nuestros propios
circuitos -- pero el ahorro de qubits fisicos que promete NO se puede
capturar correctamente con las herramientas de estimacion FTQC que este
proyecto ya usa (Azure QRE). Se investigo directamente en vez de asumir.

**Metrica nueva agregada:** `max_concurrent_qubits` (en `technique_advisor.py`,
`profile_circuit`) -- en vez del proxy crudo anterior ("ancilla candidata"
= ultimo uso antes de la mitad del circuito), se calcula el numero MINIMO
de "ranuras" logicas simultaneas que el circuito realmente necesita
(intervalos de actividad `[primer_uso, ultimo_uso]` por qubit, maximo de
superposicion -- el problema clasico de asignacion de registros via
coloreo de grafo de intervalos). Resultado sobre los 9 circuitos de
referencia: la mayoria NO tiene margen (concurrentes == declarados), pero
**`shor_like_sequential.qasm` si**: 16 qubits declarados, solo 11
concurrentes en el peor momento (31% de margen), y `vqe_ansatz.qasm` aun
mas (8 declarados, 3 concurrentes, 62% de margen).

**Pero al probar directamente si Azure QRE puede aprovechar esto (pasandole
`numQubits=11` en vez de 16 para `shor_like_sequential.qasm`, T-count y
p_phys fijos), el resultado fue CONTRAINTUITIVO:** el conteo de qubits
fisicos totales EMPEORO (706,250 -> 1,242,274), no mejoro. Se investigo el
desglose interno: `physicalQubitsForAlgorithm` SI bajo como se esperaria
(56,250 -> 34,914), pero `physicalQubitsForTfactories` SUBIO fuertemente
(650,000 -> 1,207,360), dominando el resultado neto.

**Causa raiz:** el modelo `LogicalCounts` de Azure QRE NO tiene ningun
concepto de reuso temporal de qubits via medicion+reset -- `numQubits` es
un parametro ESTATICO que representa cuantos parches de codigo de
superficie existen simultaneamente durante TODA la ejecucion. Al reducir
ese numero manualmente sin cambiar nada mas, el optimizador conjunto de
Azure (que balancea distancia de codigo, factorias de destilacion y
presupuesto de error TOTAL entre todos los qubits) reoptimiza de forma no
lineal y termina exigiendo factorias de destilacion mas caras para cumplir
el mismo presupuesto de error con menos "espacio" logico disponible. Es
decir: sustituir un `numQubits` mas chico a mano, para simular reuso, es
un USO INVALIDO del modelo de Azure QRE, y produce un resultado
literalmente al reves de lo esperado.

**Conclusion honesta:** el reuso de qubits fisicos via medicion+reset es
una tecnica real y establecida en la era NISQ (circuitos dinamicos), y nue
stro circuito Shor-like SI tiene margen medible para ella (31%) -- pero
ninguna herramienta de estimacion de recursos FTQC que este proyecto usa
(Azure QRE, ni nuestro propio estimador) modela el reuso temporal de
parches de superficie via lattice surgery. Cuantificar el beneficio real
requeriria un modelo de estimacion distinto (o construir uno propio) que
no existe en este proyecto todavia -- se declara como limitacion abierta,
no como una mejora falsa.

## Se busco la metrica y se probo directamente en CUDA-Q: el reuso temporal NO esta implementado en ninguna herramienta que revisamos

Pregunta del usuario: ¿la metrica de "maximo de qubits concurrentes" ya existe
en el estado del arte, y la herramienta CUDA-Q (NVIDIA) hace ese reuso?

**Si existe la metrica** -- se llama `logical_qubits_peak` en
`cudaq.logical` (el mismo modulo NVIDIA que ya usamos en `qft_cudaq.py`
para comparar contra Azure QRE), asi que no la inventamos aislados: es un
concepto reconocido.

**Pero se probo DIRECTAMENTE si CUDA-Q realiza el reuso, y NO lo hace por
defecto.** Se construyo un kernel de prueba (6 qubits declarados, 3 activos
en la primera mitad, otros 3 en la segunda mitad, SIN interaccion entre
ambos grupos -- exactamente el patron que si tiene margen de reuso segun
nuestra propia metrica):
  - Sin ninguna anotacion especial: `logical_qubits_peak = 6` (no 3).
  - Agregando medicion explicita (`mz`) sobre los 3 primeros qubits antes
    de usar los otros 3: sigue dando `logical_qubits_peak = 6` -- la sola
    medicion NO le indica al estimador que puede reusar el patch.
  - CUDA-Q SI expone primitivas de mas bajo nivel (`allocate`, `discard`,
    `reset`, `allocate_patch`) que sugieren que el reuso es posible en
    principio, pero pertenecen a una API de construccion manual de
    protocolos, DISTINTA de la API de kernel estandar (`@cudaq.kernel`)
    que usamos en este proyecto -- `discard()` ni siquiera es una llamada
    reconocida dentro de un kernel normal (error de compilacion real:
    "unknown function call").

**Se busco tambien en la literatura publicada (2025):** ni el paper
"Space-Time Optimisations" (arXiv:2511.08848) ni "A Resource Allocating
Compiler for Lattice Surgery" (arXiv:2506.04620) -- este ultimo,
especificamente un compilador dedicado a manejar memoria de parches de
codigo de superficie -- implementan reuso temporal de parches para qubits
LOGICOS. Este segundo paper SI reusa memoria, pero solo para las fabricas
de destilacion de magic states (`"Once a factory is complete a RESET
operation indicates to the compiler that the unit of memory required by
the factory may be deallocated for re-use"`) -- nunca para qubits logicos
de proposito general.

**Conclusion, ahora con mas confianza que antes:** no es que a este
proyecto le falte implementar el reuso -- es que, hasta donde se pudo
verificar (Azure QRE, CUDA-Q Logical, y un compilador de lattice surgery
publicado en 2025), NINGUNA herramienta de estimacion de recursos FTQC
disponible hoy modela el reuso temporal de parches para qubits logicos de
proposito general. El margen que medimos (31% en Shor-like) es real y
cuantificable con nuestra propia metrica, pero explotarlo requeriria
construir una capa de estimacion que hoy no existe en ningun lado que
hayamos podido revisar -- un gap genuino del estado del arte, no solo de
este proyecto.

## Se encontro COMO hacerlo: reuso de qubits implementado correctamente en NUESTRO PROPIO estimador

Tras confirmar que ni Azure QRE ni CUDA-Q Logical ni la literatura de 2025
modelan reuso temporal de qubits logicos, se investigo si HABIA una forma
de implementarlo nosotros mismos -- y si.

**La clave:** Azure QRE fallo (dio un resultado al reves) porque su
formula es un OPTIMIZADOR CONJUNTO no lineal que reparte un presupuesto de
error compartido entre qubits de datos y fabricas de destilacion --
cambiar el numero de qubits reoptimiza TODO, incluidas las fabricas, de
forma impredecible. Nuestro propio estimador (`open_resource_estimator.py`)
NO tiene ese acoplamiento -- ya declaraba, desde antes, que NO modela
fabricas de destilacion en absoluto. Eso significa que su formula
(`qubits_fisicos_totales = num_qubits_logicos * qubits_fisicos_por_
logico(d)`) es LINEAL y SEPARADA del calculo de runtime (que depende solo
de T-count) -- sustituir `num_qubits_logicos` por `max_concurrent_qubits`
(la metrica de la seccion anterior) es matematicamente valido aqui, sin
ningun acoplamiento oculto que lo rompa.

Verificado sobre `shor_like_sequential.qasm` (T-count=3930, p_phys=0.32%):

| | qubits logicos | qubits fisicos | runtime |
|---|---|---|---|
| sin reuso (declarado) | 16 | 18,016 | 2.712e-02s |
| CON reuso (max concurrentes) | 11 | 12,386 | 2.712e-02s (identico, correcto) |

La reduccion (18,016 -> 12,386) es EXACTAMENTE proporcional a 11/16 =
0.6875, como debe ser en una formula lineal -- y el runtime NO cambia
(correcto: nuestro modelo de tiempo depende solo de T-count, no de
concurrencia de qubits).

**Limitacion honesta de este resultado:** esto SOLO corrige la parte de
qubits de DATOS del estimador -- la parte que nuestro estimador SI modela.
No incluye el costo de fabricas de destilacion (que nuestro estimador
declara no modelar desde el inicio), asi que el numero total sigue siendo
un subconteo frente a Azure QRE (igual que antes, ahora por una razon
adicional). Para tener un numero de reuso TOTALMENTE correcto (datos +
fabricas), el siguiente paso natural seria implementar tambien el modelo
de fabricas de destilacion de Litinski (2019) DENTRO de este mismo
estimador propio -- asi ambas piezas (reuso de qubits de datos, costeo de
fabricas) estarian en un modelo que controlamos y entendemos por completo,
en vez de pelear con el optimizador conjunto de un vendor.

## Se implemento el modelo de fabricas de Litinski (2019) -- el ahorro de reuso se mantiene, pero se reduce

Siguiendo la seccion anterior (donde se propuso el modelo de fabricas de
Litinski como el siguiente paso natural), se implemento en
`open_resource_estimator.py` (`litinski_15to1_factory_qubits()` y
`estimate_open_ftqc_with_factories()`).

**La formula NO se reajusto ni se re-derivo con nuestros datos** (a
diferencia de `physical_qubits_per_logical(d)`, que si viene de nuestra
propia simulacion Stim). Se tomo LITERALMENTE del paper de Litinski
2019 ("A Game of Surface Codes: Large-Scale Quantum Computing with
Lattice Surgery", arXiv:1808.02892), verificado via fetch directo del
PDF:

- "the 15-to-1 protocol of Fig. 15 is implemented in 11*Delta using 11 tiles"
- Ejemplo numerico de la Seccion 4.1, d=13: "11 tiles x 2d^2 = 11 x 338 =
  3,718 physical data qubits" (11 x 2 x 13^2 = 3718, verificado
  aritmeticamente)

=> `qubits_por_fabrica(d) = 11 * 2 * d^2 = 22*d^2`

### Resultado para shor_like_sequential.qasm (tcount=3930, p_fisico=0.0032, d=23)

| | Solo qubits de datos (seccion anterior) | Con 1 fabrica 15-a-1 (nuevo) |
|---|---|---|
| nq=16 (sin reuso) | 18,016 | 18,016 + 11,638 = **29,654** |
| nq=11 (con reuso, max_concurrent_qubits) | 12,386 | 12,386 + 11,638 = **24,024** |
| **Ahorro por reuso** | **31.25%** | **18.99%** |

El ahorro por reuso de qubits **se mantiene positivo** -- a diferencia de
la sustitucion en Azure QRE, que dio un resultado invertido/incorrecto
(ver seccion anterior) -- pero se **reduce de 31.25% a 18.99%**, porque
el costo de fabricas (11,638 qubits, fijo una vez fijado d) NO depende de
`num_logical_qubits` en absoluto: es un costo constante que no se
beneficia de la reduccion de qubits concurrentes, mientras que el costo
de datos si se beneficia completamente (razon exacta 11/16 se mantiene
en la columna de datos).

**Limitaciones declaradas de este primer modelo de fabricas (no resueltas
aqui):**

1. **`num_factories=1` fijo por default.** Es la unica opcion consistente
   con el modelo de tiempo ya declarado (consumo secuencial de
   compuertas T, sin paralelizar fabricas). Usar mas de una fabrica sin
   tambien corregir el modelo de tiempo (que no se hizo) dar
   ia un numero de fabricas necesarias inconsistente con el runtime
   reportado -- por eso se expone como perilla auditable, no como caso
   recomendado.
2. El modelo de tiempo NO cambia respecto a la version sin fabricas
   (sigue siendo una cota superior declarada, no una optimizacion tipo
   Litinski).
3. No se modela reuso temporal de las fabricas mismas (solo de qubits de
   datos) -- el paper "A Resource Allocating Compiler for Lattice
   Surgery" (arXiv:2506.04620, ya citado en la seccion anterior) SI
   modela liberacion/reuso de la memoria de fabricas tras completarse,
   lo cual es un candidato natural para una segunda iteracion, no
   implementado aqui.

**Conclusion honesta:** el reuso de qubits sigue siendo una optimizacion
real y cuantificable (no un artefacto de un modelo incompleto), pero su
impacto relativo depende de que fraccion del costo total de un computo
FTQC concreto es qubits de datos vs. fabricas -- y esa fraccion, en este
caso, hace que el beneficio caiga de ~31% a ~19% al contarla de forma
mas completa.

## Demostracion de bandera: ¿es este circuito realizable HOY, y donde ayudan realmente QEC y QEM?

Se junto todo lo construido en una demostracion concreta sobre
`shor_like_sequential.qasm` (el circuito mas dificil de este proyecto,
estudiado en todas las secciones anteriores):

**El circuito, medido de dos formas reales:** Qiskit reporta profundidad
6,301 (3,050 capas de compuertas de 2 qubits), 8,210 instrucciones tal
como esta escrito; nuestro perfilador (que decompone a compuertas
elementales de PyZX) reporta 28,618 compuertas elementales, 3,108
compuertas de 2 qubits, T-count=3,930. Ambos conteos son mediciones reales
del mismo circuito, no estimaciones en desacuerdo -- difieren porque
cuentan cosas distintas (instrucciones tal cual vs. tras descomponer a un
gate-set fijo).

**¿Es realizable HOY solo con QEM?** Con la tasa de error de 2 qubits ya
usada en todo este proyecto (p=0.32%, calibracion real de IBM), la
probabilidad de que UN SOLO shot de las 3,108 compuertas de 2 qubits salga
sin ningun error es `(1-0.0032)^3108 ≈ 4.7e-5`. Este es el numero que
importa para un algoritmo cuya correccion depende de un bitstring de
salida limpio (no de un valor esperado). Por eso QEM (ZNE, twirling, TREX)
es la herramienta equivocada para ESTE circuito especificamente: esas
tecnicas reducen el SESGO de un valor esperado promediado sobre muchos
shots ruidosos, no la probabilidad de error por shot -- ninguna cantidad
de QEM convierte 4.7e-5 en algo usable. Es un desajuste estructural entre
tecnica y problema, no una cuestion de grado.

**Ancla de hardware real (marzo de 2026, posterior a nuestro corte de
entrenamiento, citado de cobertura web contemporanea):** Quantinuum
reporto en su procesador Helios 94 qubits logicos "detectados" (no
corregidos) desde 98 fisicos (overhead ~1:1, codigo iceberg distancia 2),
o 48 qubits logicos corregidos via un codigo concatenado [[80,48,4]]
(overhead ~1.67:1). Ambos son overheads dramaticamente menores que el
codigo de superficie que usamos en el resto de este proyecto (~1,126
qubits fisicos por logico a d=23) -- pero su prueba mas profunda (64
qubits logicos, 10 pasos de Trotter) ya vio caer la tasa de aceptacion
(post-seleccion) a ~3.2%. Eso no es casualidad: los codigos de deteccion
DESCARTAN un shot al primer error detectado (no lo corrigen), asi que la
fraccion sobreviviente se encoge multiplicativamente con la profundidad.
Nuestro circuito de bandera (profundidad 6,301, 3,930 compuertas T) esta
dos o tres ordenes de magnitud mas alla de ese regimen ya exigido -- la
lectura honesta es que el resultado real de Quantinuum, siendo
extraordinario, apunta a una clase de circuito mas superficial que la
estudiada aqui, y es evidencia EN CONTRA de usar codigos de deteccion
para este circuito, no una demostracion de que ya es alcanzable.

**Lo que dice nuestro propio estimador (con fabricas incluidas), y su
piso:**

| qubits logicos | d | qubits datos | qubits fabrica | TOTAL |
|---|---|---|---|---|
| 1 | 23 | 1,126 | 11,638 | 12,764 |
| 4 | 23 | 4,504 | 11,638 | 16,142 |
| 11 | 23 | 12,386 | 11,638 | 24,024 |
| 16 | 23 | 18,016 | 11,638 | 29,654 |

El total NO baja de ~12,764 qubits fisicos ni siquiera con 1 solo qubit
logico hipotetico, porque el costo de fabrica (11,638 a d=23) esta fijado
por p_fisico y el presupuesto de error, no por el numero de qubits.

Reduciendo T-count en cambio (nq=16 fijo) si mueve el total, porque un
T-count menor relaja el presupuesto de error por compuerta y permite una
distancia menor:

| reduccion T-count | T-count | d | TOTAL fisico |
|---|---|---|---|
| 0% | 3,930 | 23 | 29,654 |
| 70% | 1,179 | 19 | 20,390 |
| 90% | 392 | 17 | 16,406 |
| 99% | 39 | 11 | 7,046 |

Incluso con 99% de reduccion hipotetica de T-count (mucho mas alla de lo
que nuestras tecnicas reales de reduccion lograron en este circuito), el
total (7,046 qubits fisicos) sigue superando el mayor conteo de qubits
fisicos publicamente reportado para un solo dispositivo real, el
procesador Condor de IBM (1,121 qubits), citado solo como referencia de
orden de magnitud. **Este circuito, a este tamano, no es realizable en
FTQC completo en ningun dispositivo real que exista hoy, bajo ninguna
reduccion de T-count probada en este proyecto.**

**Conclusion honesta sobre donde ayuda cada tecnica:** FTQC/QEC es la
unica tecnica que alguna vez podria hacer confiable la salida de este
circuito exacto, pero no es realizable hoy a este tamano, por un margen
amplio, y los codigos de deteccion (el unico resultado real de qubits
logicos disponible) son la familia equivocada para su profundidad. QEM
por si solo tampoco arregla ese desajuste. Lo que nuestro propio
`technique_advisor` SI recomienda para este circuito exacto, aplicado
directamente sobre el: teletransportacion RZZ (recomendada), reduccion
via calculo ZX (recomendada, con la tasa base honesta de 68% declarada,
no inflada), y descomputo por medicion de sus 2 candidatos a ancilla
(recomendada, pendiente de verificacion manual). Esas son las tecnicas
que benefician a este circuito HOY, en hardware NISQ real o simulado,
reduciendo el conteo de compuertas de 2 qubits y por tanto la
probabilidad de error por shot calculada arriba -- no eliminan el piso de
T-count/qubits que mantiene fuera de alcance al FTQC completo, pero son
la palanca correcta y disponible hoy, mientras que QEC/FTQC sigue siendo
una escala que este problema concreto aun no alcanza.

## Hacia una sola herramienta: QASM como entrada unica, agnostica de hardware

Se integro todo lo construido en este proyecto en un unico punto de
entrada, `qasm_tool.py`: un archivo QASM + un perfil de hardware minimo
-> un reporte consolidado (perfilado, tecnicas recomendadas, factibilidad
estructural, estimacion de recursos FTQC con fabricas). Sin generar
codigo especifico de ningun vendor en ningun punto del pipeline.

**El unico lugar donde entra hardware:** `HardwareProfile` -- nombre,
`p_2q` (tasa de error de compuerta de 2 qubits), `connectivity`
(`all_to_all` o `limited` con grado maximo declarado), y una fuente
citada obligatoria. No es una afirmacion de diseno -- se verifica por
firma de funcion: `profile_circuit()` y `recommend_techniques()` no
reciben NINGUN parametro de hardware, y `estimate_open_ftqc_with_factories()`
solo recibe el escalar `p_2q`.

**Validado, no solo argumentado -- la MISMA herramienta sobre 3
arquitecturas reales publicadas, sin cambiar una linea de codigo:**

| arquitectura | p_2q | conectividad | d | qubits fisicos totales |
|---|---|---|---|---|
| Superconductor (IBM, calibracion real) | 0.32% | limitada (grado 3) | 23 | 29,654 |
| Trapped-ion (Quantinuum H2, publicado) | 0.10% | all-to-all | 9 | 4,790 |
| Trapped-ion (IonQ/Oxford Ionics, publicado, oct. 2025) | 0.01% | all-to-all | 5 | 1,574 |

La brecha entre filas la explica casi enteramente p_2q via la distancia
de codigo requerida, y la factibilidad estructural (un hallazgo real e
independiente para el caso superconductor) se vuelve trivialmente
verdadera para conectividad all-to-all -- exactamente como predice el
formalismo, no como un caso especial anadido despues.

**Lo que NO generaliza automaticamente (declarado, no ocultado):**

1. Conectividad reconfigurable (atomos neutros con shuttling entre capas
   del circuito) no es un mapa de acoplamiento estatico -- `structural_feasibility_generic`
   solo distingue `limited` (estatico) de `all_to_all`, sin modelo para
   conectividad que cambia a mitad del circuito.
2. Computacion fotonica basada en medicion (MBQC) no se puede alimentar
   honestamente por este pipeline: su costo de recursos se mide en
   tamano de cluster-state y profundidad de medicion adaptativa, no en
   conteo de compuertas sobre un registro fijo -- un QASM no representa
   el mismo computo en ese modelo. Se declara NO SOPORTADO en vez de
   reinterpretar silenciosamente un circuito de compuertas como uno
   fotonico.
3. El ajuste de umbral reusado (p_th=0.74%, de nuestra propia simulacion
   Stim bajo un modelo de ruido depolarizing/Pauli) se aplica sin cambios
   a los p_2q de trapped-ion arriba -- es el supuesto estandar en la
   mayoria de la literatura de codigos de superficie, no una verificacion
   de que los canales de error especificos de trapped-ion (fuga,
   calentamiento del ion, crosstalk de puerta global) sean aproximadamente
   Pauli -- una aproximacion honesta, no rederivada por canal.

## Mas alla de la estimacion analitica: que pasa si lo corres HOY (simulacion ruidosa real, no solo formula)

Se agrego `simulate_expectation()` a `qasm_tool.py`: en vez de solo
calcular analiticamente cuantos qubits fisicos necesitaria FTQC completo,
corre una simulacion RUIDOSA real (Qiskit Aer, metodo statevector con
muestreo por trayectorias, canal depolarizing generico de 2 qubits
parametrizado por p_2q de cada `HardwareProfile`) y mide que fraccion de
los shots cae en el soporte ideal del circuito (64 bitstrings de
probabilidad no nula sobre 2^16, calculados una sola vez con
`Statevector` exacto) -- el mismo criterio de exito que usaria un
circuito real tipo Shor (no exige el bitstring exacto, exige un resultado
valido).

**Resultado, 150 shots por arquitectura, misma semilla, mismo codigo --
solo cambio el HardwareProfile:**

| arquitectura | p_2q | exito medido (150 shots) |
|---|---|---|
| Superconductor (IBM) | 0.32% | 0/150 (0.0%) |
| Trapped-ion (Quantinuum H2) | 0.10% | 15/150 (10.0%) |
| Trapped-ion (IonQ/Oxford Ionics) | 0.01% | 125/150 (83.3%) |

Esto es consistente con, y mas contundente que, la estimacion analitica
de "cero errores" calculada antes (4.7e-5 a la tasa de error
superconductora): cero exitos en 150 shots es exactamente lo que predice
esa estimacion, y los dos perfiles trapped-ion de bajo error muestran el
mismo cruce cualitativo ya encontrado en la comparacion analitica de
recursos -- ahora DEMOSTRADO por ejecucion real, no solo estimado.

**Limitacion declarada:** el modelo de ruido es un canal depolarizing
generico de 2 qubits parametrizado solo por p_2q -- sin error de lectura,
sin error de 1 qubit, sin decoherencia T1/T2 dependiente del tiempo de
compuerta, sin correlaciones espaciales. Se uso deliberadamente el MISMO
modelo simplificado para las 3 arquitecturas (en vez de mezclar la
calibracion completa que si existe para el caso superconductor en otras
partes de este proyecto, via `AerSimulator.from_backend`, con dos
aproximaciones para trapped-ion) para que la comparacion aisle el efecto
de p_2q, no el de tener modelos de ruido de distinta fidelidad.

## Dependencias e independencia de vendor

El nucleo de la herramienta (`qasm_tool.py`, `technique_advisor.py`,
`open_resource_estimator.py`, `graph_layout_search.py`) depende
UNICAMENTE de librerias de codigo abierto que corren localmente, sin
cuenta ni API key de ningun vendor:

| Libreria | Licencia | Para que | Requiere nube/cuenta |
|---|---|---|---|
| Qiskit (core) | Apache 2.0 | Parsear QASM, profundidad, conteo de compuertas | No |
| Qiskit Aer | Apache 2.0 | Simulacion ruidosa (capa de expectativa) | No |
| `qiskit_ibm_runtime.fake_provider` | Apache 2.0 | Snapshot ESTATICO de calibracion real (FakeTorino) | No -- es un archivo empaquetado, no una conexion |
| PyZX | MIT | Calculo ZX, reduccion de circuito | No |
| Stim + PyMatching | BSD / MIT | Nuestra propia simulacion de codigo de superficie | No |
| rustworkx | Apache 2.0 | Grafos de interaccion, VF2 | No |
| NumPy | BSD | Algebra lineal | No |

Verificado por imports reales (no por afirmacion): ninguno de esos 4
modulos del nucleo importa `qdk.estimator` (Azure QRE) ni `cudaq`
(NVIDIA CUDA-Q). Esos dos SI se tocaron en este proyecto -- pero en vez
de eliminarlos, se organizaron explicitamente como lo que son: **comparativas
de validacion externa**, aisladas en su propia carpeta,
[`validation-comparatives/`](validation-comparatives/README.md), separadas
del nucleo vendor-independiente:

- **`validation-comparatives/pipeline_v2_azure_qre_comparative.py`**
  (antes `pipeline_v2.py`) -- usa Azure QRE (`qdk.estimator`) para
  cruzar contra nuestro propio estimador. Ese cruce (documentado en
  `cross_validate_three_way.py` y en la seccion sobre reuso de qubits)
  fue lo que nos llevo a construir `open_resource_estimator.py`: un
  estimador propio que no depende de que Azure siga disponible,
  gratuito, o con la misma API -- y que ademas corrige un caso donde el
  modelo de Azure se rompe (reuso de qubits, resultado invertido).
- **`validation-comparatives/cudaq_reuse_comparative.py`** -- prueba
  directamente si CUDA-Q Logical (NVIDIA) hace reuso temporal automatico
  de qubits (resultado: no). Consolida y reemplaza los scripts sueltos
  de la sesion original (`test_cudaq_reuse.py`, `test_cudaq_reuse3.py`),
  ahora en un solo archivo, verificado de nuevo antes de moverlo, con su
  proposito de "comparativa externa" declarado en el docstring en vez de
  implicito.

Ambos scripts REQUIEREN instalar el paquete del vendor correspondiente
para poder ejecutarse -- eso es, precisamente, lo que los hace
comparativas de validacion y no parte del nucleo. Si se borran
`qdk.estimator` y `cudaq` del entorno, `qasm_tool.py` (y todo lo que
esta fuera de `validation-comparatives/`) sigue funcionando exactamente
igual: perfilado, recomendaciones, estimacion FTQC con fabricas, y
simulacion de expectativa, todo intacto.

## Se probo el pipeline contra el experimento de "utilidad cuantica" de IBM (Kim et al. 2023, Nature)

[Kim et al. 2023](https://www.nature.com/articles/s41586-023-06096-3),
"Evidence for the utility of quantum computing before fault tolerance",
corrio 127 qubits reales (heavy-hex, la MISMA familia de dispositivo que
`FakeTorino` usado en todo este proyecto), hasta 60 capas de compuertas
de 2 qubits (2,880 CNOTs), simulando la evolucion de Trotter de un
modelo de Ising 2D con campo transverso, mitigado con Zero-Noise
Extrapolation (ZNE) -- QEM, no QEC.

No podiamos reproducir su circuito exacto a 127 qubits (el punto del
paper es que esa escala esta mas alla de simulacion clasica exacta), asi
que construimos un analogo honesto a menor escala: un parche real de 16
qubits del MISMO mapa de acoplamiento heavy-hex de IBM, extraido
programaticamente de `FakeTorino` (BFS desde un nodo de grado 3, no
dibujado a mano), coloreado en 3 capas (coincide con las "3 capas de CNOT
por paso de Trotter" del paper real) via `networkx`, con la misma
estructura Rx(2*theta_h)/Rzz(2*theta_J) a un angulo generico NO-Clifford
(theta_J=0.35).

**Un falso negativo real, encontrado al probar contra este circuito:**
el circuito dio `zz_cost_operator_motif_fraction=1.0` y, verificado
directamente con `full_reduce()`, exactamente 0% de reduccion real de
T-count -- consistente con los demas circuitos de operador de costo
diagonal de este proyecto. Pero `recommend_techniques()` seguia
devolviendo `zx_calculus: recomendada` SIN advertencia, porque la
condicion de activacion del patron QAOA (heredada de cuando el detector
solo buscaba grafos densos de grado alto) exigia `grado_maximo >= 4.5`,
y este circuito tiene grado 3 (limitado por la conectividad heavy-hex
misma). Es un falso negativo real, encontrado probando contra un
circuito publicado real, no solo sinteticos. Se elimino esa condicion de
grado/dispersion, dejando solo `motif_fraction >= 0.8` (ya suficiente
por si sola), y se re-valido contra TODOS los circuitos ya probados en
este proyecto: los dos circuitos estilo-IBM y todos los casos QAOA/Ising
anteriores ahora advierten correctamente, y ninguno de los 6 circuitos
de referencia no-diagonales (adder, Grover, QFT, aleatorio, Shor-like,
VQE-ansatz) volvio a dar falso positivo.

**Factibilidad estructural -- el contraste que este circuito deberia
mostrar:** corrido por `qasm_tool.py` contra el mismo mapa de
acoplamiento heavy-hex del que se extrajo, el circuito es (esperado, pero
ahora verificado) `feasible=True` -- a diferencia del circuito de
bandera Shor-like de este mismo proyecto, que no lo es. No es
casualidad: el circuito de IBM se diseño deliberadamente para calzar con
la conectividad heavy-hex.

**Lo que costaria este circuito exacto en FTQC completo:** el mismo
analogo de 16 qubits (T-count=465 a 15 pasos de Trotter) da, en el
perfil superconductor (p_2q=0.32%): distancia de codigo d=15, **12,854
qubits fisicos** (con costo de fabrica de Litinski incluido). Un
contraste util: el mismo computo que IBM corrio con 127 qubits ruidosos +
ZNE necesitaria mas de 12,000 qubits fisicos en FTQC completo con solo
16 qubits logicos -- reforzando, desde una familia de circuito
completamente distinta, la tesis central de ese paper: QEM y FTQC ocupan
regimenes de recursos genuinamente distintos.

**Reproduciendo el mecanismo de ZNE mismo, a pequeña escala:** la
metrica de "exito = cae en el soporte ideal" (seccion de demostracion de
bandera) es inutil aqui: el soporte ideal de este circuito cubre 99.8%
de los 2^16 estados base, asi que CUALQUIER resultado ruidoso "tiene
exito" por ese criterio -- un hallazgo honesto en si mismo (esa metrica
se diseño para circuitos dispersos tipo Shor, no para simulacion densa).
En su lugar, medimos un observable local real, `<Z4 Z5>`, con p_2q
escalado por ganancias {1, 1.5, 2, 3} (analogo simplificado y declarado
de gate-folding) y extrapolamos linealmente a ruido cero:

- Profundidad superficial (4 pasos de Trotter, 120 compuertas de 2 qubits):
  el sesgo por ruido fue menor que el ruido estadistico de 1,500 shots --
  sin beneficio visible, un resultado nulo honesto, no escondido.
- Profundidad igualada a la escala real del paper (60 capas de 2 qubits,
  15 pasos de Trotter): ideal=0.3312, crudo sin mitigar=0.2520
  (error=0.0792), ZNE lineal extrapolado=0.2854 (error=0.0458) -- una
  reduccion real de ~42% del error, el mismo mecanismo que reporta el
  paper, reproducido a una escala que podemos verificar contra
  simulacion exacta.

**El contrapeso honesto que esta comparacion no estaria completa sin
mencionar:** el marco de "utilidad cuantica" de Kim et al. fue
directamente cuestionado por [Tindall et al. 2023](https://arxiv.org/abs/2306.14887)
(PRX Quantum 2024), que reportan un metodo clasico de redes tensoriales
(belief propagation explotando la estructura tipo-arbol del enrejado
heavy-hex) "significativamente mas preciso" que el resultado cuantico
mitigado, para los mismos observables. No lo citamos para litigar esa
disputa, sino porque la etica de este proyecto es reportar el resultado
negativo junto al positivo: la leccion es metodologica -- tamaño de
circuito y entrelazamiento por si solos no garantizan intratabilidad
clasica, y cualquier herramienta de estimacion de recursos (incluida la
nuestra) que solo razone sobre conteo de qubits, profundidad o T-count,
sin buscar estructura explotable (correlaciones tipo-arbol, bond
dimension bajo), puede ser ciega exactamente a este tipo de atajo
clasico.

## Cerrando el circulo: el punto debil de Tindall convertido en metrica rutinaria del pipeline

El hallazgo de Tindall no es solo un contrapeso sobre un paper --
es evidencia de un hueco estructural en TODA herramienta de estimacion de
recursos probada en este proyecto, incluida la nuestra: tanto Azure QRE
como `open_resource_estimator.py` razonan solo sobre qubits, profundidad
y T-count, y ninguno pregunta jamas si el circuito tiene estructura
explotable (correlaciones tipo-arbol, bond dimension bajo) que haria
irrelevante todo el costo de QEC/FTQC estimado, porque un metodo clasico
de redes tensoriales resuelve el mismo problema mas barato. No nos
quedamos en citar el hueco: construimos la version mas barata y honesta
de esa pregunta que se puede calcular con lo que ya tenemos instalado (sin
agregar ninguna libreria nueva de redes tensoriales), y la integramos al
pipeline como metrica rutinaria y reusable, no como script aislado.

**Metodo**: para circuitos dentro de un presupuesto practico de
simulacion exacta (<=24 qubits), se calcula la entropia de entrelazamiento
de von Neumann (en bits) del ESTADO FINAL del circuito, sobre la
biparticion mas simple posible (primera mitad de qubits vs segunda mitad,
segun el orden del registro QASM), reportada como fraccion del maximo
teorico para ese corte (`min(n_a, n_b)` bits). Esto ya es un campo,
`final_state_entanglement_fraction`, dentro de `profile_circuit()`
(`technique_advisor.py`), y `recommend_techniques()` emite una nueva
bandera de confianza baja, `posible_atajo_clasico_bond_dimension`, cuando
esa fraccion cae debajo de 50% -- un punto medio declarado, no calibrado
contra un dataset etiquetado. Por encima de ese punto medio, la bandera
reporta el circuito como consistente con "ley de volumen" (dificil de
comprimir via redes tensoriales de bond dimension acotada), aunque
tampoco esto es una prueba formal. Se degrada con gracia (devuelve `None`,
sin senal) para circuitos de mas de 24 qubits, donde el statevector exacto
deja de ser practico.

**Resultados sobre los once circuitos de referencia del proyecto:**

| Circuito | Entropia final (% del maximo) | Lectura |
|---|---:|---|
| `kicked_ising_analog_16q_deep` | 59.5% | mas cerca de ley de volumen |
| `qaoa_farm` | 63.4% | mas cerca de ley de volumen |
| `qaoa_small_test` | 57.0% | mas cerca de ley de volumen |
| `random_circuit8` | 54.2% | mas cerca de ley de volumen |
| `kicked_ising_analog_16q` (superficial) | 41.7% | por debajo del punto medio -- marcado |
| `vqe_ansatz` | 39.5% | por debajo del punto medio -- marcado |
| `shor_like_sequential` (nuestro circuito insignia) | 29.2% | por debajo del punto medio -- marcado |
| `grover` | 28.8% | por debajo del punto medio -- marcado |
| `adder8` | ~0% (precision de maquina) | cero exacto -- mecanismo distinto |
| `qft10`, `qft6` | ~0% (precision de maquina) | cero exacto -- mecanismo distinto |

**La complicacion honesta que no se esconde:** la linea mas importante de
esa tabla es la nuestra: `shor_like_sequential.qasm`, el circuito que este
proyecto usa repetidamente como su ejemplo insignia de carga de trabajo
genuinamente dificil y relevante para FTQC, cae en apenas 29.2% del
entrelazamiento bipartito maximo -- exactamente en la misma zona de baja
entropia que esta misma metrica marca como posible atajo clasico, la
misma zona que Tindall et al. exploto contra el circuito de Kim et al.
No tenemos, en este proyecto, la herramienta para probar o descartar que
`shor_like_sequential.qasm` admite una solucion barata de redes
tensoriales; la afirmacion honesta es mas estrecha y, creemos, mas util:
**no podemos descartarlo actualmente**, y tampoco puede hacerlo ninguna
herramienta que razone solo sobre conteo de compuertas. Eso es exactamente
el hallazgo que se buscaba con este ejercicio: una complicacion sobre
nuestro propio circuito insignia, reportada con el mismo peso que un
resultado favorable.

**Los dos casos de cero exacto son un mecanismo distinto, verificado
directamente:** `adder8.qasm` y `qft10.qasm`/`qft6.qasm` muestran entropia
a precision de maquina (1e-14 a 1e-16), no solo baja -- y verificar por
que importa, porque es un mecanismo mas trivial que la compresion de
redes tensoriales de Tindall sobre un estado genuinamente entrelazado.
Para `adder8.qasm` se confirmo directamente que SI hay compuertas que
cruzan la biparticion elegida (81 de 273), descartando la explicacion
trivial de "las dos mitades nunca interactuan"; su conjunto de compuertas
(`tdg, h, cx, t`) es un circuito clasico-reversible descompuesto en
Toffolis, que actua sobre una entrada de estado de base computacional y
permanece en un estado de base computacional durante toda su ejecucion --
por lo que carga entropia de entrelazamiento cero por construccion,
independientemente de la profundidad o el numero de compuertas. Para
`qft10.qasm`/`qft6.qasm` (conjunto de compuertas `cx, u`), el cero exacto
refleja el caso especial conocido de que la Transformada de Fourier
Cuantica aplicada a una entrada todo-ceros produce un estado producto
exacto. Ambos son circuitos honestamente clasicos-simulables, pero por
una razon que esta metrica sola no puede distinguir de la compresibilidad
tipo Tindall -- otra limitacion declarada, no escondida: una lectura de 0%
o cercana a 0% en esta metrica significa "muy probablemente barato de
simular clasicamente", pero por si sola no dice si es porque el estado
factoriza trivialmente (como aqui) o porque comprime bien en una red
tensorial de bond dimension baja (como en Tindall et al.). Distinguir
esos dos casos requeriria inspeccionar la estructura real del estado, no
solo su entropia.

## Sintesis: que podemos decir ahora, con la comparacion contra un resultado externo real

Probar este pipeline contra un benchmark real, externo y revisado por
pares -- el experimento de "utilidad cuantica" de Kim et al. 2023 en
Nature, no solo nuestras propias familias sinteticas de circuitos --
cambia lo que podemos afirmar honestamente, en cinco puntos concretos:

1. **La arquitectura agnostica de hardware, QASM-como-entrada, se sostiene
   contra una familia de circuitos real que no disenamos nosotros.** Un
   subgrafo de 16 qubits extraido directamente del mapa de acoplamiento
   real de un dispositivo (`FakeTorino`), no dibujado a mano, corrio a
   traves del mismo `qasm_tool.py` sin modificar usado para nuestros
   circuitos sinteticos, en los tres perfiles de hardware, sin codigo
   especifico por circuito.
2. **Esa misma familia de circuitos real expuso y corrigio un falso
   negativo genuino** en `recommend_techniques()` (la condicion
   `qaoa_like_pattern` con compuerta de grado), porque nuestro conjunto de
   prueba sintetico nunca incluyo un circuito disperso (grado maximo 3) de
   costo diagonal -- la correccion generaliza el detector mas alla de
   grafos densos tipo-QAOA a cualquier capa Hamiltoniana diagonal unica.
3. **Nuestra propia estimacion de costo FTQC sobre este circuito
   (12,854 qubits fisicos) cuantifica exactamente por que IBM lo corrio
   como experimento NISQ+QEM en vez de esperar a tolerancia a fallas** --
   la decision QEM-vs-FTQC en la que se basa todo este proyecto no es
   hipotetica para este circuito, es la decision que IBM realmente tomo.
4. **El mecanismo de ZNE en si mismo es reproducible y cuantificable a
   una escala que podemos verificar completamente contra simulacion
   exacta**, no solo citable del paper: un resultado nulo genuino a
   profundidad superficial (el sesgo de ruido por debajo del piso de
   ruido estadistico) y una reduccion genuina de ~42% del error a
   profundidad igualada a la escala del paper.
5. **El contrapeso de Tindall et al. es el hallazgo mas consecuente de
   esta comparacion**: muestra que tamaño de circuito y entrelazamiento
   medido por si solos no establecen intratabilidad clasica, y que este
   punto ciego es compartido por toda herramienta de estimacion de
   recursos probada en este proyecto, incluida la nuestra. No lo tratamos
   como una razon para descartar la comparacion, sino como su entrega
   principal -- motivo una nueva metrica rutinaria de baja confianza
   (fraccion de entrelazamiento, seccion anterior) ya integrada al
   pipeline, y el resultado mas importante de esa metrica fue una
   complicacion sobre nuestro propio circuito insignia, reportada
   honestamente en vez de omitida.

## Hacia teoria nueva: tres afirmaciones demostrables, no solo reglas calibradas

La mayor parte de este proyecto son reglas empiricas honestamente
calibradas -- correcto para una herramienta de validacion, pero no es
teoria nueva. Se tomaron los patrones mas fuertes encontrados hasta
ahora (el punto ciego de ZX-calculus, el punto ciego de entrelazamiento,
y el intercambio costo/confiabilidad calibrado que usan las secciones de
decision y QEC parcial) y se pregunto, para cada uno, si se pueden
elevar a una afirmacion demostrable. Dos si se pueden, uno solo
parcialmente -- y ese limite se reporta con la misma honestidad que el
resto del proyecto. Las tres se apoyan en piezas de teoria distintas e
independientes: la Proposicion 1 es consecuencia directa de la teoria de
polinomios de fase de ZX-calculus (Kissinger & van de Wetering) mas
independencia lineal de caracteres de paridad (analisis de
Fourier/Walsh-Hadamard booleano); la Proposicion 2 se apoya en el teorema
de Markov & Shi sobre treewidth y costo de simulacion de redes
tensoriales; la Proposicion 3, agregada en esta revision, no necesita
ningun teorema externo -- es consecuencia algebraica directa de las dos
formulas cerradas (qubits fisicos del codigo de superficie, y el ajuste
propio Stim+PyMatching de tasa de error logico) que este proyecto ya
usa empiricamente pero nunca habia demostrado nada sobre ellas.

### Proposicion 1 (demostrada): condicion estructural para que ZX-calculus no sirva

La regla empirica ya validada (`zz_cost_operator_motif_fraction >= 0.8`
=> 0% de reduccion medida de T-count, verificado en 24/24 instancias
QAOA y en el circuito real derivado de heavy-hex de IBM) resulta ser
consecuencia directa de la propia teoria de polinomios de fase en la que
se apoya ZX-calculus (Kissinger & van de Wetering), y se puede enunciar
y demostrar:

**Proposicion 1.** Sea `C` un circuito Clifford con una unica capa de
rotaciones diagonales de 2 qubits `{exp(i*theta_e*Z_a*Z_b) : e=(a,b) en
E}` sobre un conjunto de pares de qubits `E` (escritas como compuerta
diagonal nativa, o como el sandwich `CX(a,b); RZ(2*theta_e); CX(a,b)`),
con todas las aristas de `E` distintas entre si y todo `theta_e != 0`.
Entonces `full_reduce()` no puede reducir el T-count de `C` por debajo
de `|E|`, y `|E|` es tambien el minimo numero de terminos de rotacion
`Z_aZ_b` de una sola arista necesarios para realizar la misma unitaria
diagonal.

**Demostracion (resumen).** En la representacion de polinomio de fase
que usan las reescrituras de ZX-calculus, cada termino diagonal
corresponde a un "phase gadget" adjunto a la paridad (funcion booleana
lineal) `x_a XOR x_b`. Las reglas que reducen T-count funcionan
FUSIONANDO dos gadgets que comparten una paridad IDENTICA; ninguna otra
reescritura de `full_reduce()` cambia el T-count. Como las aristas de
`E` son distintas entre si, ningun par de gadgets comparte paridad aqui,
asi que ninguna fusion aplica jamas -- el T-count no cambia. Para la
segunda parte: las funciones `x -> (-1)^(x_a XOR x_b)`, una por arista,
son caracteres de `F_2^n` restringidos a elementos distintos, y los
caracteres distintos de `F_2^n` son linealmente independientes como
funciones reales sobre `{0,1}^n` (hecho estandar de analisis de Fourier
booleano / Walsh-Hadamard). La unitaria diagonal que implementa `C` es
una combinacion lineal no nula, con coeficientes `{theta_e}`, de
exactamente esas `|E|` funciones linealmente independientes -- por
independencia lineal, ningun conjunto de menos de `|E|` terminos puede
reproducir esa combinacion para `theta_e` genericos. QED.

**Corolario.** La regla empirica de `technique_advisor.py` no es solo
una heuristica calibrada: para cualquier circuito cuyo contenido
no-Clifford sea exactamente una capa de rotaciones `ZZ` diagonales de
aristas distintas (operadores de costo estilo QAOA, capas de Trotter de
modelos de Ising, o cualquier otra capa Hamiltoniana diagonal unica),
0% de reduccion de T-count por `full_reduce()` esta GARANTIZADO, no solo
observado. Esto tambien explica por que la regla generalizo
correctamente de grafos QAOA densos al circuito heavy-hex disperso
(grado maximo 3): la demostracion nunca uso densidad ni grado del grafo,
solo que las aristas sean distintas y el mecanismo de fusion -- exactamente
la correccion que se hizo empiricamente en este proyecto, ahora mostrada
como la generalizacion correcta por una razon estructural, no por
casualidad.

**Alcance declarado (para no sobre-reclamar):** esto es sobre un sistema
de reescritura especifico (fusion de phase gadgets de ZX-calculus) y una
familia de circuitos especifica (una capa diagonal de aristas
distintas). NO afirma que `|E|` T-gates sea el minimo global sobre
CUALQUIER circuito Clifford+T que implemente la misma unitaria (una
afirmacion mas fuerte que necesitaria la maquinaria de particion
matroidal de Amy, Maslov & Mosca); afirma solo que este sistema de
reescritura especifico y practico no puede hacerlo mejor -- que es
exactamente lo que la herramienta necesita saber antes de gastar computo
en `full_reduce()`.

### Proposicion 2 (parcial): treewidth como condicion necesaria, no suficiente

La metrica de entropia bipartita (seccion anterior) es un proxy
empirico honesto, sin garantia formal en ninguna direccion. Existe una
alternativa con fundamento teorico real en la literatura de redes
tensoriales:

**Teorema (Markov & Shi, SIAM J. Comput. 2008).** Si la red tensorial
que representa un computo cuantico tiene treewidth `<= w`, el computo se
puede simular clasicamente en tiempo `poly(tamano del circuito) * 2^O(w)`.

Se implemento una aplicacion directa de este teorema
(`treewidth_simulability.py`): para cada circuito de referencia, se
calcula el GRAFO DE INTERACCION ESTATICO (qubits como vertices, una
arista por cada par de qubits que comparte alguna compuerta de 2 qubits
en algun momento, colapsado en el tiempo), y se acota su treewidth por
arriba usando dos heuristicas de aproximacion estandar de `networkx`
(`min_degree`, `min_fill_in`), reportando la menor de las dos -- cada
heuristica devuelve una descomposicion en arbol VALIDA, asi que su ancho
es una cota superior legitima del treewidth real, sea o no optima la
heuristica.

**El limite de alcance, declarado con honestidad.** El teorema esta
demostrado para la red tensorial del circuito COMPLETO, que codifica
tiempo ademas de topologia; el grafo de interaccion estatico que
calculamos colapsa todos los pasos de tiempo en una sola topologia. Esto
hace que la cota calculada sea exacta y valida SOLO para una capa de
profundidad O(1) sobre esa topologia -- una afirmacion real y demostrada
sobre la CONECTIVIDAD DEL HARDWARE, independiente de la profundidad del
circuito, pero NO por si sola una cota valida del costo de simular un
circuito arbitrariamente profundo sobre esa topologia. La razon fisica
de que una cota de topologia estatica no se extienda automaticamente a
profundidad arbitraria es exactamente lo que ya medimos directamente: el
entrelazamiento puede crecer con la profundidad incluso sobre una
topologia fija, dispersa, de treewidth bajo, hasta saturar el limite de
ley de volumen. El treewidth de la topologia estatica es entonces una
condicion NECESARIA (estructural, topologica, demostrada) para que
exista un atajo clasico impulsado por la topologia, pero -- a diferencia
de la Proposicion 1 -- no afirmamos que sea suficiente para circuitos
profundos; eso requiere combinarla con la evidencia (mas debil, empirica)
del crecimiento de entrelazamiento.

**Resultados sobre los circuitos de referencia, lo que hace esto
concreto y no abstracto:**

| Circuito | treewidth (estatico) | 2^tw (cota por capa) | Fraccion de entropia |
|---|---:|---:|---:|
| `kicked_ising_analog_16q_deep` | 1 | 2 | 59.5% |
| `vqe_ansatz` | 1 | 2 | 39.5% |
| `adder8` | 2 | 4 | ~0% |
| `grover` | 4 | 16 | 28.8% |
| `qaoa_farm` | 4 | 16 | 63.4% |
| `qft6` | 5 | 32 | ~0% |
| `random_circuit8` | 5 | 32 | 54.2% |
| `shor_like_sequential` | 8 | 256 | 29.2% |
| `qft10` | 9 | 512 | ~0% |

La fila mas informativa es `kicked_ising_analog_16q_deep`: su topologia
estatica (un subgrafo real derivado de heavy-hex) tiene treewidth 1 -- el
puntaje estructural mas favorable de todos los circuitos probados,
exactamente consistente con la conectividad dispersa y tipo-arbol de
heavy-hex que Tindall et al. exploto -- pero su fraccion de
entrelazamiento MEDIDA es 59.5%, la mas alta de cualquier circuito de
esta tabla, mas cerca de ley de volumen que cualquier otro circuito de
referencia. Esto es exactamente el caso que el limite de alcance
declarado de la Proposicion 2 predice y que el mecanismo de la
Proposicion 1 no puede abordar: una condicion topologica favorable y
demostrada NO se mantiene favorable por si sola al crecer la
profundidad. Leer las dos metricas juntas, no una sola, es el aporte
honesto real de esta seccion: `shor_like_sequential.qasm` (treewidth 8,
entropia 29.2%) no tiene ni la garantia topologica mas fuerte ni una
lectura de entrelazamiento saturada -- exactamente por eso no se pudo
descartar un atajo clasico para nuestro propio circuito insignia, y esta
tabla muestra precisamente por que ningun numero solo -- ni treewidth
solo, ni entropia sola -- hubiera resuelto la pregunta.

## Formalizando QEM-vs-FTQC como un problema de optimizacion con restricciones (con una brecha de QEC explicita)

A diferencia de las Proposiciones 1 y 2, esto NO es un teorema -- es una
FORMALIZACION: convertir la comparacion if/else que ya existia en este
proyecto (`decide_regime()` en `comparative_benchmark_multi.py`,
comparaba tiempo esperado de QEM vs FTQC caso por caso) en un problema de
decision explicito y auditable, construido enteramente con piezas que
este proyecto ya valido por separado: nuestro propio estimador FTQC con
fabricas de Litinski, la tasa de exito de QEM medida empiricamente
(simulacion ruidosa real, no una formula), y el certificado de atajo
clasico de las Proposiciones 1-2.

**Definicion.** Dado un perfil de circuito `P=(n, tcount, tw, ef)`
(qubits, T-count, treewidth estatico, fraccion de entrelazamiento
final), un contexto de hardware (`p_2q`, presupuesto fisico de qubits
`B`), y un umbral de confiabilidad objetivo `epsilon en (0,1]`:

```
regimen* = argmin_{r en {QEM, FTQC}} Costo(r)
           sujeto a  Confiabilidad(r) >= epsilon  y  Costo(r) <= B
```

con `Costo(QEM) = n` (qubits fisicos, sin overhead de codigo),
`Confiabilidad(QEM) = ` fraccion de exito MEDIDA por
`simulate_expectation()` (una cantidad empirica, no una formula),
`Costo(FTQC) = physicalQubits` de nuestro propio estimador con fabricas,
y `Confiabilidad(FTQC) = 1 - error_budget` (se logra por construccion si
es factible). Antes de resolver este problema, se consulta primero el
aviso de atajo clasico de las Proposiciones 1-2 (no vinculante). **Cuando
el conjunto factible es vacio** -- ni QEM alcanza epsilon, ni FTQC cabe
en `B` -- se reporta una BRECHA CUANTIFICADA en vez de forzar una
respuesta: exactamente el espacio que un esquema de QEC parcial (no
completamente tolerante a fallas) tendria que llenar, y que este proyecto
no tiene estimador para cubrir todavia.

**Ejecutado, no solo definido** (`regime_decision.py`), sobre los
circuitos de referencia (`B=1,121` qubits fisicos, el procesador
superconductor real mas grande publicamente reportado -- IBM Condor,
citado solo como referencia de orden de magnitud; `epsilon=0.5`, el
mismo umbral de usabilidad ya usado en el dashboard):

| Circuito | QEM (confiabilidad) | Qubits FTQC | Elegido |
|---|---:|---:|---|
| grover, qft6/10, qaoa\_\*, random8, vqe | 1.00 | 5.4k-10.1k | QEM |
| adder8 | 0.75 | 13.8k | QEM |
| **shor\_like\_sequential (insignia)** | **0.00** | 29.7k | **BRECHA** |

**El resultado honesto: mayormente un resultado nulo, con una brecha
real.** A la escala de qubits de los circuitos de referencia de este
proyecto, el costo de FTQC nunca cabe en el presupuesto declarado, asi
que el procedimiento formalizado elige QEM por defecto en ocho de nueve
circuitos -- una limitacion declarada del conjunto de referencia (muy
pequeño para exhibir un cruce genuino a favor de FTQC; el barrido
separado de `ftqc_crossover_sweep.py` ya mostro que esos cruces si
existen a otras combinaciones de parametros). El unico circuito donde la
formalizacion realmente aporta es nuestro propio insignia,
`shor_like_sequential.qasm`: la confiabilidad de QEM medida es
exactamente 0% en este perfil de hardware, FTQC cuesta 29,654 qubits
fisicos (~26x el presupuesto declarado), y el procedimiento reporta esto
directamente como una BRECHA sin resolver -- confiabilidad corta por
0.5, qubits cortos por 28,533 -- en vez de elegir por defecto cualquier
regimen que corra sin error. Que sea exactamente el mismo circuito que
las Proposiciones 1-2 no pudieron descartar como atajo clasico no es una
coincidencia fabricada: es lo que pasa cuando un procedimiento de
decision formal se aplica honestamente al circuito que este proyecto
repetidamente uso como su ejemplo mas dificil, en vez de solo a
circuitos elegidos porque la respuesta se ve limpia.

## Un primer intento de llenar la brecha: barrido de QEC de distancia parcial

`regime_decision.py` deliberadamente se detenia en reportar la brecha en
vez de adivinar un regimen para el que este proyecto no tiene estimador.
Se dio un paso concreto para intentar cerrarla, en vez de dejarla solo
declarada: las dos piezas que el estimador FTQC completo ya usa (nuestro
propio ajuste de tasa de error logico con Stim+PyMatching, y la formula
de qubits fisicos del codigo de superficie) no requieren, en si mismas,
resolver para la distancia `d` GRANDE que exige un presupuesto de error
estricto. `solve_min_distance()` solo busca ese `d` grande; nada impide
BARRER las mismas dos formulas desde `d` chico hacia arriba, para ver si
una distancia parcial (insuficiente para tolerancia a fallas asintotica,
pero suficiente para bajar el error logico por debajo del error fisico
crudo, a un costo de qubits mucho menor) alcanza el umbral de
confiabilidad dentro del presupuesto.

**Metodo, y su limite honesto** (`partial_qec_sweep.py`): para cada `d`
impar de 3 a 51, se calcula `costo(d) = n*(2d^2+3d-1) + qubits de
fabrica de Litinski(d)` y `confiabilidad(d) = (1-p_logico(d))^tcount`
usando el ajuste ya validado del proyecto, y se reporta el `d` de menor
costo que cierra el umbral dentro del presupuesto, o un fallo honesto si
ninguno lo logra. **Esto NO es un esquema de QEC parcial validado** --
es una extrapolacion de un modelo de umbral de codigo de superficie
calibrado (via Stim+PyMatching) en el rango de distancias que se usa
para las estimaciones de FTQC completo, evaluado aqui a distancias por
debajo de ese rango calibrado; no se verifico por separado si el ajuste
se mantiene preciso tan abajo, y se declara asi en vez de presentar el
barrido con mas certeza de la que tiene.

**Resultado sobre el circuito de la brecha**
(`shor_like_sequential.qasm`, n=16, tcount=3,930, p_2q=0.0032): la
confiabilidad cruza el umbral de 0.5 entre d=11 (41%, 7,046 qubits) y
d=13 (68%, 9,734 qubits) -- aproximadamente UN TERCIO del costo de la
estimacion de tolerancia a fallas completa (29,654 qubits a d=23) para
el mismo umbral de confiabilidad usado en el resto del proyecto. Pero
9,734 qubits sigue siendo ~9x el presupuesto declarado de 1,121, asi que
ni siquiera este camino mas barato cierra la brecha a este presupuesto;
la salida honesta del barrido es `BRECHA_SIN_CERRAR`, no un falso
positivo. El hallazgo util es mas estrecho y, creemos, mas util que
cualquiera de los dos extremos solos: la distancia parcial compra una
reduccion real y cuantificada de ~3x en costo de qubits fisicos para el
mismo umbral de confiabilidad -- exactamente el tipo de numero que una
hoja de ruta de hardware de corto plazo necesita, aunque todavia no
alcance para hacer `shor_like_sequential.qasm` ejecutable dentro de un
presupuesto de qubits realista hoy.

### Reformulando la pregunta: no pasa/no-pasa, sino en que cuartil de confianza cae

Un presupuesto duro no tiene por que dar una respuesta binaria. Se
agrego una segunda consulta (`best_feasible_within_hard_budget()`) que
pregunta algo distinto y mas honesto: dado que el presupuesto `B` es
fijo e innegociable, ¿cual es la MEJOR confiabilidad alcanzable en
absoluto (el `d` mas grande que todavia cabe en `B`, sin exigir que
cumpla ningun umbral de entrada), y en que cuartil de confianza cae --
`[0,25%)` baja, `[25,50%)` baja-moderada, `[50,75%)` moderada,
`[75,100%]` alta?

Barriendo el presupuesto mismo para `shor_like_sequential.qasm`:

| Presupuesto B | Mejor d | Confiabilidad | Cuartil |
|---:|---:|---:|---|
| 1,121 (declarado) | 3 | ~0% | baja |
| 2,000 | 5 | ~0% | baja |
| 5,000 | 9 | 12.6% | baja |
| 8,000 | 11 | 40.8% | baja-moderada |
| 10,000 | 13 | 67.8% | moderada |
| 15,000 | 15 | 84.6% | alta |

Al presupuesto declarado de este proyecto (1,121 qubits), la respuesta
honesta no es "ningun codigo ayuda" en abstracto -- es que incluso el
parche de codigo de superficie mas barato posible (d=3) ya consume mas
de la mitad del presupuesto mientras entrega proteccion practicamente
nula (confiabilidad ~1e-12, cuartil mas bajo): el presupuesto es
sencillamente demasiado chico frente al T-count de este circuito para
que CUALQUIER distancia importe todavia. La confiabilidad recien sale
del cuartil mas bajo cuando el presupuesto llega a ~5-8x el declarado.
Esto es una afirmacion mas util que un solo bit de pasa/no-pasa: le dice
al lector exactamente cuanto tendria que crecer el presupuesto antes de
que valga la pena considerar QEC parcial para este circuito, en vez de
dejar "el presupuesto es insuficiente" como un callejon sin cuantificar.

## Proposicion 3: la monotonia del intercambio costo/confiabilidad calibrado, y por que el barrido es correcto (no solo funciona)

Las dos secciones anteriores (la formalizacion QEM-vs-FTQC y el barrido de
QEC parcial) dependian de dos supuestos que solo estaban declarados como
COMENTARIOS en el codigo, nunca demostrados: que el costo crece
monotonamente con la distancia `d` (por lo que un barrido ascendente que
se detiene en el primer valor que excede el presupuesto es seguro), y que
la confiabilidad tambien crece monotonamente con `d` (por lo que la `d`
mas grande que cabe en un presupuesto duro es automaticamente la de mejor
confiabilidad alcanzable, y la "escalera" de cuartiles de la tabla de
presupuesto no puede retroceder a medida que el presupuesto crece).

Resulta que ambos supuestos son DEMOSTRABLES directamente, sin necesidad
de ningun teorema externo de la literatura -- solo algebra aplicada con
honestidad a las dos formulas cerradas que este proyecto ya valido y usa:

- `Costo(d) = n*(2d^2+3d-1) + 22*d^2` (qubits de datos verificados contra
  Stim, mas una fabrica de destilacion de Litinski) es una parabola en
  `d` con coeficiente principal positivo y vertice en `d* < 0` -- por lo
  tanto es ESTRICTAMENTE CRECIENTE para todo `d >= 3`, sin importar `n`.
- `Confiabilidad(d) = (1 - A*(p/p_th)^((d+1)/2))^tcount`, usando nuestro
  propio ajuste Stim+PyMatching: como `p/p_th < 1`, el termino
  `(p/p_th)^((d+1)/2)` decrece estrictamente al crecer `d` (exponente
  creciente sobre una base menor a 1), asi que el error logico decrece y
  la confiabilidad crece estrictamente en `d`.

De esto se sigue, como corolario, dos cosas que antes eran solo supuestos
de diseño: (1) `sweep_partial_qec()` y
`best_feasible_within_hard_budget()` pueden detenerse en el primer cruce
de umbral en vez de barrer exhaustivamente hasta `d_max` -- es una
garantia de optimalidad DEMOSTRADA, no solo un atajo de eficiencia; y (2)
la escalera de cuartiles de confianza nunca puede bajar de cuartil al
aumentar el presupuesto -- es una propiedad estructural del modelo
calibrado, no una coincidencia de los seis valores de presupuesto que se
barrieron.

**Limite declarado, con la misma honestidad de siempre:** esto demuestra
que el ALGORITMO es correcto DADO el modelo calibrado -- NO valida que el
modelo en si sea preciso a distancias pequeñas, lo cual la seccion
anterior ya declara como extrapolacion fuera del rango calibrado. Un
modelo con constantes ajustadas incorrectamente en `d=3` seguiria
cumpliendo esta proposicion igual (la monotonia es una propiedad de la
FORMA funcional `A*r^((d+1)/2)`, no evidencia de que las constantes sean
precisas ahi) -- son dos preguntas independientes, y esta proposicion solo
responde la primera.

## Prueba de esfuerzo: escala, predictores aprendidos (ML), y costo operacional

Todo lo medido hasta aca se calibro sobre el dataset de 98 instancias / 9
familias (circuitos hasta 20 qubits) o sobre los 9 circuitos de
referencia. Quedaban 4 preguntas abiertas por ese alcance: ¿la regla
desplegada generaliza a circuitos genuinamente mas grandes que los usados
para calibrarla?, ¿combinar features con ML estandar mejora la prediccion
cuando se evalua con la MISMA honestidad LOFO que el resto del proyecto?,
¿que tan caro es correr este pipeline en la practica?, y ¿que implica todo
esto para planificar segun evolucione el hardware? Se respondieron las 4
con experimentos nuevos, no con mas afirmaciones.

**1) Generalizacion a escala.** Se generaron 21 circuitos nuevos, nunca
vistos, reusando los mismos generadores de familia del dataset original
pero a tamanos mayores -- hasta 32 qubits (GHZ), contra un maximo previo
de 20. Tres instancias de QFT (n=16,20,24) fallaron INICIALMENTE al
exportar a QASM por un bug de precision numerica (un valor de fase
demasiado pequeño se formatea de forma que pyzx lo lee como variable no
ligada) -- este bug, y dos mas que esta misma prueba de esfuerzo
encontro, se corrigieron en la misma sesion (ver el punto 3.5 mas abajo).
Con el ground truth ya disponible para las 21/21 instancias: la regla YA
DESPLEGADA, evaluada sobre las 19 que ademas corren en tiempo razonable
por el pipeline completo (ver 3.5 para por que 2 quedan afuera), acierta
73.7%, contra un baseline trivial de 57.9% en ese mismo conjunto -- mejor
que su propio resultado LOFO dentro de distribucion (70.4%, que no supera
baseline). Los dos fallos reales son informativos: fallo en las 4
instancias grandes de QAOA (el detector de motivo diagonal que acerto
24/24 en el dataset original no disparo aqui), y fallo en una instancia
aleatoria donde ZX-calculus de hecho AUMENTO el T-count (-3.8%) -- un caso
que la regla no tiene forma de anticipar.

**2) ¿Ayuda combinar features con Machine Learning, evaluado con
honestidad?** Una iteracion anterior de este proyecto entreno un arbol de
decision (85.1%) y una regresion logistica (88.8%) combinando varias
features, pero con validacion cruzada ORDINARIA de 5 pliegues -- no LOFO
-- sobre un dataset mas chico. Nunca se habia re-evaluado ese enfoque
combinado con la misma honestidad LOFO ya aplicada a la regla de una sola
feature. Se hizo aqui: ambos modelos, con TODAS las features estructurales
ya calculadas para las 98 instancias (incluyendo el costo de contraccion
de red tensorial, antes usado solo para una correlacion de 9 puntos, aqui
como feature real por instancia), evaluados con `LeaveOneGroupOut` real
por familia.

Resultado honesto, con un matiz genuino: **negativo, confirmado** --
combinar features con ML estandar, evaluado en familias nunca vistas, NO
supera el baseline mayoritario (62.2% regresion logistica, 52.0% arbol,
ambos por debajo de 66.3%) -- la regla de una sola feature ya desplegada
(70.4% LOFO) sigue siendo el mejor predictor honesto que generaliza a
familia encontrado en todo el proyecto. **Matiz, nuevo** -- esos MISMOS
modelos, entrenados una vez con las 98 instancias completas y evaluados en
los 19 circuitos grandes del punto 1 (instancias nuevas, pero de familias
YA vistas, solo que a una escala nunca vista), llegan a 94.7% -- superando
claramente tanto el baseline como la regla desplegada (73.7%) en ese mismo
conjunto. Son dos preguntas de generalizacion distintas: el ML aqui falla
al generalizar a una FAMILIA nueva, pero generaliza bien a una ESCALA
nueva de una familia conocida. Reportar los dos numeros, no solo el
favorable, es el punto.

**3) Costo operacional del propio decision layer.** Un decision layer solo
sirve si calcular su recomendacion es suficientemente barato para correr
antes de despachar un trabajo. Se midio el pipeline real de punta a punta
(`profile_circuit` + `recommend_techniques`) -- los numeros de abajo son
DESPUES de los 3 arreglos del punto 3.5: `vqe_ansatz.qasm` (8 qubits)
0.06s, `qft10.qasm` (10 qubits) 0.18s, `qaoa_farm.qasm` (17 qubits) 0.41s,
`adder8.qasm` (18 qubits) 0.99s -- y el circuito insignia
(`shor_like_sequential.qasm`, 16 qubits, miles de compuertas), que ANTES
tardaba 44.4s, ahora tarda 11.6s (-74%). El costo restante es el propio
parser QASM de pyzx (una libreria de terceros, no codigo de este
proyecto). Conclusion honesta: el pipeline ya es practico para correr en
linea, hoy, incluso en el circuito mas pesado de este proyecto.

**3.5) Los 3 bugs encontrados, corregidos en la misma sesion.**

- **Bug 1 -- `profile_circuit()` duplicado:** el patron usado en casi todo
  este proyecto (`p = profile_circuit(path)` seguido de
  `recommend_techniques(path)`) calculaba el perfil completo DOS VECES,
  porque `recommend_techniques()` siempre lo recalculaba internamente sin
  importar que el llamador ya lo tuviera. *Arreglo:* `recommend_techniques()`
  ahora acepta `profile=` opcional y se salta el recalculo si se lo dan;
  se actualizo el unico caller real con este patron
  (`qasm_tool.analyze_circuit`). *Efecto medido en el circuito insignia:*
  44.4s -> 11.6s.
- **Bug 2 -- asignacion de memoria exponencial en el rastreador QCP (la
  causa real de los cuelgues >60s):** en `qcp_bounded_tracker.py`, CADA
  grupo de qubits fusionado -- incluso uno que se iba a marcar
  `known=False` (abandonar el rastreo, el mecanismo que deberia
  mantener esto barato) -- reservaba sin condicion un vector de estado
  de tamano `2**qubits_del_grupo` en su constructor, ANTES de chequear
  si alguna vez se iba a usar. *Arreglo:* el constructor ahora recibe un
  flag `known` y se salta la reserva por completo cuando es `False`
  (ningun codigo lee `.state` para un grupo desconocido, asi que nunca
  hacia falta). *Efecto medido directo en el circuito que colgaba*
  (`Adder_n10`, 22 qubits): >60s (matado) -> 0.18s.
- **Bug 3 -- notacion cientifica ilegible en el QASM exportado:**
  `qiskit.qasm2.dumps()` escribe angulos muy pequeños en notacion
  cientifica (p.ej. `9.587...e-05`); el parser QASM de pyzx no la puede
  leer y falla con un error enganoso de "variable no ligada". Verificado
  directamente: el MISMO valor en notacion decimal plana parsea sin
  problema. *Arreglo:* una utilidad nueva (`qasm_safe_export.py`) que
  reescribe cualquier literal en notacion cientifica a su equivalente
  decimal antes de pasarle el texto a pyzx. *Efecto:* las 3 instancias de
  QFT que antes fallaban ahora parsean y miden correctamente -- ground
  truth disponible para 21/21 circuitos (antes 18/21).
- **Un cuarto hallazgo, encontrado al arreglar los 3 anteriores, distinto
  de un bug propio:** una vez que el Bug 2 dejo de esconderlo, aparecio
  un costo real aparte -- el chequeo exacto de entrelazamiento
  (`_final_state_entanglement_fraction`), ya limitado a <=24 qubits, es
  caro CERCA de ese propio limite para algunos circuitos (p.ej.
  `GHZ_n24`, solo 24 compuertas, igual tardo >40s) -- el cuello de
  botella parece ser memoria, no cantidad de compuertas. Se bajo el
  limite de 24 a 20 qubits, el mayor tamano confirmado rapido en la
  practica, en vez de dejar un limite que en teoria "degrada con
  gracia" pero en la practica colgaba el pipeline antes de llegar a esa
  degradacion.
- **Lo que queda abierto, honestamente:** el parser QASM de pyzx (una
  libreria de terceros) es lento para circuitos con muchos literales de
  fase DISTINTOS -- no por cantidad de qubits -- y QFT tiene O(n^2) de
  ellos. Medido directo: `QFT_n16` 2.3s, `QFT_n20` 25.2s, `QFT_n24` no
  termino en varios minutos. Esto no es un bug de este proyecto y queda
  fuera de alcance arreglarlo -- por eso la prueba de precision del punto
  1 excluye especificamente `QFT_n20`/`n24` (19 de 21 circuitos), aunque
  su ganancia real de ZX-calculus SI se pudo medir sin correr el pipeline
  completo.

**4) Que implica esto para planificar segun evolucione el hardware.**
Juntando los 3 puntos anteriores con el barrido de crossover y la tabla de
cuartiles de presupuesto: (i) la decision de regimen no es fija -- a medida
que el presupuesto declarado crece siguiendo la trayectoria de esa tabla
(1,121 -> 15,000), el mismo circuito insignia pasa del cuartil de menor
confianza al de mayor confianza SIN cambiar nada del circuito -- una hoja
de ruta de hardware y una hoja de ruta de reduccion de circuitos responden
la misma pregunta desde dos lados y deberian planificarse juntas; (ii)
como la Proposicion 3 demuestra que esa progresion no puede revertirse, un
equipo puede comprometerse a re-correr el barrido a medida que crece el
presupuesto confiando en que la confiabilidad solo mejora; (iii) como la
regla desplegada degrada de forma gradual, no catastrofica, a mayor escala
(15.8 puntos sobre baseline fuera de distribucion), sigue siendo razonable
consultarla automaticamente antes de despachar, siempre que sus dos modos
de fallo conocidos (QAOA grande y disperso, instancias raras de ganancia
negativa) se marquen para revision manual; y (iv) el hallazgo de costo
operacional identifico una prioridad de ingenieria concreta -- eliminar la
llamada duplicada a `profile_circuit()` y acotar el costo del rastreo QCP
-- y ambas se corrigieron en la misma sesion (punto 3.5), bajando la
latencia del circuito insignia en 74% (44.4s -> 11.6s) y eliminando por
completo los cuelgues >60s, dejando el parser QASM de pyzx (una libreria
de terceros) como el unico cuello de botella restante, ya identificado y
fuera de nuestro alcance corregir.

## ¿La medicion intermedia realmente entrega el reuso que este proyecto viene estimando?

Todo lo anterior mide el POTENCIAL de reuso a partir de estructura
estatica del circuito (`qubit_reuse_potential_fraction`,
`max_concurrent_qubits`) -- ningun circuito de referencia, hasta ahora,
usaba de verdad medicion intermedia + retroalimentacion clasica para
REALIZAR ese potencial. Se construyo uno real: Estimacion de Fase
Iterativa (IQPE, Griffiths & Niu 1996, tambien llamada "QFT
semiclasica") -- la tecnica de libro de texto para reemplazar una
lectura de fase basada en QFT de n qubits por UN SOLO ancilla reusado n
veces, con correcciones de fase clasicamente condicionadas que
sustituyen la red de fases controladas coherente de la QFT inversa --
exactamente el mecanismo que las metricas de reuso de este proyecto
fueron diseñadas para anticipar, y directamente relevante a los
circuitos tipo QFT/estimacion de fase (`qft6/10.qasm`,
`shor_like_sequential.qasm`) que ya se usan en todo el proyecto.

**Correctitud primero, no solo conteo de qubits.** Se implementaron
ambas versiones en Qiskit (`qc.if_test` para la retroalimentacion
clasica) y se verifico -- no se asumio -- que la version iterativa
reproduce EXACTAMENTE los mismos bits de salida que la version estandar,
al 100% de probabilidad medida sobre 200 shots, para cuatro fases de
prueba elegidas independientemente. El ahorro es entonces exactamente lo
que predice la teoria, e independiente de la precision: de 5 a 2 qubits
declarados-y-concurrentes para 4 bits de conteo (60% de reduccion), y --
a diferencia de cualquier otro caso de reuso medido antes en este
proyecto -- esta proporcion MEJORA mientras mas bits de precision se
pidan, porque la version iterativa siempre usa 2 qubits sin importar n.

**Una limitacion arquitectonica honesta que esto realmente expone.**
Este proyecto construyo su pipeline central sobre la promesa
"QASM-en-entrada, agnostico de hardware". Se probo esa promesa contra
este circuito real en vez de asumir que se sostiene, en dos niveles:

- **Exportacion**: Qiskit NO PUEDE serializar el circuito clasicamente
  condicionado a OpenQASM 2 en absoluto (`"OpenQASM 2 only supports
  register-equality conditions"`), aunque la exportacion a OpenQASM 3 si
  funciona.
- **Ingestion**: aunque `pyzx` -- el motor debajo de
  `technique_advisor.profile_circuit()`, el punto de entrada de TODO el
  pipeline de analisis de este proyecto -- documenta soporte para
  sentencias `if` de OpenQASM 3, pasarle ese archivo OpenQASM 3 exacto y
  valido FALLA (`TypeError: Phase expression 'out[0]' contains unbound
  variables`), y se confirmo que la falla no es especifica de pyzx sino
  que llega hasta `profile_circuit()` mismo.

El enfoque de este proyecto de "evitar codigo especifico de vendor, solo
tomar QASM" solo se habia puesto a prueba, hasta ahora, sobre circuitos
estaticos; los circuitos dinamicos reales con retroalimentacion clasica
-- exactamente el mecanismo cuyo beneficio las metricas de reuso de este
proyecto vienen estimando -- actualmente quedan FUERA de lo que el
pipeline puede ingerir de verdad. Se reporta esto como una limitacion de
alcance genuina y antes no descubierta, no como una nota al pie: la
herramienta que mide el potencial de reuso todavia no puede analizar la
clase de circuito que mas directamente lo realiza.
