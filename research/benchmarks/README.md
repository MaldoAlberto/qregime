# Guia de validacion: tecnica x familia de algoritmo x metrica

Tabla estilo Metriq, pero construida sobre experimentos propios ejecutados
en este proyecto (codigo en `techniques/`, `qec/`, `decision-layer/`), no
sobre resultados enviados por la comunidad.

## CORRECCION IMPORTANTE (verificacion posterior, mismo dia)

Una version anterior de este documento reportaba **82.9%** de precision
leave-one-family-out para `spectral_sparsifiability` como predictor de si
ZX-calculus ayuda. Al re-verificar con un script limpio sobre el dataset
completo (98 instancias, 9 familias) y el mismo protocolo honesto (LOFO),
la cifra correcta es **70.4%** -- y ese 70.4% es EXACTAMENTE igual a la
precision de un baseline trivial que simplemente asume "SI ayuda" por
defecto (salvo el filtro exacto tcount_frac==0). Es decir: ninguna de las
features probadas en este proyecto (spectral_sparsifiability, grado
maximo del grafo, costo de contraccion de red tensorial, razon de angulos
de rotacion distintos, y varias combinaciones) supera de forma honesta ese
baseline trivial. Se corrige aqui explicitamente en vez de dejar la cifra
inflada -- el 82.9% original probablemente mezclo precision de
entrenamiento (sin held-out) con precision LOFO real.

| Metodo | Precision honesta (LOFO, 98 instancias) | Nota |
| --- | --- | --- |
| Baseline mayoria (asumir SI salvo tcount_frac==0) | **70.4%** | El techo real disponible hoy |
| Umbral fijo tcount_frac > 0.10 (regla original) | 70.4% | Empata con el baseline trivial |
| Regresion logistica: spectral_sparsifiability | 70.4% | Empata con el baseline trivial |
| Arbol de decision (tcount_frac + max_degree) | 49.0% | Peor que el baseline trivial (sobreajuste) |
| Regresion logistica (tcount_frac + max_degree) | 59.2% | Peor que el baseline trivial |
| Costo de contraccion tensorial (log2_flops) solo | 59.2% | Peor que el baseline trivial |
| Arbol (tcount_frac + max_degree + spectral_sparsifiability) | 67.3% | El mejor modelo combinado encontrado, aun asi peor que el baseline |

**Conclusion honesta**: el unico filtro EXACTO (no probabilistico) es
`tcount_frac == 0` (no hay margen no-Clifford, ZX-calculus no puede
ayudar, siempre correcto). Fuera de eso, la mejor estrategia disponible
hoy es asumir que SI ayuda por defecto, con una **advertencia de baja
confianza** para circuitos que muestran el patron estructural de QAOA
(grado maximo alto + dispersion espectral moderada-alta) -- el unico caso
sistematico donde ese default falla (0/24 instancias QAOA se benefician,
en todos los metodos probados esta sesion, sin excepcion).

## Ganancia real de ZX-calculus por familia (full_reduce, `techniques/zx_calculus`)

| Familia | # instancias | Ganancia T-count (promedio) | Rango |
| --- | --- | --- | --- |
| QPE | 5 | 77.5% | 60.0% - 90.5% |
| QFT | 10 | 45.3% | 22.2% - 55.6% |
| Adder (Cuccaro) | 7 | 42.9% | 42.9% (constante) |
| ToffoliChain | 7 | 21.7% | 14.3% - 25.7% |
| VQEansatz | 12 | 21.0% | 15.8% - 25.2% |
| Grover | 5 | 18.4% | 0.0% - 38.5% |
| Random | 21 | 15.3% | 7.4% - 22.3% |
| GHZ | 7 | 0.0% | 0.0% (trivial, sin T-gates) |
| **QAOA** | 24 | **0.0%** | 0.0% (constante, en las 24 instancias) |

## Otras 6 tecnicas: medidas solo en 4 circuitos de referencia (no en las 9 familias completas)

| Tecnica | QFT10 | Sumador8 | QAOA/farm | random8 | Medido en mas familias |
| --- | --- | --- | --- | --- | --- |
| Teletransportacion RZZ | Ayuda (grado prom. bajo) | Ayuda | Depende del grado | Ayuda | No todavia |
| AND-gate + descomputo por medicion | Sin ancillas candidatas claras | Ancillas reales, requiere verificar reuso a mano | Sin ancillas candidatas | Variable | No todavia |
| QCP acotado (n_pcm=4) | Fraccion rastreable parcial | Fraccion rastreable parcial | Entrelaza globalmente (0%) | Fraccion rastreable parcial | No todavia |
| Circuit cutting (`find_cuts` real) | No medido | 2 cortes / 256x overhead | 18-46 cortes / 1e17-1e44x overhead | No medido | No todavia |
| RQAOA (reproduccion de correlacion) | No aplica | No aplica | r=-0.95 (opuesto a la hipotesis inicial) | No aplica | No todavia |
| Rondas RQAOA-style (extension propia) | 20-28 rondas (n_pcm=4-8) | 20-28 rondas | 22-33 rondas | Baja con presupuesto mayor | No todavia |

## Simulabilidad clasica (nuevo)

Ver `techniques/tensor_network_simulability/` (costo de contraccion vs
spectral_sparsifiability, r=0.16, y vs ganancia ZX-calculus, r=0.66 con
solo 9 puntos -- exploratorio) y `techniques/qaoa_ising_ground_state/`
(QAOA p=1,2,3 nunca supera a simulated annealing clasico, 0/27 instancias
en el barrido de profundidad).
