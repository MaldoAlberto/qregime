# Simulabilidad clasica via redes tensoriales (cotengra)

La otra mitad de la propuesta original del usuario (lattice + minima
energia): cada circuito se representa como una red tensorial real (un
tensor por compuerta, post-transpile a REAL_BASIS para garantizar solo
compuertas de 1-2 qubits), y se usa `cotengra.HyperOptimizer` para buscar
el orden de contraccion que MINIMIZA el costo (flops) -- el algoritmo por
defecto de cotengra usa simulated annealing internamente, el mismo tipo de
optimizacion combinatoria que el experimento de Ising/MaxCut de este
proyecto (`../qaoa_ising_ground_state/`).

Se mide el costo de UNA amplitud de transicion (entrada |0...0>, salida a
un bitstring fijo) -- la metrica estandar en la literatura de "quantum
advantage" (Boixo et al. 2018) para caracterizar que tan dificil es
simular un circuito clasicamente.

## Resultado medido (1 instancia representativa por familia, 9 familias)

- **spectral_sparsifiability NO correlaciona con el costo de contraccion
  tensorial** (r=0.16, practicamente nulo) -- estas dos metricas, pese a
  que ambas describen "que tan disperso/estructurado" es el circuito,
  miden propiedades estructurales DISTINTAS. Esto es un resultado honesto:
  la hipotesis de que serian intercambiables NO se confirma con esta
  muestra.
- **El costo de contraccion SI muestra una correlacion moderada con la
  ganancia real de ZX-calculus** (r=0.66) -- circuitos mas caros de
  contraer tienden a tener mas ganancia real de reduccion de T-count. Esto
  es sugerente, no un resultado validado: son 9 puntos (una instancia por
  familia), sin leave-one-family-out posible con esta muestra.

Ver `tensor_network_correlation.png` para ambos paneles.

## Limites declarados

- El costo reportado depende del heuristico de busqueda de cotengra
  (annealing con repeticiones acotadas, `max_repeats=24`), no es el optimo
  global garantizado del problema NP-duro de particionado de hipergrafos.
- Una sola instancia por familia -- no es una prueba estadistica, es una
  primera exploracion de si esta metrica vale la pena medir a mayor
  escala. Ampliar a mas instancias por familia (como en las 9 familias del
  dataset completo de ZX-calculus) es el siguiente paso natural antes de
  proponer esta metrica como parte de la capa de decision.

## Reproducir

```
pip install cotengra quimb opt_einsum
python3 tensor_network_simulability_experiment.py
```
