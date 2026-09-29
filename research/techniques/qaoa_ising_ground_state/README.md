# QAOA como problema de Ising: estado fundamental real vs QAOA vs annealing clasico

Responde a la propuesta del usuario de mapear el circuito (el problema que
codifica QAOA) a un modelo de Ising y buscar su energia minima. Se hizo
sobre los MISMOS 24 grafos aleatorios ya usados para construir los
circuitos QAOA de `build_circuit_dataset.py` (n=6-16, densidad 0.2-0.8).

Tres metodos, todos reales, sobre el mismo problema MaxCut/Ising:

1. **Exacto** (fuerza bruta, 2^n asignaciones, factible para n<=16).
2. **Simulated annealing clasico** (sin nada cuantico) -- algoritmo estandar
   de optimizacion combinatoria.
3. **QAOA p=1 real**, con parametros optimizados de verdad (scipy sobre el
   valor esperado exacto via Statevector), y tambien con los parametros
   FIJOS (gamma=0.7, beta=0.4) que el proyecto ya usaba en el dataset.

## Resultado medido (24/24 instancias)

- **Simulated annealing iguala el optimo exacto en 24/24 instancias**
  (ratio = 1.000 siempre) -- estos grafos son pequenos y faciles para un
  metodo clasico simple.
- **QAOA p=1 optimizado da un ratio promedio de 0.740**, y **nunca supera**
  al annealing clasico en ninguna de las 24 instancias.
- Los parametros FIJOS que el dataset original usaba (0.7, 0.4, sin
  optimizar) dan resultados peores todavia que el QAOA optimizado -- una
  limitacion adicional honesta: los circuitos QAOA de este proyecto nunca
  tuvieron sus parametros ajustados, asi que su "ganancia real" reportada
  en otras partes del repositorio corresponde a ese caso no optimizado.

Ver `qaoa_vs_classical_ground_state.png` para la curva completa por
densidad y tamano.

## Por que se reporta asi, sin suavizarlo

Este es exactamente el resultado esperado y bien documentado en la
literatura de QAOA (p=1 en grafos aleatorios pequenos rara vez supera a un
buen heuristico clasico) -- no es una falla del proyecto, es la firma
conocida de que QAOA de profundidad baja no ofrece ventaja en instancias
de este tamano. Se declara aqui explicitamente porque es relevante para
cualquiera que use este repositorio para decidir si vale la pena correr
QAOA en instancias parecidas: en este rango de tamano, un annealer clasico
simple ya resuelve el problema exactamente.

## Extension: profundidad p=1,2,3 (`qaoa_depth_sweep_experiment.py`)

¿El patron se mantiene al aumentar la profundidad? Se probo p=1,2,3 sobre
9 grafos (subconjunto mas chico por costo de optimizacion). Resultado:
el ratio de aproximacion mejora ligeramente con la profundidad (0.708 en
p=1 -> 0.738 en p=3, promedio), pero **QAOA no supera al annealing clasico
en NINGUNA de las 27 instancias probadas (0/27), en ninguna profundidad**.
Ver `qaoa_depth_sweep.png`.

## Reproducir

```
python3 qaoa_ising_ground_state_experiment.py
python3 qaoa_depth_sweep_experiment.py
```
