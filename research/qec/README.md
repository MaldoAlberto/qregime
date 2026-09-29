# QEC real: surface code + decodificador (Stim + PyMatching)

Cierra una brecha declarada del proyecto: hasta este punto, todo lo
relacionado con "FTQC" pasaba por Azure QRE como estimador de recursos
(qubits fisicos, runtime) a partir de un modelo de error asumido. Aqui SI
se construye el circuito de sindromes de un codigo de superficie rotado
(`stim.Circuit.generated`) y se decodifica con Minimum Weight Perfect
Matching real (`pymatching.Matching.from_detector_error_model`), midiendo
la tasa de error logico observada, no asumida.

## Resultado medido (20000 shots por punto, 27 puntos)

Barrido de distancia de codigo `d in {3,5,7}` x tasa de error fisico
`p in [0.001, 0.10]`. Firma esperada de un codigo funcionando: por debajo
del umbral, mas distancia = menos error logico; por encima, lo contrario.

**Observado**: el cruce ocurre entre p=0.005 y p=0.008 -- es decir, un
umbral aproximado de **~0.8%-1.0%**, consistente con el umbral publicado
del codigo de superficie (~1% para ruido depolarizado, Fowler et al. 2012).
Ver `qec_threshold_plot.png` y los 27 puntos completos en
`qec_surface_code_results.json`.

## Limites declarados

- Ruido depolarizado uniforme (no el perfil de ruido de un backend real
  especifico -- eso es lo que si se usa en la parte QEM de este proyecto,
  via `real_two_qubit_error_rate` sobre FakeTorino).
- Simulacion clasica exacta de un circuito Clifford puro (Stim es exacto
  para esto, no una aproximacion) -- no es ejecucion en hardware real.
- **Actualizacion**: esta simulacion ya esta conectada a
  `../decision-layer/pipeline_v2.py` (`cross_validate_logical_error_rate`)
  como validacion cruzada del modelo analitico de Azure QRE -- de paso se
  encontro y corrigio un bug real: la funcion que llama a Azure QRE
  recibia `p_phys` (la tasa de error real medida del backend) pero nunca
  la usaba, siempre corria con un perfil de qubit generico fijo. Ya
  corregido. La validacion cruzada en si tiene resolucion estadistica
  limitada: con 5000 shots no se pueden medir tasas de error logico tan
  bajas como las que Azure QRE predice (ej. 3e-7), asi que por ahora solo
  confirma que ambos estan en el mismo orden de magnitud cuando la tasa
  esperada es medible, no una confirmacion numerica exacta.

## Reproducir

```
pip install stim pymatching
python3 qec_surface_code_experiment.py
```
