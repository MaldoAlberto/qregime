# Comparativas de validacion externa

Los dos scripts de esta carpeta NO son parte del nucleo del proyecto y
NO son dependencias de `qasm_tool.py` (verificado por imports reales,
ver la seccion "Dependencies and vendor independence" del paper y del
README principal). Existen exclusivamente para responder una pregunta
con codigo ejecutado: **¿coincide nuestro pipeline propio, vendor-independiente,
con lo que reportan herramientas de vendors externos?**

| Script | Vendor comparado | Pregunta que responde | Resultado |
|---|---|---|---|
| `pipeline_v2_azure_qre_comparative.py` | Microsoft (Azure Quantum Resource Estimator, `qdk.estimator`) | ¿Coincide el estimador FTQC propio (`open_resource_estimator.py`) en orden de magnitud con el de Azure? | Si, cruzados en `cross_validate_three_way.py`; y se encontro que el modelo de Azure se rompe (da resultados invertidos) al intentar modelar reuso de qubits -- por eso existe un estimador propio ademas de este |
| `cudaq_reuse_comparative.py` | NVIDIA (CUDA-Q Logical) | ¿CUDA-Q hace reuso temporal automatico de qubits logicos? | No, verificado directamente: `logical_qubits_peak` reporta 6 en vez de 3 en un circuito de dos fases sin solapamiento, con y sin medicion explicita |

**Requieren instalar el paquete del vendor correspondiente** (`qdk` /
Azure Quantum Development Kit, o `cudaq` de NVIDIA) para poder
ejecutarse -- eso es exactamente lo que los hace comparativas de
validacion y no parte del nucleo: el nucleo (`qasm_tool.py` y los
modulos que orquesta) sigue funcionando identico si estos dos paquetes
no estan instalados en absoluto.
