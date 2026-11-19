# Device calibrations

qregime decides between error mitigation, partial error correction and full
fault tolerance. For small circuits, the mitigation side of that decision is
only as good as the noise model behind it. A single depolarizing rate per
two-qubit gate is optimistic. On IBM Torino it predicts 0.85 success for a
60-CX mirror circuit, while the full calibration gives 0.66 (see
[qregime_and_mitiq.md](qregime_and_mitiq.md)).

qregime therefore reads device calibrations in a small, vendor-neutral JSON
format, and simulates circuits on them exactly. No quantum SDK is needed:
qregime reads OpenQASM 2 itself (`qregime.circuit`) and simulates with numpy
(`qregime.simulate`, `qregime.calibration`).

## Bundled calibrations

| Name | Device | Source | Used by profile |
|---|---|---|---|
| `ibm_torino` | IBM Torino, 133 qubits, heavy-hex, CZ | `qiskit-ibm-runtime` `FakeTorino` snapshot, exported with [`research/calibrations/export_qiskit_backend.py`](../research/calibrations/export_qiskit_backend.py) | `superconductor_ibm` |
| `quantinuum_h2_2` | Quantinuum H2-2, 56 qubits, all-to-all, ZZ | Quantinuum [hardware-specifications](https://github.com/Quantinuum/quantinuum-hardware-specifications) repository, emulator parameters, H2-2 2024-12-06 | `trapped_ion_quantinuum_h2` |

`trapped_ion_ionq_oxford` has no calibration: the published result
(arXiv:2510.17286) gives the two-qubit error only. That profile keeps the
depolarizing model.

## Format (`qregime-calibration/1`)

```json
{
  "schema": "qregime-calibration/1",
  "name": "my_device",
  "source": "where every number comes from",
  "date": "2026-01-01",
  "num_qubits": 56,
  "connectivity": "all_to_all",
  "single_qubit": {"error": 8e-5, "duration": 0.0, "decomposition": "arbitrary"},
  "two_qubit": {"native": "rzz", "error": 1.4e-3, "duration": 1.0,
                "pairs": {"0,1": {"error": 2e-3, "duration": 1.0}},
                "zz_overrotation": 0.0},
  "readout": {"p01": 9e-4, "p10": 1.8e-3},
  "idle": {"memory_error_per_layer": 5.3e-4}
}
```

| Field | Meaning |
|---|---|
| `connectivity` | `"all_to_all"`, or a list of coupled pairs `[[0, 1], ...]` |
| `single_qubit.error`, `.duration` | per physical pulse; one number or a per-qubit list |
| `single_qubit.decomposition` | `"arbitrary"`: one pulse per rotation (trapped ions). `"zsx"`: IBM style, 0 to 2 `sx` pulses per rotation. Z rotations are virtual in both cases |
| `two_qubit.native` | `"cz"`, `"cx"` or `"rzz"`. With `"rzz"`, an `rzz` gate in the circuit costs one native gate |
| `two_qubit.error`, `.duration` | default for every coupler; `pairs` overrides per coupler |
| `two_qubit.zz_overrotation` | optional coherent error: an extra `rzz(angle)` after every two-qubit gate |
| `readout.p01`, `.p10` | P(read 1 given 0), P(read 0 given 1); one number or a per-qubit list |
| `idle` | either `t1`, `t2` in seconds (per-qubit lists allowed, with durations in seconds), or `memory_error_per_layer`: dephasing per qubit per two-qubit layer, with durations in layers |

Missing sections mean no error of that kind. Errors are average gate
infidelities. As in Qiskit Aer, the part of a gate's error that relaxation
during the gate already explains is not counted twice.

## What the simulator does

1. **Layout.** It embeds the circuit's interaction graph into the coupling
   graph, one connected component at a time, scoring candidates by
   readout, gate and decoherence error. If no embedding exists, it picks a
   connected region and routes with SWAPs.
2. **Compilation.** It translates to the native entangler and fuses runs of
   single-qubit gates into one rotation, as a hardware compiler would.
3. **Scheduling.** Gates run as soon as their qubits are free. Waiting
   qubits decohere (T1/T2 or memory error).
4. **Noise.** Each gate is followed by depolarizing noise and relaxation
   during the gate. Readout flips are applied to the final distribution.
5. **Metric.** The result is the probability of landing in the circuit's
   ideal output support, which is qregime's success fraction. The
   simulation is exact (density matrix) up to 10 qubits.

**Validation.** Against Qiskit Aer with the FakeTorino noise model, using
the same layout and the same single-qubit fusion, success fractions agree
within 0.003 on mirror circuits from 6 to 160 CX
(`tests/test_calibration.py`).

**Not modelled.** Leakage, crosstalk, drift and non-Markovian noise.
Coherent errors are modelled only through `zz_overrotation`.

## Use

```python
import qregime
from qregime import calibration

qregime.decide_regime("circuit.qasm")                         # profile's own calibration
qregime.decide_regime("circuit.qasm", calibration="quantinuum_h2_2")
qregime.decide_regime("circuit.qasm", calibration="my_device.json")
calibration.success_fraction("circuit.qasm", "ibm_torino")    # just the simulation
```

```bash
qregime decide circuit.qasm --calibration my_device.json --zne
```

To export a live IBM backend (needs `pip install "qregime[ibm]"`):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qregime.calibration import from_qiskit_backend
import json

backend = QiskitRuntimeService().backend("ibm_torino")
json.dump(from_qiskit_backend(backend), open("ibm_torino_today.json", "w"))
```
