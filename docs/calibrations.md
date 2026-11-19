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

| Name | Device | Date | Source | Used by profile |
|---|---|---|---|---|
| `ibm_boston` | IBM Heron, 156 qubits, heavy-hex, CZ | 2026-04-17 | `qiskit-ibm-runtime` 0.52.1 `FakeBoston` snapshot, exported with [`export_qiskit_backend.py`](../research/calibrations/export_qiskit_backend.py) | `superconductor_ibm_heron` |
| `ibm_miami` | IBM Nighthawk, 120 qubits, square lattice, CZ | 2026-04-17 | `qiskit-ibm-runtime` 0.52.1 `FakeMiami` snapshot, same script | `superconductor_ibm_nighthawk` |
| `ibm_torino` | IBM Heron, 133 qubits, heavy-hex, CZ | 2025-02-26 | `FakeTorino` snapshot, same script | `superconductor_ibm` (kept for the paper's results) |
| `google_willow` | Google Willow, 105 qubits, square grid, CZ | 2024-08-16 | Google's calibration snapshot `willow_pink_d7v1-2024_08_16` bundled with `cirq-google`, converted as in `cirq_google.engine.noise_properties_from_calibration` by [`export_cirq_google.py`](../research/calibrations/export_cirq_google.py); includes the coherent fSim error per coupler. Google has published no newer per-qubit data (cirq-google 1.7 ships the same snapshot) | `superconductor_google_willow` |
| `iqm_emerald` | IQM Emerald, 54 qubits, square lattice, CZ | 2026-03-30 | Medians from arXiv:2603.28486, Table 1, on the 54-qubit Crystal coupling map, built by [`build_iqm_emerald.py`](../research/calibrations/build_iqm_emerald.py). Uniform; readout error (3e-2) and gate durations assumed | `superconductor_iqm_emerald` |
| `iqm_garnet` | IQM Garnet, 20 qubits, square lattice, CZ | 2024 | `IQMFakeGarnet` error profile in `iqm-client`, exported with [`export_iqm_fake.py`](../research/calibrations/export_iqm_fake.py) | `superconductor_iqm_garnet` |
| `quantinuum_helios` | Quantinuum Helios, 98 qubits, all-to-all, ZZ | 2025-11-07 | Ransford et al., arXiv:2511.05465: zone-averaged errors, SPAM split, linear memory error | `trapped_ion_quantinuum_helios` |
| `quantinuum_h2_2` | Quantinuum H2-2, 56 qubits, all-to-all, ZZ | 2024-12-06 | Quantinuum [hardware-specifications](https://github.com/Quantinuum/quantinuum-hardware-specifications) repository, emulator parameters (the repository has no Helios data) | `trapped_ion_quantinuum_h2` |
| `ionq_forte` | IonQ Forte, 30 qubits, all-to-all, ZZ | 2023-08-09 | Chen et al., arXiv:2308.05071 (median DRB errors, SPAM, gate times) and IonQ's Forte page (T1, T2) | `trapped_ion_ionq_forte` |

IonQ Tempo (100 qubits, #AQ 64, October 2025) is not included: IonQ
publishes only targets for it (99.9% "target fidelity"), no measured gate,
readout or coherence numbers.

`trapped_ion_ionq_oxford` has no calibration: the published result
(arXiv:2510.17286) gives the two-qubit error only. That profile keeps the
depolarizing model.

Medians of the bundled files (average gate infidelity; readout is the mean
of p01 and p10):

| Calibration | 1q error | 2q error | Readout error |
|---|---|---|---|
| `ibm_torino` | 2.7e-4 | 4.2e-3 | 2.3e-2 |
| `ibm_boston` | 1.4e-4 | 1.3e-3 | 5.1e-3 |
| `ibm_miami` | 2.0e-4 | 3.0e-3 | 1.8e-2 |
| `google_willow` | 2.7e-4 | 2.0e-3 + coherent fSim error | 7.3e-3 |
| `iqm_garnet` | 1.3e-3 | 8.0e-3 | 2.5e-2 |
| `iqm_emerald` | 6.0e-4 | 5.5e-3 | 3.0e-2 (assumed) |
| `quantinuum_h2_2` | 8.0e-5 | 1.4e-3 | 1.4e-3 |
| `quantinuum_helios` | 2.5e-5 | 7.9e-4 | 4.8e-4 |
| `ionq_forte` | 2.0e-4 | 4.6e-3 | 5.0e-3 |

The `iqm_garnet` errors include relaxation during the gate (T2 is about
10 us), as IQM's own fake-backend noise model does. They are higher than
the randomized-benchmarking medians in IQM's paper (arXiv:2408.12433:
single-qubit 9e-4, CZ 5e-3, readout 3e-2).

Success fraction of the same 6-qubit mirror circuits
([`zne_calibrated.py`](../examples/zne_calibrated.py) construction, seed 7),
unmitigated, readout error included:

| CX | IBM Torino | IBM Boston | IBM Miami | Google Willow | IQM Garnet | IQM Emerald | Quantinuum H2-2 | Quantinuum Helios | IonQ Forte |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.92 | 0.98 | 0.93 | 0.95 | 0.82 | 0.80 | 0.99 | 0.99 | 0.94 |
| 20 | 0.82 | 0.95 | 0.88 | 0.93 | 0.73 | 0.73 | 0.97 | 0.98 | 0.87 |
| 40 | 0.72 | 0.91 | 0.83 | 0.84 | 0.54 | 0.60 | 0.92 | 0.95 | 0.75 |
| 80 | 0.58 | 0.84 | 0.75 | 0.72 | 0.33 | 0.42 | 0.85 | 0.91 | 0.57 |
| 160 | 0.39 | 0.73 | 0.61 | 0.54 | 0.14 | 0.22 | 0.73 | 0.84 | 0.34 |
| 320 | 0.17 | 0.53 | 0.40 | 0.35 | 0.03 | 0.06 | 0.51 | 0.68 | 0.12 |

These are snapshots from different dates and sources, not a ranking of
today's devices.

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
| `qubit_labels` | optional: the vendor's name for each qubit index (e.g. `"4_5"`, `"QB3"`) |
| `single_qubit.error`, `.duration` | per physical pulse; one number or a per-qubit list |
| `single_qubit.decomposition` | `"arbitrary"`: one pulse per rotation (trapped ions). `"zsx"`: IBM style, 0 to 2 `sx` pulses per rotation. Z rotations are virtual in both cases |
| `two_qubit.native` | `"cz"`, `"cx"` or `"rzz"`. With `"rzz"`, an `rzz` gate in the circuit costs one native gate |
| `two_qubit.error`, `.duration` | default for every coupler; `pairs` overrides per coupler |
| `two_qubit.zz_overrotation` | optional coherent error: an extra `rzz(angle)` after every two-qubit gate |
| `pairs[...].swap_angle`, `.conditional_phase` | optional coherent error per coupler: an fSim(θ, φ) after each native gate, as in Cirq's Google noise model. When set, that pair's `error` is the incoherent part only |
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

**Validation** against each vendor's own noise model, on the same compiled
circuit (`tests/test_calibration.py`):

- `ibm_torino`, `ibm_boston`, `ibm_miami` vs Qiskit Aer with each fake
  backend's noise model: within 0.005 (`ibm_torino`: 0.003 on mirror
  circuits from 6 to 160 CX);
- `google_willow` vs Cirq's `NoiseModelFromGoogleNoiseProperties` built
  from the same Google calibration: within 0.0013 up to 32 CZ;
- `iqm_garnet` vs Aer with IQM's `IQMFakeGarnet` noise model and the same
  idle times: within 0.0011 up to 40 CZ.

`quantinuum_helios`, `quantinuum_h2_2`, `iqm_emerald` and `ionq_forte`
come from published numbers with no vendor simulator to compare against.

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

Export scripts for other sources are in
[`research/calibrations/`](../research/calibrations/): Google devices
through `cirq-google` (`pip install "qregime[google]"`) and IQM devices
through `iqm-client` (`pip install "qregime[iqm]"`). To export a live IBM
backend (needs `pip install "qregime[ibm]"`):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qregime.calibration import from_qiskit_backend
import json

backend = QiskitRuntimeService().backend("ibm_torino")
json.dump(from_qiskit_backend(backend), open("ibm_torino_today.json", "w"))
```
