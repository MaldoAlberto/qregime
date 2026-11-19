"""Export a Google device calibration bundled with cirq-google to qregime's format.

    pip install "cirq-google>=1.6"
    python research/calibrations/export_cirq_google.py \
        willow_pink_d7v1-2024_08_16 google_willow

Writes src/qregime/data/calibrations/<name>.json. The conversion follows
cirq_google.engine.noise_properties_from_calibration, Google's own path from
a calibration to a noise model:

- T1 per qubit from single_qubit_idle_t1_micros;
- Tphi per qubit from the incoherent single-qubit RB error minus the T1
  part, tphi = t_gate / (3 (e_incoherent - t_gate / (3 T1))); then
  1/T2 = 1/(2 T1) + 1/Tphi;
- single-qubit error from single_qubit_rb_pauli_error_per_gate and CZ error
  from two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle (the whole
  cycle, as cirq does), both converted from Pauli error p to average gate
  infidelity d p / (d + 1);
- the coherent fSim error per pair (cirq applies PhasedFSimGate(theta,
  phi) after each CZ) from two_qubit_parallel_cz_gate_xeb_entangler_
  theta_error_per_cycle and ..._phi_error_per_cycle; its entanglement
  infidelity is subtracted from the CZ Pauli error, as cirq does;
- readout from single_qubit_p00_error (P(1|0)) and single_qubit_p11_error
  (P(0|1));
- gate durations from the device specification bundled with cirq-google.
"""
from __future__ import annotations

import json
import os
import re
import sys

import cirq
import cirq_google
import numpy as np

from qregime.calibration import fsim_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "src", "qregime", "data", "calibrations")
PKG = os.path.dirname(cirq_google.__file__)


def _metric(cal, name):
    return {k: v[0] for k, v in cal[name].items()} if name in cal else {}


def _durations_ns(processor: str) -> dict:
    path = os.path.join(PKG, "devices", "specifications",
                        f"{processor}_device_spec_for_grid_device.proto.txt")
    text = open(path).read()
    return {gate: int(ps) / 1000 for ps, gate in
            re.findall(r"gate_duration_picos: (\d+)\s*\n\s*(\w+) \{", text)}


def main(processor: str = "willow_pink_d7v1-2024_08_16", name: str = "google_willow"):
    cal = cirq.read_json(os.path.join(PKG, "devices", "calibrations",
                                      f"{processor}_calibration.json"))
    durations = _durations_ns(processor)
    t_1q, t_cz = durations["phased_xz"], durations["cz"]
    t1_us = {q[0]: v for q, v in _metric(cal, "single_qubit_idle_t1_micros").items()}
    qubits = sorted(t1_us)
    index = {q: i for i, q in enumerate(qubits)}
    incoherent = {q[0]: v for q, v in
                  _metric(cal, "single_qubit_rb_incoherent_error_per_gate").items()}
    pauli_1q = {q[0]: v for q, v in _metric(cal, "single_qubit_rb_pauli_error_per_gate").items()}
    p00 = {q[0]: v for q, v in _metric(cal, "single_qubit_p00_error").items()}
    p11 = {q[0]: v for q, v in _metric(cal, "single_qubit_p11_error").items()}
    prefix = "two_qubit_parallel_cz_gate_xeb_"
    pauli_2q = _metric(cal, prefix + "pauli_error_per_cycle")
    phi = _metric(cal, prefix + "entangler_phi_error_per_cycle")
    theta = _metric(cal, prefix + "entangler_theta_error_per_cycle")

    t1, t2 = [], []
    for q in qubits:
        t1_ns = t1_us[q] * 1000
        tphi_err = incoherent.get(q, 0.0) - t_1q / (3 * t1_ns)
        tphi_ns = 1e10 if tphi_err <= 0 else t_1q / (3 * tphi_err)
        t1.append(t1_ns * 1e-9)
        t2.append(1e-9 / (1 / (2 * t1_ns) + 1 / tphi_ns))
    pairs = {}
    for (a, b), p in pauli_2q.items():
        i, j = sorted((index[a], index[b]))
        # the XEB Pauli error includes the coherent fSim error; cirq subtracts
        # its entanglement infidelity 1 - |Tr U|^2 / 16 and so do we
        u = fsim_matrix(theta.get((a, b), 0.0), phi.get((a, b), 0.0))
        p = max(0.0, p - (1 - abs(np.trace(u)) ** 2 / 16))
        pairs[f"{i},{j}"] = {"error": 4 * p / 5, "duration": t_cz * 1e-9,
                             "swap_angle": theta.get((a, b), 0.0),
                             "conditional_phase": phi.get((a, b), 0.0)}
    errors_2q = sorted(v["error"] for v in pairs.values())
    data = {
        "schema": "qregime-calibration/1",
        "name": name,
        "source": (f"Google calibration snapshot {processor} bundled with cirq-google "
                   f"{cirq_google.__version__} (Apache-2.0), converted as in "
                   "cirq_google.engine.noise_properties_from_calibration; exported with "
                   "research/calibrations/export_cirq_google.py. Qubit i is "
                   f"{qubits[0]!r}, ... in sorted GridQubit order. Not modelled: leakage, "
                   "crosstalk, single-qubit Z phases (zeta, gamma)."),
        "date": processor.rsplit("-", 1)[-1].replace("_", "-"),
        "num_qubits": len(qubits),
        "qubit_labels": [f"{q.row}_{q.col}" for q in qubits],
        "connectivity": sorted([sorted((index[a], index[b])) for a, b in pauli_2q]),
        "single_qubit": {"error": [2 * pauli_1q.get(q, 0.0) / 3 for q in qubits],
                         "duration": t_1q * 1e-9, "decomposition": "arbitrary"},
        "two_qubit": {"native": "cz", "error": errors_2q[len(errors_2q) // 2],
                      "duration": t_cz * 1e-9, "pairs": pairs},
        "readout": {"p01": [p00.get(q, 0.0) for q in qubits],
                    "p10": [p11.get(q, 0.0) for q in qubits]},
        "idle": {"t1": t1, "t2": t2},
    }
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {path}: {len(qubits)} qubits, {len(pairs)} couplers")


if __name__ == "__main__":
    main(*sys.argv[1:])
