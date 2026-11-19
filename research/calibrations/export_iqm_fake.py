"""Export an IQM fake backend's error profile to qregime's format.

    pip install "iqm-client[qiskit]"
    python research/calibrations/export_iqm_fake.py IQMFakeGarnet iqm_garnet

IQM publishes per-qubit and per-coupler noise for its processors as
IQMErrorProfile objects in iqm-client (Apache-2.0). Conversions:

- IQM's noise model applies T1/T2 relaxation during each gate and then a
  depolarizing channel with parameter lambda (rho -> (1 - lambda) rho +
  lambda I/d), without subtracting one from the other. qregime's "error"
  is the total average gate infidelity, so it is computed from the
  composed process fidelity, F = (1 - lambda) F_relax + lambda / d**2,
  as error = d (1 - F) / (d + 1);
- T1, T2 and gate durations are in ns there, seconds here;
- readout_errors {"0": P(1|0), "1": P(0|1)} map to p01, p10;
- the single-qubit gate is PRX(theta, phi) plus virtual Z: one pulse for
  any single-qubit gate ("arbitrary").
"""
from __future__ import annotations

import json
import os
import sys

import importlib

from qregime.calibration import _relax_fidelity

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "src", "qregime", "data", "calibrations")


def main(fake: str = "IQMFakeGarnet", name: str = "iqm_garnet"):
    from importlib.metadata import version

    module = importlib.import_module(
        "iqm.qiskit_iqm.fake_backends.fake_" + fake.removeprefix("IQMFake").lower())
    backend = getattr(module, fake)()
    profile = backend.error_profile
    labels = sorted(profile.t1s, key=lambda q: int(q.removeprefix("QB")))
    index = {q: i for i, q in enumerate(labels)}
    (gate_1q, errors_1q), = profile.single_qubit_gate_depolarizing_error_parameters.items()
    (gate_2q, errors_2q), = profile.two_qubit_gate_depolarizing_error_parameters.items()
    t1 = {q: profile.t1s[q] * 1e-9 for q in labels}
    t2 = {q: profile.t2s[q] * 1e-9 for q in labels}
    d1 = profile.single_qubit_gate_durations[gate_1q] * 1e-9
    d2 = profile.two_qubit_gate_durations[gate_2q] * 1e-9

    def total_error(lam, dim, f_relax):
        f = (1 - lam) * f_relax + lam / dim ** 2
        return dim * (1 - f) / (dim + 1)

    pairs = {}
    for (a, b), lam in errors_2q.items():
        i, j = sorted((index[a], index[b]))
        f_relax = _relax_fidelity(d2, t1[a], t2[a]) * _relax_fidelity(d2, t1[b], t2[b])
        pairs[f"{i},{j}"] = {"error": total_error(lam, 4, f_relax)}
    errors = sorted(v["error"] for v in pairs.values())
    data = {
        "schema": "qregime-calibration/1",
        "name": name,
        "source": (f"{fake} error profile ({profile.name}) in iqm-client "
                   f"{version('iqm-client')} (Apache-2.0), exported with "
                   "research/calibrations/export_iqm_fake.py. Consistent with the medians in "
                   "Abdurakhimov et al., arXiv:2408.12433 (single-qubit error 9e-4, CZ error "
                   "5e-3, readout error 3e-2, T1 about 40 us)."),
        "num_qubits": len(labels),
        "qubit_labels": labels,
        "connectivity": sorted([sorted((index[a], index[b])) for a, b in errors_2q]),
        "single_qubit": {"error": [total_error(errors_1q[q], 2, _relax_fidelity(d1, t1[q], t2[q]))
                                   for q in labels],
                         "duration": d1,
                         "decomposition": "arbitrary"},
        "two_qubit": {"native": "cz" if gate_2q == "cz" else gate_2q,
                      "error": errors[len(errors) // 2],
                      "duration": d2,
                      "pairs": pairs},
        "readout": {"p01": [profile.readout_errors[q]["0"] for q in labels],
                    "p10": [profile.readout_errors[q]["1"] for q in labels]},
        "idle": {"t1": [t1[q] for q in labels], "t2": [t2[q] for q in labels]},
    }
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {path}: {len(labels)} qubits, {len(pairs)} couplers")


if __name__ == "__main__":
    main(*sys.argv[1:])
