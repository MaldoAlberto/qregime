"""Build the IQM Emerald calibration from published medians.

    pip install "iqm-client[qiskit]"
    python research/calibrations/build_iqm_emerald.py

Emerald (54 qubits) has no per-qubit data in iqm-client. The coupling map
is IQM's 54-qubit Crystal architecture from iqm-client's IQMFakeAphrodite
(only its topology is used: its noise is a sample model, not Emerald data).
Every qubit and coupler gets the medians of Joch et al., arXiv:2603.28486,
Table 1 (March 2026): T1 50.05 us, Ramsey T2 15.75 us, PRX fidelity
99.94%, CZ fidelity 99.45%. Not in that paper, hence assumptions:
readout error 3e-2 (IQM's median for Garnet, arXiv:2408.12433) and gate
durations 20 ns (PRX) and 40 ns (CZ, upper end of 20-40 ns in the same
paper). Ramsey T2 is shorter than an echo T2, so idle dephasing is
pessimistic.
"""
from __future__ import annotations

import importlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "src", "qregime", "data", "calibrations")


def main(name: str = "iqm_emerald"):
    module = importlib.import_module("iqm.qiskit_iqm.fake_backends.fake_aphrodite")
    backend = module.IQMFakeAphrodite()
    edges = sorted({tuple(sorted(k)) for k in backend.target["cz"]})
    data = {
        "schema": "qregime-calibration/1",
        "name": name,
        "source": ("IQM Emerald medians from Joch et al., 'Emergent-Coupling-Based Ansatz "
                   "Evaluated on a Superconducting Quantum Processor', arXiv:2603.28486, "
                   "Table 1: T1 50.05 us, T2 (Ramsey) 15.75 us, PRX fidelity 99.94%, CZ "
                   "fidelity 99.45%, 54 qubits. Coupling map of IQM's 54-qubit Crystal "
                   "architecture from iqm-client's IQMFakeAphrodite. Assumed (not in the "
                   "paper): readout error 3e-2 (IQM Garnet median, arXiv:2408.12433), PRX "
                   "20 ns, CZ 40 ns. Uniform: no per-qubit spread. Built with "
                   "research/calibrations/build_iqm_emerald.py."),
        "date": "2026-03-30",
        "num_qubits": backend.num_qubits,
        "connectivity": [list(e) for e in edges],
        "single_qubit": {"error": 6e-4, "duration": 20e-9, "decomposition": "arbitrary"},
        "two_qubit": {"native": "cz", "error": 5.5e-3, "duration": 40e-9},
        "readout": {"p01": 3e-2, "p10": 3e-2},
        "idle": {"t1": 50.05e-6, "t2": 15.75e-6},
    }
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {path}: {data['num_qubits']} qubits, {len(edges)} couplers")


if __name__ == "__main__":
    main()
