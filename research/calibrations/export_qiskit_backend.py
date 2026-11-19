"""Export a Qiskit backend's calibration to qregime's vendor-neutral format.

    pip install qiskit qiskit-ibm-runtime
    python research/calibrations/export_qiskit_backend.py FakeTorino ibm_torino

Writes src/qregime/data/calibrations/<name>.json. Works with any BackendV2,
including a live IBM backend obtained from QiskitRuntimeService.
"""
import json
import os
import sys

from qiskit_ibm_runtime import fake_provider

from qregime.calibration import from_qiskit_backend

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(fake_name: str = "FakeTorino", name: str = "ibm_torino"):
    backend = getattr(fake_provider, fake_name)()
    data = from_qiskit_backend(
        backend, name=name,
        source=f"qiskit-ibm-runtime fake_provider.{fake_name} snapshot of IBM {backend.name} "
               "(per-qubit sx/readout errors, per-pair CZ errors, T1/T2, gate durations)")
    path = os.path.join(ROOT, "src", "qregime", "data", "calibrations", f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {path}: {data['num_qubits']} qubits, "
          f"{len(data['connectivity'])} couplers, native {data['two_qubit']['native']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
