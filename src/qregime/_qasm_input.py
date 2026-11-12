"""Normalise a user's OpenQASM 2 file before analysis.

The analysis layers (pyzx parsing, the QCP tracker, exact statevectors)
only handle the unitary part of a circuit. Real QASM files almost always end
in ``measure`` (and often carry ``barrier``), which used to make
``qregime.profile`` fail with ``tipo de compuerta no soportado: Measurement``
(audit finding #6).

:func:`unitary_qasm` yields a path to analyse: the original file when it has
no measurements, resets or barriers, otherwise a temporary copy with final
measurements and all barriers removed. Mid-circuit measurements and resets
change the computation, so they are rejected with a clear error instead of
being dropped silently.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager

_NON_UNITARY = {"measure", "reset", "barrier"}


@contextmanager
def unitary_qasm(qasm_path: str):
    from qiskit import QuantumCircuit

    from .qasm_export import safe_qasm2_dumps

    qc = QuantumCircuit.from_qasm_file(qasm_path)
    if not {inst.operation.name for inst in qc.data} & _NON_UNITARY:
        yield qasm_path
        return

    stripped = qc.remove_final_measurements(inplace=False)
    remaining = {inst.operation.name for inst in stripped.data} & {"measure", "reset"}
    if remaining:
        raise ValueError(
            f"{qasm_path}: mid-circuit {' and '.join(sorted(remaining))} is not supported; "
            "qregime analyses the unitary part of a circuit. Remove them or defer the "
            "measurements to the end of the circuit.")

    unitary = QuantumCircuit(*stripped.qregs)
    for inst in stripped.data:
        if inst.operation.name != "barrier":
            unitary.append(inst.operation, inst.qubits)

    fd, tmp_path = tempfile.mkstemp(suffix=".qasm", prefix="qregime_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(safe_qasm2_dumps(unitary))
        yield tmp_path
    finally:
        os.unlink(tmp_path)
