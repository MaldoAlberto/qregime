"""Vendor-neutral calibrations and the calibrated simulator."""
import math

import numpy as np
import pytest

from qregime import calibration as K
from qregime import circuit as C

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _mirror(n=4, layers=4, seed=1):
    rng = np.random.default_rng(seed)
    ops = []
    for layer in range(layers):
        ops += [f"ry({rng.uniform(0.1, 3.0)!r}) q[{q}];" for q in range(n)]
        ops += [f"cx q[{q}],q[{q + 1}];" for q in range(layer % 2, n - 1, 2)]
    inverse = []
    for op in reversed(ops):
        if op.startswith("ry("):
            angle, rest = op[3:].split(")", 1)
            inverse.append(f"ry({-float(angle)!r}){rest}")
        else:
            inverse.append(op)
    return C.parse(HEADER + f"qreg q[{n}];\n" + "\n".join(ops + inverse) + "\n")


def _noiseless(**overrides):
    data = {"schema": K.SCHEMA, "name": "ideal", "num_qubits": 8,
            "connectivity": [[i, i + 1] for i in range(7)],
            "two_qubit": {"native": "cz"}, "single_qubit": {"decomposition": "zsx"}}
    data.update(overrides)
    return K.load(data)


@pytest.mark.parametrize("native", ["cx", "cz", "rzz"])
def test_native_rewrites_equal_cx(native):
    m = np.eye(4, dtype=complex)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    for op in K._native_cx(0, 1, native):
        if op[0] == "1q":
            g = np.kron(np.eye(2), op[2]) if op[1] == (0,) else np.kron(op[2], np.eye(2))
        else:
            g = op[2] if op[1] == (0, 1) else swap @ op[2] @ swap
        m = g @ m
    assert np.allclose(m / m[0, 0], K._CX)


def test_routing_on_a_line_preserves_the_output():
    text = HEADER + "qreg q[6];\nx q[0];\nx q[3];\n" + "".join(
        f"cx q[{a}],q[{b}];\n" for a in range(6) for b in range(6) if a != b and (a + b) % 3 == 0)
    circ = C.parse(text + "ccx q[0],q[5],q[2];\n")
    r = K.success_fraction(circ, _noiseless())
    assert r["success_fraction"] == pytest.approx(1.0)
    assert r["two_qubit_gates"] > 20                      # SWAPs were inserted
    all_to_all = _noiseless(connectivity="all_to_all", two_qubit={"native": "rzz"})
    assert K.success_fraction(circ, all_to_all)["success_fraction"] == pytest.approx(1.0)


def test_noise_lowers_success_and_readout_mitigation_helps():
    circ = _mirror()
    for name in K.available():
        r = K.success_fraction(circ, name)
        assert 0 < r["success_fraction"] < r["success_fraction_readout_mitigated"] < 1, name


def test_layout_avoids_bad_qubits():
    bad = [0.5] * 8
    bad[6] = bad[7] = 0.0
    cal = _noiseless(readout={"p01": bad, "p10": bad})
    circ = C.parse(HEADER + "qreg q[2];\nx q[0];\ncx q[0],q[1];\n")
    r = K.success_fraction(circ, cal)
    assert sorted(r["physical_qubits"]) == [6, 7]
    assert r["success_fraction"] == pytest.approx(1.0)


def test_memory_error_per_layer_matches_dephasing():
    m = 0.01
    cal = K.load({"schema": K.SCHEMA, "name": "mem", "num_qubits": 1,
                  "idle": {"memory_error_per_layer": m},
                  "two_qubit": {"duration": 1.0}})
    assert math.exp(-1 / cal.t2[0]) == pytest.approx(1 - 2 * m)


def test_schema_is_checked():
    with pytest.raises(ValueError, match="schema"):
        K.load({"name": "x", "num_qubits": 1})
    with pytest.raises(ValueError, match="expected 2 values"):
        K.load({"schema": K.SCHEMA, "num_qubits": 2, "readout": {"p01": [0.1]}})
    with pytest.raises(ValueError, match="unknown calibration"):
        K.load("no_such_device")


def test_bundled_calibrations_load():
    assert {"ibm_torino", "quantinuum_h2_2"} <= set(K.available())
    torino = K.load("ibm_torino")
    assert torino.num_qubits == 133 and torino.native == "cz" and not torino.all_to_all
    h2 = K.load("quantinuum_h2_2")
    assert h2.all_to_all and h2.native == "rzz"


def test_agrees_with_qiskit_aer_on_fake_torino(tmp_path):
    """Same layout, Aer density matrix with the FakeTorino noise model and
    single-qubit fusion: success fractions agree within 0.005."""
    pytest.importorskip("qiskit_ibm_runtime")
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import PassManager, generate_preset_pass_manager
    from qiskit.transpiler.passes import (ALAPScheduleAnalysis, Optimize1qGatesDecomposition,
                                          PadDelay)
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    circ = _mirror(n=4, layers=6)
    qc = QuantumCircuit.from_qasm_str(C.to_qasm(circ))
    backend = FakeTorino()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
    pm.init = generate_preset_pass_manager(optimization_level=0, backend=backend).init
    pm.optimization = PassManager([Optimize1qGatesDecomposition(target=backend.target)])
    tqc = pm.run(qc)
    phys = tqc.layout.final_index_layout()
    timed = PassManager([ALAPScheduleAnalysis(target=backend.target),
                         PadDelay(target=backend.target)]).run(tqc)
    timed.save_probabilities(qubits=phys)
    probs = np.asarray(AerSimulator(method="density_matrix",
                                    noise_model=NoiseModel.from_backend(backend))
                       .run(timed).result().data()["probabilities"])
    support = np.abs(Statevector(qc).data) ** 2 > 1e-9
    ours = K.success_fraction(circ, "ibm_torino", layout=list(phys))
    assert ours["success_fraction_readout_mitigated"] == pytest.approx(probs[support].sum(), abs=5e-3)
