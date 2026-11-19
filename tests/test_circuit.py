"""qregime's own OpenQASM 2 reader and simulators (no Qiskit needed)."""
import math

import numpy as np
import pytest

from qregime import circuit as C
from qregime import simulate as S

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def test_registers_broadcast_and_expressions():
    c = C.parse(HEADER + "qreg a[2];\nqreg b[1];\nh a;\nrz(-pi/4 + 2*sin(pi/2)) b[0];\n"
                "cx a[1], b[0];\n")
    assert c.num_qubits == 3
    assert [(i.name, i.qubits) for i in c.instructions] == [("h", (0,)), ("h", (1,)),
                                                           ("rz", (2,)), ("cx", (1, 2))]
    assert c.instructions[2].params[0] == pytest.approx(-math.pi / 4 + 2)


def test_custom_gates_and_builtins_expand():
    c = C.parse(HEADER + "gate bell a, b { h a; CX a, b; }\nqreg q[2];\nbell q[0], q[1];\n")
    psi = S.statevector(c)
    assert np.allclose(np.abs(psi) ** 2, [0.5, 0, 0, 0.5])


def test_gate_library_matches_known_states():
    # X on qubit 1 only -> basis index 2 (qubit 0 is the least significant bit)
    c = C.parse(HEADER + "qreg q[2];\nx q[1];\n")
    assert np.argmax(np.abs(S.statevector(c))) == 2
    # Toffoli flips the target only when both controls are 1
    c = C.parse(HEADER + "qreg q[3];\nx q[0];\nx q[1];\nccx q[0],q[1],q[2];\n")
    assert np.argmax(np.abs(S.statevector(c))) == 7


@pytest.mark.parametrize("bad, message", [
    ("qreg q[1];\nfoo q[0];\n", "not defined"),
    ("qreg q[2];\ncx q[0], q[0];\n", "same qubit twice"),
    ("qreg q[1];\nrx q[0];\n", "takes 1 parameters"),
    ("qreg q[1];\nh r[0];\n", "unknown quantum register"),
])
def test_errors_are_explicit(bad, message):
    with pytest.raises(C.QASMError, match=message):
        C.parse(HEADER + bad)


def test_strip_final_measurements():
    text = HEADER + "qreg q[2];\ncreg c[2];\nh q[0];\nbarrier q;\ncx q[0],q[1];\nmeasure q -> c;\n"
    stripped = C.strip_final_measurements(C.parse(text))
    assert "measure" not in stripped and "barrier" not in stripped and "creg" not in stripped
    assert [i.name for i in C.parse(stripped).instructions] == ["h", "cx"]
    assert C.strip_final_measurements(C.parse(HEADER + "qreg q[1];\nh q[0];\n")) is None


@pytest.mark.parametrize("body", [
    "measure q[0] -> c[0];\nh q[0];\n",
    "reset q[0];\nh q[0];\n",
    "if(c==1) x q[0];\n",
])
def test_dynamic_circuits_are_rejected(body):
    with pytest.raises(C.QASMError, match="mid-circuit"):
        C.strip_final_measurements(C.parse(HEADER + "qreg q[1];\ncreg c[1];\n" + body))


def test_density_matrix_depolarizing_and_relaxation():
    dm = S.DensityMatrix(1)
    dm.apply_1q(S.u_matrix(math.pi, 0, math.pi), 0)       # |1>
    dm.relax(0, 1.0, t1=1.0, t2=2.0)
    assert dm.probabilities()[1] == pytest.approx(math.exp(-1))
    dm = S.DensityMatrix(2)
    dm.depolarize((0, 1), 1.0)
    assert np.allclose(dm.probabilities(), 0.25)


def test_readout_error_hits_the_right_qubit():
    probs = np.zeros(4)
    probs[2] = 1.0                                         # qubit 1 = 1, qubit 0 = 0
    out = S.apply_readout_error(probs, p01=[0.1, 0.0], p10=[0.0, 0.0])
    assert np.allclose(out, [0, 0, 0.9, 0.1])


def test_depolarizing_success_closed_form():
    # two CX on |00>: rho -> (1-p) rho + p I/4 after each, and CX maps I/4 to
    # itself, so P(00) = (1-p)^2 + ((1-p) p + p) / 4
    c = C.parse(HEADER + "qreg q[2];\ncx q[0],q[1];\ncx q[0],q[1];\n")
    support = np.abs(S.statevector(c)) ** 2 > 1e-9          # only |00>
    p = 0.1
    expected = (1 - p) ** 2 + ((1 - p) * p + p) / 4
    assert S.depolarizing_success(c, support, p) == pytest.approx(expected)
