"""The fast QASM loader must give exactly what pyzx's own parser gives.

qregime._qasm.load_pyzx skips pyzx's lark grammar for plain decimal angles
(the cause of the slowdown measured in research/comparison/PERF_STRESS.md).
These tests pin that it changes speed only: same gates, same phases (as
exact Fractions), and the same errors for input pyzx rejects.
"""
import glob
import os
import random

import pytest
import pyzx as zx

from qregime._qasm import load_pyzx

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
ROOT = os.path.join(os.path.dirname(__file__), "..")


def _gates(circuit):
    return [(type(g).__name__, repr(g), getattr(g, "phase", None)) for g in circuit.gates]


def _assert_same(path):
    expected = zx.Circuit.from_qasm_file(path)
    got = load_pyzx(path)
    assert _gates(got) == _gates(expected)
    assert (got.qubits, got.bits, got.name) == (expected.qubits, expected.bits, expected.name)


def _literals(rng):
    forms = [
        lambda: f"{rng.uniform(-9, 9):.{rng.randint(0, 14)}f}",  # long decimals
        lambda: str(rng.randint(-20, 20)),
        lambda: f".{rng.randint(0, 99999)}",
        lambda: f"-.{rng.randint(1, 999)}",
        lambda: f"{rng.randint(0, 9)}.",
        lambda: rng.choice(["pi", "-pi", "pi/7", "2*pi/3", "0.5*pi", "pi*0.25", "(pi/2)"]),
    ]
    return rng.choice(forms)()


def test_matches_pyzx_on_example_circuits():
    paths = [p for p in glob.glob(os.path.join(ROOT, "examples", "circuits", "*.qasm"))]
    assert paths
    for path in paths:
        _assert_same(path)


def test_matches_pyzx_on_random_angle_literals(tmp_path):
    rng = random.Random(7)
    lines = []
    for i in range(2000):
        lines.append(f"rz({_literals(rng)}) q[{i % 3}];")
        if i % 40 == 0:
            lines.append(f"u3({_literals(rng)},{_literals(rng)}, {_literals(rng)}) q[{i % 3}];")
            lines.append(f"cx q[{i % 3}],q[{(i + 1) % 3}];")
    path = tmp_path / "angles.qasm"
    path.write_text(HEADER + "\n".join(lines) + "\n")
    _assert_same(str(path))


@pytest.mark.parametrize("angle", ["1.5e-3", "+0.5", "theta"])
def test_rejects_what_pyzx_rejects_with_the_same_error(tmp_path, angle):
    path = tmp_path / "bad.qasm"
    path.write_text(HEADER + f"rz({angle}) q[0];\n")
    with pytest.raises(Exception) as expected:
        zx.Circuit.from_qasm_file(str(path))
    with pytest.raises(type(expected.value)) as got:
        load_pyzx(str(path))
    assert str(got.value) == str(expected.value)
