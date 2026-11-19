import pytest
"""Regression tests for the bugs found by the large-scale stress test
(paper, section 'Bugs found and fixed'). Each test fails on the old code."""
import time

import pyzx as zx

from qregime import advisor, regime
from qregime.qasm_export import safe_qasm2_dumps, to_plain_decimal_qasm
from qregime.qcp_tracker import UnionGroup, run_tracker


def test_bug1_recommend_reuses_profile(circuit, monkeypatch):
    path = circuit("qft10")
    prof = advisor.profile_circuit(path)

    def fail(*_a, **_k):
        raise AssertionError("profile_circuit was recomputed")

    monkeypatch.setattr(advisor, "profile_circuit", fail)
    advisor.recommend_techniques(path, profile=prof)


def test_bug1_decide_regime_reuses_profile(circuit, monkeypatch):
    path = circuit("vqe_ansatz")
    prof = advisor.profile_circuit(path)

    def fail(*_a, **_k):
        raise AssertionError("profile_circuit was recomputed")

    monkeypatch.setattr(regime, "profile_circuit", fail)
    regime.decide_regime_formal(path, tcount=prof["tcount"], num_qubits=prof["num_qubits"],
                                p_phys=0.0032, qem_success_fraction=0.9,
                                verbose=False, profile=prof)


def test_bug2_unknown_group_allocates_nothing():
    # The old constructor allocated 2**40 complex numbers (16 TiB) here.
    g = UnionGroup(range(40), known=False)
    assert g.state is None and g.known is False


def test_bug2_tracker_scales_to_22_qubits(tmp_path):
    pytest.importorskip("qiskit")
    from qiskit import transpile
    from qiskit.circuit.library import CDKMRippleCarryAdder
    qc = transpile(CDKMRippleCarryAdder(10).decompose(reps=3),
                   basis_gates=["cz", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    path = tmp_path / "adder_22q.qasm"
    path.write_text(safe_qasm2_dumps(qc))
    t0 = time.time()
    r = run_tracker(str(path), n_pcm=4)
    assert r["num_qubits"] == 22
    assert time.time() - t0 < 10  # used to hang for more than 60 s


def test_bug3_scientific_notation_rewritten():
    text = "rz(9.587379924269612e-05) q[0];"
    fixed = to_plain_decimal_qasm(text)
    assert "e-05" not in fixed
    assert abs(float(fixed[3:fixed.index(")")]) - 9.587379924269612e-05) < 1e-18


def test_bug3_large_qft_parses_in_pyzx():
    pytest.importorskip("qiskit")
    from qiskit import transpile
    from qiskit.circuit.library import QFT
    qc = transpile(QFT(16).decompose(reps=3), basis_gates=["cz", "rz", "sx", "x"],
                   optimization_level=1, seed_transpiler=0)
    c = zx.Circuit.from_qasm(safe_qasm2_dumps(qc))
    assert c.qubits == 16
