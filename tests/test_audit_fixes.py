"""Regression tests for the findings of the November 2026 audit.
Each test fails on the code before the fix."""
import pytest

import qregime
from qregime import cli
from qregime.regime import decide_regime_formal
from qregime.resources import _load_fit, estimate_open_ftqc, predicted_logical_error_rate
from qregime.tool import HARDWARE_PROFILES, simulate_expectation

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


@pytest.fixture
def qasm(tmp_path):
    def _write(body, name="c.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body)
        return str(path)
    return _write


# Finding 1: the error budget is shared by every patch at every step.
def test_ftqc_reliability_counts_every_logical_qubit():
    n, t, p = 16, 3930, 0.0032
    r = estimate_open_ftqc(n, t, p)
    p_l = predicted_logical_error_rate(r["codeDistance"], p, _load_fit())
    assert (1 - p_l) ** (n * t) >= 0.99


def test_more_logical_qubits_need_at_least_the_same_distance():
    d = [estimate_open_ftqc(n, 500, 0.0032)["codeDistance"] for n in (1, 10, 100, 1000)]
    assert d == sorted(d) and d[0] < d[-1]


# Finding 2: FTQC must meet the reliability target, not only fit the budget.
def test_ftqc_meets_a_target_stricter_than_its_default_budget(circuit):
    r = qregime.decide_regime(circuit("adder8"), hardware="trapped_ion_ionq_oxford",
                              reliability_target=0.999, physical_qubit_budget=10**7)
    assert r["ftqc"]["reliability"] >= 0.999
    assert r["ftqc"]["meets_target"] is True


def test_full_reliability_target_is_reported_not_estimated(circuit):
    r = qregime.decide_regime(circuit("adder8"), reliability_target=1.0,
                              qem_success_fraction=0.0, physical_qubit_budget=10**7)
    assert r["ftqc"]["cost_qubits"] is None
    assert r["ftqc"]["meets_target"] is False


# Finding 3: partial QEC competes in the argmin even when FTQC fits.
def test_partial_qec_cheaper_than_fitting_ftqc_is_chosen(circuit):
    r = qregime.decide_regime(circuit("adder8"), physical_qubit_budget=10**7,
                              reliability_target=0.9, qem_success_fraction=0.0)
    assert r["ftqc"]["fits_budget"] is True
    assert r["regime"].startswith("PARTIAL_QEC")
    assert r["cost_physical_qubits"] < r["ftqc"]["cost_qubits"]
    assert r["gap"] is None


# Finding 4: arbitrary rotations are charged their synthesis cost.
def test_profile_splits_exact_t_from_rotations(qasm):
    path = qasm("qreg q[1];\nt q[0];\ntdg q[0];\nrz(0.123) q[0];\nrz(1.1) q[0];\n")
    p = qregime.profile(path)
    assert (p["tcount_exact"], p["rotation_count"]) == (2, 2)


def test_rotation_synthesis_costs_many_t_gates():
    from qregime.resources import rotation_synthesis_tcount
    assert rotation_synthesis_tcount(0, 5e-3) == 0
    assert rotation_synthesis_tcount(1, 5e-3) >= 20
    assert rotation_synthesis_tcount(100, 5e-3) > 100 * rotation_synthesis_tcount(1, 5e-3) / 2


def test_decide_regime_charges_rotations(circuit):
    r = qregime.decide_regime(circuit("vqe_ansatz"))
    assert r["profile"]["rotation_count"] == 48
    assert r["ftqc"]["tcount_effective"] > 20 * r["profile"]["tcount"]


# Finding 5: a Clifford-only circuit (T-count 0) no longer crashes.
def test_clifford_circuit_is_decided(qasm):
    path = qasm("qreg q[2];\nh q[0];\ncx q[0],q[1];\n")
    r = qregime.decide_regime(path)
    assert r["regime"] == "QEM"
    assert r["ftqc"]["cost_qubits"] > 0


def test_clifford_circuit_cli(qasm, capsys):
    path = qasm("qreg q[2];\nh q[0];\ncx q[0],q[1];\n")
    assert cli.main(["decide", path]) == 0


def test_analyze_clifford_circuit(qasm):
    path = qasm("qreg q[2];\nh q[0];\ncx q[0],q[1];\n")
    assert qregime.analyze(path)["resource_estimate_declared_qubits"]["feasible"]


# Finding 6: final measurements and barriers are accepted.
def test_measured_circuit_is_profiled(qasm):
    path = qasm("qreg q[2];\ncreg c[2];\nh q[0];\nt q[0];\ncx q[0],q[1];\n"
                "barrier q;\nmeasure q -> c;\n")
    p = qregime.profile(path)
    assert p["num_qubits"] == 2 and p["tcount"] == 1
    assert qregime.decide_regime(path)["qasm_path"] == path


def test_mid_circuit_measurement_is_rejected_clearly(qasm):
    path = qasm("qreg q[2];\ncreg c[2];\nh q[0];\nmeasure q[0] -> c[0];\ncx q[0],q[1];\n")
    with pytest.raises(ValueError, match="mid-circuit measure"):
        qregime.profile(path)


# Finding 7: an infeasible FTQC estimate is reported, not a KeyError.
def test_ftqc_above_threshold_is_reported(circuit):
    path = circuit("vqe_ansatz")
    prof = qregime.profile(path)
    r = decide_regime_formal(path, tcount=prof["tcount_exact"], num_qubits=prof["num_qubits"],
                             p_phys=0.05, qem_success_fraction=0.0, verbose=True, profile=prof,
                             rotation_count=prof["rotation_count"])
    assert r["cost_ftqc_qubits"] is None
    assert r["ftqc_estimable"] is False
    assert r["gap_report"]["gap_de_qubits_FTQC"] is None


# Finding 8: every two-qubit gate carries noise in the simulation.
def test_toffoli_and_swap_are_noisy(qasm):
    path = qasm("qreg q[3];\n" + "ccx q[0],q[1],q[2];\nswap q[0],q[2];\ncy q[1],q[2];\n" * 10)
    noisy = HARDWARE_PROFILES["superconductor_ibm"]
    r = simulate_expectation(path, noisy, shots=2000, verbose=False)
    assert r["success_fraction_measured"] < 0.95


def test_cli_rejects_target_outside_unit_interval(circuit):
    with pytest.raises(SystemExit):
        cli.main(["decide", circuit("vqe_ansatz"), "--target", "1.5"])
