"""Edge cases that used to crash or give a wrong regime (review 2026-11)."""
import pytest

import qregime
from qregime.regime import decide_regime_formal
from qregime.resources import estimate_open_ftqc_with_factories, uncertainty_band
from qregime.tool import HardwareProfile

GHZ = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
barrier q;
measure q -> c;
"""


@pytest.fixture
def ghz(tmp_path):
    path = tmp_path / "ghz.qasm"
    path.write_text(GHZ)
    return str(path)


def test_final_measurements_are_accepted(ghz):
    r = qregime.decide_regime(ghz)
    assert r["qasm_path"] == ghz
    assert r["profile"]["num_qubits"] == 3


def test_mid_circuit_measurement_is_rejected_explicitly(tmp_path):
    path = tmp_path / "dyn.qasm"
    path.write_text(GHZ.replace("barrier q;\nmeasure q -> c;",
                                "measure q[0] -> c[0];\nh q[0];"))
    with pytest.raises(ValueError, match="mid-circuit"):
        qregime.profile(str(path))


def test_clifford_only_circuit_does_not_crash(ghz):
    r = qregime.decide_regime(ghz)
    assert r["profile"]["tcount"] == 0
    assert r["regime"] in {"QEM", "FTQC"} or r["regime"].startswith("PARTIAL_QEC")
    est = estimate_open_ftqc_with_factories(3, tcount=0, p_phys=0.001)
    assert est["feasible"] and est["physicalQubitsForFactories"] == 0


def test_above_threshold_hardware_reports_gap_not_keyerror(circuit, monkeypatch):
    monkeypatch.setitem(qregime.HARDWARE_PROFILES, "noisy",
                        HardwareProfile("noisy", 0.01, "all_to_all"))
    r = qregime.decide_regime(circuit("shor_like_sequential"), hardware="noisy",
                              qem_success_fraction=0.0)
    assert r["regime"] == "GAP"
    assert r["ftqc"]["cost_qubits"] is None
    assert r["ftqc"]["infeasible_reason"]


def test_ftqc_not_chosen_when_target_exceeds_design_reliability(circuit):
    r = decide_regime_formal(circuit("shor_like_sequential"), tcount=3930, num_qubits=16,
                             p_phys=1e-4, qem_success_fraction=0.0, error_budget=1e-2,
                             reliability_target=0.999, physical_qubit_budget=10**7,
                             verbose=False)
    assert r["chosen_regime"] != "FTQC"
    assert r["ftqc_meets_target"] is False


def test_uncertainty_band_optimistic_never_costs_more():
    # near the sd6 limit, so the +/- sigma shift changes d (43/43/45)
    band = uncertainty_band(16, 3930, p_phys=0.0036)
    opt, mid, pes = (band[k]["codeDistance"] for k in ("optimista", "central", "pesimista"))
    assert opt <= mid <= pes and opt < pes


def test_ftqc_noise_band_brackets_every_model():
    band = qregime.estimate_ftqc_band(16, 3930, 0.001)
    assert set(band["per_model"]) == set(qregime.NOISE_MODELS)
    costs = [m["physical_qubits"] for m in band["per_model"].values()]
    assert band["min_qubits"] == min(costs) and band["max_qubits"] == max(costs)
    assert band["near_threshold"] is False


def test_ftqc_noise_band_near_threshold_is_open_ended():
    band = qregime.estimate_ftqc_band(16, 3930, 0.0045)
    assert band["near_threshold"] is True
    assert band["feasible_models"] == ["toy"]
    assert band["max_qubits"] is None  # unbounded: some model finds FTQC infeasible
