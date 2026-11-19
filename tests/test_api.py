"""Public API behaviour, pinned to numbers reported in the paper."""
import qregime


def test_profile_has_core_fields(circuit):
    p = qregime.profile(circuit("vqe_ansatz"))
    for key in ("num_qubits", "num_gates", "tcount", "two_qubit_count",
                "max_concurrent_qubits", "zz_cost_operator_motif_fraction"):
        assert key in p
    assert p["num_qubits"] == 8


def test_recommendations_use_english_keys(circuit):
    recs = qregime.recommend(circuit("qft10"))
    assert "zx_calculus_tcount_reduction" in recs
    assert all({"recommended", "reason"} <= set(v) for v in recs.values())


def test_ftqc_estimate_matches_paper():
    # Flagship circuit of the paper: 16 logical qubits, T-count 3,930.
    # Trapped-ion H2 (p=0.1%, ToyNoise fit): feasible at d=11.
    r = qregime.estimate_ftqc(16, 3930, 0.001, noise_model="toy")
    assert r["feasible"]
    assert r["codeDistance"] == 11
    assert r["physicalQubits"] == 6534
    # Superconducting (p=0.32%, SI1000 fit, p_th ~0.40%): too close to
    # threshold, no distance up to 51 meets the 1% error budget.
    r = qregime.estimate_ftqc(16, 3930, 0.0032, noise_model="si1000")
    assert r["feasible"] is False


def test_best_within_budget_matches_paper_table():
    r = qregime.best_within_budget(16, 3930, 0.001, 3_000, noise_model="toy")
    assert r["code_distance"] == 7
    assert r["cost_physical_qubits"] == 2646
    assert abs(r["reliability"] - 0.597) < 0.01
    assert r["confidence_quartile"] == 3


def test_above_threshold_is_reported_not_forced():
    r = qregime.best_within_budget(16, 3930, 0.05, 10**9)
    assert r["realizable"] is False


def test_decide_regime_small_circuit_picks_qem(circuit):
    r = qregime.decide_regime(circuit("vqe_ansatz"))
    assert r["regime"] == "QEM"


def test_uninformative_simulation_metric_is_not_used(circuit):
    # VQE ansatz output covers every bitstring: every noisy shot would "succeed".
    # With a calibration, qregime uses the mirror estimate instead...
    r = qregime.decide_regime(circuit("vqe_ansatz"))
    assert "mirror estimate" in r["qem"]["reliability_source"]
    assert r["qem"]["calibration"]["metric"] == "mirror"
    assert 0 < r["qem"]["reliability"] < 1.0
    # ...without one, it falls back to the analytic proxy, not the support metric
    r = qregime.decide_regime(circuit("vqe_ansatz"), use_calibration=False)
    assert r["qem"]["reliability_source"].startswith("analytic_proxy")
    assert r["qem"]["reliability"] < 1.0


def test_unknown_hardware_raises(circuit):
    import pytest
    with pytest.raises(ValueError):
        qregime.decide_regime(circuit("vqe_ansatz"), hardware="no_such_device")
