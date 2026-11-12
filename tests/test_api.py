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


def test_ftqc_estimate_flagship_circuit():
    # Flagship circuit of the paper: 16 logical qubits, T-count 3,930, p=0.32%.
    # The paper reported d=23 / 29,654 qubits, which shared the error budget
    # among the T gates only; with all 16 patches failing independently at
    # every step, d=23 reaches ~91%, not the declared 99% (audit finding #1).
    r = qregime.estimate_ftqc(16, 3930, 0.0032)
    assert r["feasible"]
    assert r["codeDistance"] == 29
    assert r["physicalQubits"] == 46790


def test_best_within_budget_flagship_circuit():
    # Same d=13 point as the paper's table, but its reliability is
    # (1 - p_L)^(16 * 3930) ~= 0.2%, not (1 - p_L)^3930 ~= 68% (finding #1).
    r = qregime.best_within_budget(16, 3930, 0.0032, 10_000)
    assert r["code_distance"] == 13
    assert r["cost_physical_qubits"] == 9734
    assert r["reliability"] < 0.01
    assert r["confidence_quartile"] == 1


def test_above_threshold_is_reported_not_forced():
    r = qregime.best_within_budget(16, 3930, 0.05, 10**9)
    assert r["realizable"] is False


def test_decide_regime_small_circuit_picks_qem(circuit):
    r = qregime.decide_regime(circuit("vqe_ansatz"))
    assert r["regime"] == "QEM"


def test_uninformative_simulation_metric_is_not_used(circuit):
    # VQE ansatz output covers every bitstring: every noisy shot would "succeed".
    r = qregime.decide_regime(circuit("vqe_ansatz"))
    assert r["qem"]["reliability_source"].startswith("analytic_proxy")
    assert r["qem"]["reliability"] < 1.0


def test_unknown_hardware_raises(circuit):
    import pytest
    with pytest.raises(ValueError):
        qregime.decide_regime(circuit("vqe_ansatz"), hardware="no_such_device")
