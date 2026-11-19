"""ZNE sampling-cost model: sanity and agreement with Mitiq measurements."""
import json
import math
import os

import pytest

import qregime
from qregime.mitigation import _intercept_weights, zne_sampling_cost

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_intercept_weights_reproduce_line():
    w = _intercept_weights((1.0, 2.0, 3.0))
    assert w == pytest.approx([4 / 3, 1 / 3, -2 / 3])
    # extrapolating y = 5 - 2x to x = 0 must give 5
    assert sum(wi * (5 - 2 * x) for wi, x in zip(w, (1, 2, 3))) == pytest.approx(5)


def test_cost_grows_as_reliability_drops():
    shots = [zne_sampling_cost(f)["total_shots"] for f in (0.95, 0.8, 0.5, 0.2)]
    assert shots == sorted(shots) and shots[-1] > 100 * shots[0]


def test_precision_scales_inverse_square():
    a = zne_sampling_cost(0.5, precision=0.02)["shots_per_circuit"]
    b = zne_sampling_cost(0.5, precision=0.01)["shots_per_circuit"]
    assert b / a == pytest.approx(4, rel=1e-3)


def test_edge_cases():
    assert zne_sampling_cost(0.0)["feasible"] is False
    assert zne_sampling_cost(1.0)["shots_per_circuit"] == 1
    with pytest.raises(ValueError):
        zne_sampling_cost(0.5, scale_factors=(1, 1))
    with pytest.raises(ValueError):
        zne_sampling_cost(0.5, precision=0)


def test_matches_mitiq_measured_spread():
    """Predicted std at 10,000 shots vs. ExpFactory spread over 20 repeats
    (docs/qregime_to_mitiq.json). 20 repeats leave ~16% uncertainty on a std."""
    with open(os.path.join(ROOT, "docs", "qregime_to_mitiq.json")) as f:
        data = json.load(f)
    for row in data["rows"]:
        cost = zne_sampling_cost(row["raw_exact"], precision=1.0 / math.sqrt(data["shots"]))
        predicted_std = math.sqrt(cost["shots_per_circuit"] / data["shots"]) / math.sqrt(data["shots"])
        measured = row["zne"]["exponential"]["std"]
        assert 0.6 < predicted_std / measured < 1.7, row["cx"]


def test_decide_regime_reports_mitigation():
    path = os.path.join(ROOT, "examples", "circuits")
    qasm = next(os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".qasm"))
    m = qregime.decide_regime(qasm)["qem"]["mitigation"]
    assert m["method"] == "zne_exponential" and m["precision"] == 0.01


# --- simulated ZNE on a device calibration ---------------------------------

def _mirror_qasm(tmp_path, layers=2):
    import numpy as np
    from qiskit import QuantumCircuit

    rng = np.random.default_rng(1)
    u = QuantumCircuit(3)
    for layer in range(layers):
        for q in range(3):
            u.ry(float(rng.uniform(0.1, 3.0)), q)
        u.cx(layer % 2, layer % 2 + 1)
    path = tmp_path / "mirror.qasm"
    path.write_text(qregime.safe_qasm2_dumps(u.compose(u.inverse())))
    return str(path)


def test_fold_gates_reaches_scale():
    from qiskit import QuantumCircuit
    from qregime.mitigation import _fold_gates

    qc = QuantumCircuit(2)
    for _ in range(10):
        qc.sx(0)
        qc.cz(0, 1)
    for scale in (1, 2, 3):
        folded, reached = _fold_gates(qc, scale)
        assert reached == pytest.approx(scale)
        assert len(folded.data) == pytest.approx(scale * len(qc.data))
    # folding leaves the unitary unchanged
    from qiskit.quantum_info import Operator
    assert Operator(_fold_gates(qc, 2)[0]).equiv(Operator(qc))


def test_simulate_zne_unavailable_reasons(tmp_path):
    from qregime.mitigation import simulate_zne

    path = _mirror_qasm(tmp_path)
    assert simulate_zne(path, hardware="trapped_ion_quantinuum_h2")["available"] is False


def test_simulate_zne_on_calibration(tmp_path):
    pytest.importorskip("qiskit_ibm_runtime")
    from qregime.mitigation import simulate_zne

    r = simulate_zne(_mirror_qasm(tmp_path))
    assert r["available"] and r["scale_factors"] == [1.0, 2.0, 3.0]
    rem = r["zne_with_readout_mitigation"]
    # noise lowers the success fraction monotonically with the scale
    assert rem["values"][0] > rem["values"][1] > rem["values"][2]
    # readout mitigation can only help a success fraction here
    assert rem["values"][0] >= r["zne"]["values"][0]
    # a shallow circuit: ZNE + readout mitigation recovers the ideal value
    assert r["best_abs_bias"] < 0.02


def test_decide_regime_uses_calibration(tmp_path):
    pytest.importorskip("qiskit_ibm_runtime")
    path = _mirror_qasm(tmp_path)
    calibrated = qregime.decide_regime(path)
    assert calibrated["qem"]["reliability_source"].startswith("calibrated_simulation")
    assert calibrated["qem"]["calibration"]["success_fraction"] == calibrated["qem"]["reliability"]
    plain = qregime.decide_regime(path, use_calibration=False)
    assert plain["qem"]["reliability_source"].startswith("noisy_simulation")
    assert plain["qem"]["calibration"] is None
    # a measured value always wins
    assert qregime.decide_regime(path, qem_success_fraction=0.3)["qem"]["reliability"] == 0.3
