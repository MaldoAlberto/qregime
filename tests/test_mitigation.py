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

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _mirror_qasm(tmp_path, layers=2):
    import numpy as np

    rng = np.random.default_rng(1)
    ops = []
    for layer in range(layers):
        ops += [(f"ry", float(rng.uniform(0.1, 3.0)), q) for q in range(3)]
        ops.append(("cx", None, layer % 2))
    lines = []
    for name, angle, q in ops + [(n, -a if a is not None else None, q) for n, a, q in reversed(ops)]:
        lines.append(f"ry({angle!r}) q[{q}];" if name == "ry" else f"cx q[{q}],q[{q + 1}];")
    path = tmp_path / "mirror.qasm"
    path.write_text(HEADER + "qreg q[3];\n" + "\n".join(lines) + "\n")
    return str(path)


def test_fold_reaches_scale():
    import numpy as np
    from qregime.calibration import _fold

    ops = [("1q", (0,), np.eye(2))] * 5 + [("2q", (0, 1), np.eye(4), "cz")] * 5
    for scale in (1, 2, 3):
        folded, reached = _fold(ops, scale)
        assert reached == pytest.approx(scale)
        assert len(folded) == pytest.approx(scale * len(ops))


def test_simulate_zne_unavailable_reasons(tmp_path):
    from qregime.mitigation import simulate_zne

    text = HEADER + "qreg q[11];\nx q;\n"
    path = tmp_path / "big.qasm"
    path.write_text(text)
    assert "density-matrix limit" in simulate_zne(str(path))["reason"]


@pytest.mark.parametrize("device", ["ibm_torino", "quantinuum_h2_2"])
def test_simulate_zne_on_calibration(tmp_path, device):
    from qregime.mitigation import simulate_zne

    r = simulate_zne(_mirror_qasm(tmp_path), device)
    assert r["available"] and r["scale_factors"] == pytest.approx([1, 2, 3], abs=0.1)
    rem = r["zne_with_readout_mitigation"]
    assert rem["values"][0] > rem["values"][1] > rem["values"][2]
    assert rem["values"][0] >= r["zne"]["values"][0]
    # a shallow circuit: ZNE + readout mitigation recovers the ideal value
    assert r["best_abs_bias"] < 0.02


def test_decide_regime_uses_calibration(tmp_path):
    path = _mirror_qasm(tmp_path)
    calibrated = qregime.decide_regime(path)
    assert calibrated["qem"]["reliability_source"].startswith("calibrated_simulation (ibm_torino")
    assert calibrated["qem"]["calibration"]["success_fraction"] == calibrated["qem"]["reliability"]
    ion = qregime.decide_regime(path, hardware="trapped_ion_quantinuum_h2")
    assert "quantinuum_h2_2" in ion["qem"]["reliability_source"]
    assert ion["qem"]["reliability"] > calibrated["qem"]["reliability"]
    plain = qregime.decide_regime(path, use_calibration=False)
    assert plain["qem"]["reliability_source"].startswith("noisy_simulation")
    assert plain["qem"]["calibration"] is None
    no_cal = qregime.decide_regime(path, hardware="trapped_ion_ionq_oxford")
    assert no_cal["qem"]["calibration"]["available"] is False
    # an explicit calibration and a measured value
    assert "quantinuum_h2_2" in qregime.decide_regime(
        path, calibration="quantinuum_h2_2")["qem"]["reliability_source"]
    assert qregime.decide_regime(path, qem_success_fraction=0.3)["qem"]["reliability"] == 0.3
