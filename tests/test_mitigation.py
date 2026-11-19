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
