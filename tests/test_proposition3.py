"""Proposition 3 of the paper, checked numerically over a parameter grid:
cost and reliability are strictly increasing in code distance, so the
best reliability reachable under a budget can never decrease as the
budget grows."""
import itertools

import pytest

import qregime
from qregime.resources import (NOISE_MODELS, _load_fit, litinski_15to1_factory_qubits,
                               physical_qubits_per_logical, predicted_logical_error_rate)


def cost(n, d):
    return n * physical_qubits_per_logical(d) + litinski_15to1_factory_qubits(d)


def reliability(t, p, d, fit):
    return (1 - predicted_logical_error_rate(d, p, fit)) ** t


def test_cost_strictly_increasing():
    for n in (1, 4, 16, 100):
        costs = [cost(n, d) for d in range(3, 52, 2)]
        assert all(a < b for a, b in zip(costs, costs[1:]))


@pytest.mark.parametrize("noise_model", NOISE_MODELS)
def test_reliability_non_decreasing_below_threshold(noise_model):
    fit = _load_fit(noise_model)
    for t, frac in itertools.product((1, 100, 16 * 3930), (0.1, 0.5, 0.9)):
        p = frac * fit["p_th"]
        rel = [reliability(t, p, d, fit) for d in range(3, 52, 2)]
        assert all(a <= b for a, b in zip(rel, rel[1:]))


@pytest.mark.parametrize("noise_model,p", [("sd6", 0.002), ("si1000", 0.002), ("toy", 0.001)])
def test_budget_staircase_never_regresses(noise_model, p):
    previous = -1.0
    for budget in range(1_000, 40_001, 500):
        r = qregime.best_within_budget(16, 3930, p, budget, noise_model=noise_model)
        rel = r["reliability"] if r["realizable"] else 0.0
        assert rel >= previous
        previous = rel
