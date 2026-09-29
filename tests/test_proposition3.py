"""Proposition 3 of the paper, checked numerically over a parameter grid:
cost and reliability are strictly increasing in code distance, so the
best reliability reachable under a budget can never decrease as the
budget grows."""
import itertools

import qregime
from qregime.resources import (_load_fit, litinski_15to1_factory_qubits,
                               physical_qubits_per_logical, predicted_logical_error_rate)

FIT = _load_fit()


def cost(n, d):
    return n * physical_qubits_per_logical(d) + litinski_15to1_factory_qubits(d)


def reliability(t, p, d):
    return (1 - predicted_logical_error_rate(d, p, FIT)) ** t


def test_cost_strictly_increasing():
    for n in (1, 4, 16, 100):
        costs = [cost(n, d) for d in range(3, 52, 2)]
        assert all(a < b for a, b in zip(costs, costs[1:]))


def test_reliability_non_decreasing_below_threshold():
    for t, p in itertools.product((1, 100, 3930), (0.001, 0.0032, 0.006)):
        assert p < FIT["p_th"]
        rel = [reliability(t, p, d) for d in range(3, 52, 2)]
        assert all(a <= b for a, b in zip(rel, rel[1:]))


def test_budget_staircase_never_regresses():
    previous = -1.0
    for budget in range(1_000, 40_001, 500):
        r = qregime.best_within_budget(16, 3930, 0.0032, budget)
        rel = r["reliability"] if r["realizable"] else 0.0
        assert rel >= previous
        previous = rel
