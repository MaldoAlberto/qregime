"""Audit: does qregime give what theory and the literature say it should?

Each check prints PASS / WARN / FAIL with the expected value, the observed
value and its source. Theory checks are exact or statistical; literature
checks compare against numbers quoted verbatim from the cited papers.

    python research/audit/literature_audit.py          # ~3-5 min (Deltakit part)
    python research/audit/literature_audit.py --fast   # skip Deltakit simulations

Literature values used (verified against the papers / spec sheet):
  [Fowler12]  Fowler, Mariantoni, Martinis, Cleland, PRA 86, 032324 (2012),
              arXiv:1208.0928, Eq. (11): P_L ~= 0.03 (p/p_th)^((d+1)/2),
              p_th = 0.57% (every operation incl. idle fails with p);
              "a logical qubit will need to contain 10^3 - 10^4 physical
              qubits in order to achieve logical error rates below
              10^-14 - 10^-15" at p <~ 10^-3 (unrotated, (2d-1)^2 qubits).
  [Gidney21]  Gidney, Newman, Fowler, Broughton, arXiv:2108.10457, Sec. 4:
              rotated surface code threshold "within 0.5%-0.7%" (SD6) and
              "0.3%-0.5%" (SI1000), correlated MWPM decoder, per d-round
              block; Sec. 4: "the rotated surface code requires 2d^2-1"
              qubits.
  [Google24]  Google Quantum AI, arXiv:2408.13687 (Nature 2025): Lambda =
              2.14 +/- 0.02 (neural-network decoder), 2.04 +/- 0.02
              (ensembled matching), d = 3, 5, 7, per-cycle error.
  [Willow]    Willow spec sheet (Dec 2024): CZ error 0.33%, single-qubit
              0.035%, measurement 0.77% (means).
  [Litinski19] arXiv:1808.02892, Sec. 4.1: 15-to-1 factory at d=13 uses
              11 x 338 = 3,718 physical qubits.
"""
from __future__ import annotations

import glob
import math
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import qregime
from qregime import resources
from qregime.partial_qec import best_feasible_within_hard_budget
from qregime.resources import (NOISE_MODELS, _load_fit, litinski_15to1_factory_qubits,
                               physical_qubits_per_logical, predicted_logical_error_rate,
                               solve_min_distance)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CIRCUITS = sorted(glob.glob(os.path.join(ROOT, "examples", "circuits", "*.qasm")))
FAST = "--fast" in sys.argv
RESULTS = []


def report(check_id, status, claim, expected, observed, note=""):
    RESULTS.append((check_id, status))
    print(f"[{status:4s}] {check_id:4s} {claim}")
    print(f"             expected: {expected}")
    print(f"             observed: {observed}")
    if note:
        print(f"             note:     {note}")


# ----------------------------------------------------------------------------
# THEORY
# ----------------------------------------------------------------------------
def t1_closed_form_distance():
    """d* = smallest odd d >= 3 with A (p/p_th)^((d+1)/2) <= eps/(n T)."""
    rng = random.Random(0)
    mismatches, n_cases = 0, 0
    for _ in range(2000):
        model = rng.choice(NOISE_MODELS)
        fit = _load_fit(model)
        p = fit["p_th"] * rng.uniform(0.01, 0.95)
        n, t = rng.randint(1, 200), rng.randint(1, 10**6)
        target = 1e-2 / (n * t)
        k = math.log(target / fit["A"]) / math.log(p / fit["p_th"])
        d = max(3, 2 * math.ceil(k) - 1)
        if d % 2 == 0:
            d += 1
        expected = d if d <= 51 else None
        got = solve_min_distance(p, target, fit)
        n_cases += 1
        mismatches += expected != got
    report("T1", "PASS" if mismatches == 0 else "FAIL",
           "estimator distance equals the closed-form solution of the threshold model",
           "0 mismatches", f"{mismatches} / {n_cases} random (model, n, T, p) cases")


def t2_lambda_from_measurements():
    """Lambda = p_L(d)/p_L(d+2) should equal p_th/p for the fitted model."""
    worst = 0.0
    rows = []
    for model in NOISE_MODELS:
        fit = _load_fit(model)
        pts = {(r["d"], r["p"]): r for r in fit["points"] if r["fails"] >= 30}
        for (d, p), r in pts.items():
            nxt = pts.get((d + 2, p))
            if d < 5 or nxt is None or p > 0.6 * fit["p_th"]:
                continue
            lam_meas = r["p_L"] / nxt["p_L"]
            lam_model = fit["p_th"] / p
            dev = abs(math.log(lam_meas / lam_model))
            worst = max(worst, dev)
            rows.append(f"{model} d={d}->{d + 2} p={p}: {lam_meas:.2f} vs {lam_model:.2f}")
    status = "PASS" if math.exp(worst) < 1.35 else "WARN"
    report("T2", status, "measured error suppression Lambda matches p_th/p (fitted points)",
           "within a factor 1.35", f"worst factor {math.exp(worst):.2f} over {len(rows)} pairs",
           "; ".join(rows[:3]) + " ...")


def t3_monotone_in_hardware_and_size():
    bad = []
    for model in NOISE_MODELS:
        p_th = _load_fit(model)["p_th"]
        prev = 0
        for frac in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            r = qregime.estimate_ftqc(16, 3930, frac * p_th, noise_model=model)
            q = r["physicalQubits"] if r["feasible"] else math.inf
            if q < prev:
                bad.append(f"{model} p={frac}p_th")
            prev = q
        for n1, n2 in ((4, 16), (16, 64)):
            a = qregime.estimate_ftqc(n1, 3930, 0.1 * p_th, noise_model=model)["physicalQubits"]
            b = qregime.estimate_ftqc(n2, 3930, 0.1 * p_th, noise_model=model)["physicalQubits"]
            if b < a:
                bad.append(f"{model} n {n1}->{n2}")
        for t1, t2 in ((100, 10_000), (10_000, 10**6)):
            a = qregime.estimate_ftqc(16, t1, 0.1 * p_th, noise_model=model)["physicalQubits"]
            b = qregime.estimate_ftqc(16, t2, 0.1 * p_th, noise_model=model)["physicalQubits"]
            if b < a:
                bad.append(f"{model} T {t1}->{t2}")
    report("T3", "PASS" if not bad else "FAIL",
           "FTQC cost never decreases with higher p, more logical qubits or more T gates",
           "monotone", "monotone" if not bad else ", ".join(bad))


def t4_qem_simulation_vs_proxy():
    """Simulated QEM success >= (1-p)^n2q (proxy counts every 2q error as fatal)."""
    from qregime.tool import simulate_expectation
    bad, n = [], 0
    for path in CIRCUITS:
        prof = qregime.profile(path)
        if prof["num_qubits"] > qregime.MAX_QUBITS_FOR_SIMULATION:
            continue
        for key, hw in qregime.HARDWARE_PROFILES.items():
            sim = simulate_expectation(path, hw, shots=400, verbose=False)
            if sim["ideal_support_size"] / 2 ** prof["num_qubits"] > 0.5:
                continue  # uninformative metric, qregime does not use it either
            s = sim["success_fraction_measured"]
            proxy = (1 - hw.p_2q) ** prof["two_qubit_count"]
            sigma = math.sqrt(max(proxy * (1 - proxy), 1e-4) / 400)
            n += 1
            if s < proxy - 3 * sigma:
                bad.append(f"{os.path.basename(path)}@{key}: {s:.3f} < {proxy:.3f}")
    report("T4", "PASS" if not bad else "FAIL",
           "noisy-simulation QEM success is never below the all-errors-fatal proxy",
           "sim >= proxy - 3 sigma", f"{n - len(bad)}/{n} circuit x hardware cases",
           "; ".join(bad))


def t5_qem_ordered_by_hardware():
    bad = []
    order = ["superconductor_ibm", "trapped_ion_quantinuum_h2", "trapped_ion_ionq_oxford"]
    for path in CIRCUITS:
        prof = qregime.profile(path)
        rel = [(1 - qregime.HARDWARE_PROFILES[k].p_2q) ** prof["two_qubit_count"] for k in order]
        if not (rel[0] <= rel[1] <= rel[2]):
            bad.append(os.path.basename(path))
    report("T5", "PASS" if not bad else "FAIL",
           "QEM reliability improves monotonically IBM -> H2 -> Oxford for every circuit",
           "ordered", "ordered for all" if not bad else ", ".join(bad))


def t6_proposition1():
    import pyzx as zx
    from qregime.hypergraph import resource_report
    from qregime.zx_reduction import zx_reduce
    rows, bad = [], []
    for path in CIRCUITS:
        name = os.path.basename(path)[:-5]
        motif = qregime.profile(path)["zz_cost_operator_motif_fraction"]
        if motif < 1.0 and name not in ("adder8", "qft10"):
            continue
        circ = zx.Circuit.from_qasm_file(path).to_basic_gates()
        before = resource_report(circ)["tcount"]
        after = resource_report(zx_reduce(circ))["tcount"]
        red = (before - after) / before if before else 0.0
        rows.append(f"{name} motif={motif:.1f} reduction={red:.0%}")
        if motif >= 1.0 and red != 0:
            bad.append(name)
    report("T6", "PASS" if not bad else "FAIL",
           "Proposition 1: ZX-calculus gives exactly 0% T-count reduction on the ZZ-layer motif",
           "0% where motif = 1", "; ".join(rows))


def t7_clifford_and_staircase():
    e = qregime.estimate_ftqc(5, 0, 1e-3, noise_model="toy")
    prev, ok = -1.0, True
    for model, p in (("si1000", 0.002), ("toy", 0.001), ("sd6", 0.002)):
        prev = -1.0
        for b in range(500, 60_001, 250):
            r = best_feasible_within_hard_budget(16, 3930, p, b, verbose=False, noise_model=model)
            rel = r["reliability"] if r["realizable"] else 0.0
            ok &= rel >= prev
            prev = rel
    status = "PASS" if (e["feasible"] and e["physicalQubitsForFactories"] == 0 and ok) else "FAIL"
    report("T7", status, "Clifford-only circuits need no factories; Proposition 3 staircase never steps down",
           "0 factory qubits; monotone staircase (3 noise models x 240 budgets)",
           f"factory qubits = {e.get('physicalQubitsForFactories')}; staircase monotone = {ok}")


# ----------------------------------------------------------------------------
# LITERATURE
# ----------------------------------------------------------------------------
def l1_gidney_thresholds():
    si, sd = _load_fit("si1000")["p_th"], _load_fit("sd6")["p_th"]
    report("L1a", "PASS" if 0.003 <= si <= 0.005 else "WARN",
           "SI1000 rotated-surface-code threshold [Gidney21]",
           "0.3%-0.5%", f"{si * 100:.3f}%")
    report("L1b", "PASS" if 0.005 <= sd <= 0.007 else "WARN",
           "SD6 rotated-surface-code threshold [Gidney21]",
           "0.5%-0.7%", f"{sd * 100:.3f}%",
           "fit uses native CX gates, as Gidney21's SD6 does (compiling CX to H-CZ-H gave "
           "0.44%); qregime decodes with plain PyMatching, Gidney21 with a correlated MWPM "
           "decoder, which adds ~0.1 points at d=5->9.")


def l2_fowler():
    sd = _load_fit("sd6")
    rows = []
    for d in (5, 11, 21):
        ours = predicted_logical_error_rate(d, 1e-3, sd)
        fowler = 0.03 * (1e-3 / 0.0057) ** ((d + 1) / 2)
        rows.append((d, ours / fowler))
    worst = max(max(r, 1 / r) for _, r in rows)
    report("L2a", "PASS" if worst < 10 else "WARN",
           "logical error rate at p = 0.1% vs Fowler12 Eq. (11) (SD6-like noise)",
           "same order of magnitude (A = 0.03, p_th = 0.57%)",
           f"A = {sd['A']:.3f}, p_th = {sd['p_th'] * 100:.2f}%; qregime/Fowler at d=5,11,21: "
           + ", ".join(f"{r:.1f}x" for _, r in rows),
           "ratio < 1 at large d because the fitted p_th (0.65%) is above Fowler12's 0.57%; "
           "A is about 2x Fowler12's.")
    lo = hi = None
    for target in (1e-14, 1e-15):
        d = solve_min_distance(1e-3, target, sd)
        q = physical_qubits_per_logical(d)
        lo, hi = (q if lo is None else min(lo, q)), (q if hi is None else max(hi, q))
    report("L2b", "PASS" if 1e3 <= lo and hi <= 1e4 else "WARN",
           "physical qubits per logical qubit for P_L = 1e-14..1e-15 at p = 0.1% [Fowler12]",
           "10^3 - 10^4", f"{lo:,} - {hi:,} (SD6 fit, 2d^2)")


def l3_qubit_counts():
    ok = all(physical_qubits_per_logical(d) == 2 * d * d for d in (3, 5, 29)) and \
        litinski_15to1_factory_qubits(13) == 3718
    report("L3", "PASS" if ok else "FAIL",
           "patch = 2d^2-1 qubits [Gidney21] + 1 routing qubit; 15-to-1 factory at d=13 [Litinski19]",
           "2d^2 per tile; 3,718 at d=13",
           f"2d^2 at d=3,5,29 = {[physical_qubits_per_logical(d) for d in (3, 5, 29)]}; "
           f"factory(13) = {litinski_15to1_factory_qubits(13):,}")


def _per_cycle(p_block, d):
    """Per-cycle logical error from a d-round block error (standard conversion)."""
    return (1 - (1 - 2 * p_block) ** (1 / d)) / 2


LAMBDA_WILLOW_MATCHING = 2.04


def l4_willow():
    si = _load_fit("si1000")
    p = 0.0033  # Willow mean CZ error
    lam_model = _per_cycle(predicted_logical_error_rate(5, p, si), 5) / \
        _per_cycle(predicted_logical_error_rate(7, p, si), 7)
    report("L4a", "PASS" if lam_model >= 1.5 else "WARN",
           "Willow-class device (CZ 0.33%) under qregime's SI1000 mapping [Google24, Willow]",
           "Lambda(5->7, per cycle) = 2.04 (matching) / 2.14 (neural net)",
           f"Lambda = {lam_model:.2f}",
           "SI1000 sets measurement = 5p = 1.65% but Willow measures 0.77% (2.3p): the "
           "mapping is pessimistic for this hardware. Verdicts at p_2q ~0.3% under SI1000 "
           "(e.g. the IBM profile) are on the conservative side.")
    if FAST:
        return
    sys.path.insert(0, os.path.join(ROOT, "research", "qec"))
    import fit_threshold_deltakit as F
    from deltakit.explorer.qpu import ToyNoise
    # Willow-like circuit noise: 2q 0.33%, 1q & idle 0.033%, measurement 0.77%
    pl = {d: F.measure(d, ToyNoise(p=0.0033, p_measurement_flip=0.0077))["p_L"] for d in (3, 5, 7)}
    eps = {d: _per_cycle(pl[d], d) for d in pl}
    lam35, lam57 = eps[3] / eps[5], eps[5] / eps[7]
    si_lam = _per_cycle(predicted_logical_error_rate(5, 0.0033, si), 5) / \
        _per_cycle(predicted_logical_error_rate(7, 0.0033, si), 7)
    bracket = si_lam <= LAMBDA_WILLOW_MATCHING <= lam57
    report("L4b", "PASS" if bracket else "WARN",
           "measured Willow Lambda lies between qregime's SI1000 mapping and a Pauli model "
           "built from Willow's headline component errors",
           f"SI1000 Lambda <= {LAMBDA_WILLOW_MATCHING} (Google24, matching) <= headline-Pauli Lambda",
           f"{si_lam:.2f} <= {LAMBDA_WILLOW_MATCHING} <= {lam57:.2f} (headline-Pauli "
           f"Lambda(3->5) = {lam35:.2f}, eps_7 = {eps[7]:.2e}/cycle vs measured 1.71e-3)",
           "A Pauli model from headline numbers is optimistic (no leakage, crosstalk, "
           "correlated errors, idling during measurement); SI1000's 5p measurement is "
           "pessimistic. qregime's SI1000 Lambda is within ~16% of the real device.")


def main():
    print("== THEORY ==")
    for check in (t1_closed_form_distance, t2_lambda_from_measurements,
                  t3_monotone_in_hardware_and_size, t4_qem_simulation_vs_proxy,
                  t5_qem_ordered_by_hardware, t6_proposition1, t7_clifford_and_staircase):
        check()
    print("\n== LITERATURE ==")
    for check in (l1_gidney_thresholds, l2_fowler, l3_qubit_counts, l4_willow):
        check()
    counts = {s: sum(1 for _, x in RESULTS if x == s) for s in ("PASS", "WARN", "FAIL")}
    print(f"\nSummary: {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
