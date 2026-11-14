"""Cross-check qregime's surface-code model with Riverlane's Deltakit.

qregime turns every FTQC / partial-QEC answer into a code distance d using a
fitted model of the logical error rate of one rotated surface-code patch:

    p_L(d) = A * (p / p_th) ** ((d + 1) / 2)       (per patch, per d rounds)

with (A, p_th) fitted per noise model by research/qec/fit_threshold_deltakit.py
and shipped in src/qregime/data/threshold_fit_<noise>.json. This script checks:

  1. Qubit count: qregime's 2d^2 per logical qubit vs. the 2d^2 - 1 qubits a
     bare rotated patch actually uses (the extra qubit per tile is the
     lattice-surgery routing allowance of Litinski 2019).
  2. The legacy fit (0.74%, threshold_fit_result.json, no longer used) is
     reproducible: Stim's generated circuits with its uniform noise, decoded
     through Deltakit. Deltakit's SD6 (0.65%, native CX) sits below it: SD6 adds idle noise, which
     Stim's generator does not add on idle qubits.
  3. Held-out validation of the shipped fits: fresh samples at (d, p) points
     not in the fit, including d = 13, beyond the fitted range (d <= 11).
     Measured / predicted p_L should stay within a factor ~2.
  4. The flagship circuit through qregime on each hardware profile.

Run (about 2-3 minutes):

    pip install 'qregime[deltakit]'
    python examples/deltakit_crosscheck.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    import deltakit_stim as stim
    from deltakit.circuit import Circuit
    from deltakit.circuit.gates import PauliBasis
    from deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit
    from deltakit.explorer.qpu import SI1000Noise, ToyNoise
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This example needs Deltakit: pip install 'qregime[deltakit]'") from exc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "qec"))
import fit_threshold_deltakit as fitter  # noqa: E402  (same measure/fit code as the shipped fits)

import qregime  # noqa: E402
from qregime import resources  # noqa: E402

LEGACY_FIT = os.path.join(os.path.dirname(resources.__file__), "data", "threshold_fit_result.json")
FLAGSHIP = (16, 3930)  # logical qubits, T-count

# (noise model name, Deltakit class, held-out (d, p) points)
HELD_OUT = (
    ("si1000", SI1000Noise, ((5, 0.00125), (7, 0.00175), (9, 0.00225), (13, 0.0015), (13, 0.002))),
    ("toy", ToyNoise, ((5, 0.0025), (7, 0.0035), (9, 0.0045), (13, 0.003), (13, 0.004))),
)


def stim_uniform_circuit(d: int, p: float) -> Circuit:
    """Circuit family of the legacy fit (research/qec/qec_surface_code_experiment.py)."""
    return Circuit.from_stim_circuit(stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p))


def used_qubits(stim_circuit) -> int:
    return len({t.value for inst in stim_circuit.flattened()
                for t in inst.targets_copy() if t.is_qubit_target})


def section_qubits():
    print("\n1. Physical qubits per logical qubit")
    print("     d   qregime (2d^2)   stim.num_qubits   stim used   Deltakit patch")
    for d in (3, 5, 11, 29):
        sc = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=1)
        dk = css_code_memory_circuit(RotatedPlanarCode(d, d), 1, PauliBasis.Z)
        print(f"    {d:2d}   {resources.physical_qubits_per_logical(d):14,}   "
              f"{sc.num_qubits:15,}   {used_qubits(sc):9,}   {len(dk.qubits):14,}")


def section_legacy():
    import json
    print("\n2. Legacy fit reproduced (Stim-generated circuits, decoded through Deltakit)")
    points = []
    for d in (5, 7, 9):
        for p in (0.002, 0.003, 0.004, 0.005):
            m = fitter.measure_circuit(stim_uniform_circuit(d, p))
            points.append({"d": d, "p": p, **m})
    f = fitter.fit(points)
    with open(LEGACY_FIT) as fh:
        legacy = json.load(fh)
    sd6 = resources._load_fit("sd6")
    print(f"    legacy shipped (Stim, 20k shots)    p_th = {legacy['p_th'] * 100:.3f}% "
          f"+/- {legacy['p_th_stderr'] * 100:.3f}")
    print(f"    reproduced via Deltakit             p_th = {f['p_th'] * 100:.3f}% "
          f"+/- {f['p_th_stderr'] * 100:.3f}   chi2/dof {f['chi2_per_dof']:.2f}")
    print(f"    Deltakit SD6 (adds idle noise)      p_th = {sd6['p_th'] * 100:.3f}% "
          f"+/- {sd6['p_th_stderr'] * 100:.3f}")


def section_held_out():
    print("\n3. Held-out validation of the shipped fits (points not used in the fit)")
    print("    noise    d   p        measured p_L        predicted   ratio")
    worst = 1.0
    for name, noise_cls, pts in HELD_OUT:
        fit = resources._load_fit(name)
        for d, p in pts:
            m = fitter.measure(d, noise_cls(p=p))
            pred = resources.predicted_logical_error_rate(d, p, fit)
            ratio = m["p_L"] / pred
            worst = max(worst, ratio, 1 / ratio)
            print(f"    {name:7s} {d:2d}  {p:.5f}  {m['p_L']:.2e} +/- {m['stderr']:.1e}  "
                  f"{pred:.2e}   {ratio:5.2f}")
    print(f"    worst factor off: {worst:.2f}  ->  {'OK' if worst < 2 else 'CHECK'} (target < 2)")


def section_flagship():
    n, t = FLAGSHIP
    print(f"\n4. Flagship circuit (n={n}, T-count={t:,}) through qregime")
    for key, hw in qregime.HARDWARE_PROFILES.items():
        e = qregime.estimate_ftqc(n, t, hw.p_2q, noise_model=hw.noise_model)
        out = (f"d={e['codeDistance']}, {e['physicalQubits']:,} qubits" if e["feasible"]
               else f"not feasible: {e['reason']}")
        print(f"    {key:27s} p={hw.p_2q:<7g} {hw.noise_model:7s} FTQC {out}")


def main():
    section_qubits()
    section_legacy()
    section_held_out()
    section_flagship()


if __name__ == "__main__":
    main()
