"""Case study: IBM Nighthawk r2 surface-code memory (arXiv:2610.23840, Oct 2026).

Kan et al. ran rotated surface-code memories on ibm_phoenix (IBM Nighthawk r2,
120 qubits, square lattice) and measured, with exact maximum-likelihood
decoding, a logical error per round falling from 0.526% (d=3) to 0.244% (d=6)
in Z memory, Lambda = 1.68 [1.57, 1.81] in Z and 1.85 [1.71, 2.00] in X.

This script rebuilds that experiment in Deltakit from the paper's own
calibration (Appendix B1, Table 1, day 2 = the distance-scaling run) and
timings (Methods .1), and asks two questions:

  1. Does a calibration-derived circuit noise model reproduce the measured
     error per round and Lambda? (The paper itself reports that correlations
     build up over long memories, possibly leakage or drift, that its model
     does not include.)
  2. What does the measured Lambda imply for qregime's fault-tolerance cost
     of a real circuit on this hardware?

Noise model (Deltakit PhysicalNoise, matching the paper's Methods .4):
  - depolarizing noise per gate, matched to the calibrated AVERAGE gate error
    (2q: p_depol = e_avg * 5/4; 1q: p_depol = e_avg * 3/2);
  - Pauli-twirled T1/T2 idle noise for the actual idle durations;
  - measurement flips with the calibrated readout error; the same probability
    after reset, as in the paper.
CZ is native (68 ns), 1q gates 40 ns, short readout measure_2 1.14 us with
fused reset (rounds of ~1.59 us). Decoding: PyMatching (MWPM). The paper finds
exact ML decoding fails 0.69-0.86 times as often as MWPM, so MWPM numbers are
expected to sit somewhat above the measured ML ones.

    pip install 'qregime[deltakit]'
    python research/case_studies/ibm_nighthawk_2026.py      # writes ibm_nighthawk_2026_results.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from unittest import mock

import numpy as np

warnings.filterwarnings("ignore")

from deltakit.circuit.gates import CZ, H, MZ, RZ, S, X, Y, Z, PauliBasis
from deltakit.decode import PyMatchingDecoder
from deltakit.decode.analysis import run_decoding_on_circuit
from deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit.explorer.qpu import QPU, NativeGateSetAndTimes, PhysicalNoise

HERE = os.path.dirname(os.path.abspath(__file__))

# arXiv:2610.23840, Appendix B1 Table 1, day 2 (distance-scaling run), medians.
CALIBRATION = {
    "t1_us": 184.0,
    "t2_us": 162.0,
    "sx_error": 2.1e-4,
    "cz_error": 1.6e-3,
    "measure_2_error": 0.46e-2,
}
# Methods .1
TIMES = {"cz_s": 68e-9, "one_qubit_s": 40e-9, "measure_2_s": 1.14e-6}
# Methods .4: without the echo, unprotected idles use an effective T2 = 0.3 T2.
NO_ECHO_T2_FACTOR = 0.3

# Measured (arXiv:2610.23840, abstract and Results), exact ML decoding,
# other distance 3, fit over 3-13 rounds.
MEASURED = {
    "Z": {"eps": {3: 0.526e-2, 6: 0.244e-2}, "lambda": 1.68, "lambda_ci": (1.57, 1.81)},
    "X": {"eps": {3: 0.523e-2, 6: 0.220e-2}, "lambda": 1.85, "lambda_ci": (1.71, 2.00)},
}

ROUNDS = (3, 5, 9, 13)
SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
MAX_SHOTS = 400_000
MIN_FAILS = 200


def noise_model(echo: bool = True, scale: float = 1.0) -> PhysicalNoise:
    """``scale`` multiplies every error probability and divides T1 and T2,
    i.e. a uniformly k-times-worse device than the median calibration."""
    c = CALIBRATION
    t1 = c["t1_us"] / scale
    t2 = c["t2_us"] * (1 if echo else NO_ECHO_T2_FACTOR) / scale
    t2 = min(t2, 2 * t1)  # physical bound, as in the paper
    return PhysicalNoise(
        t1=t1 * 1e-6,
        t2=t2 * 1e-6,
        p_1_qubit_gate_error=scale * c["sx_error"] * 3 / 2,
        p_2_qubit_gate_error=scale * c["cz_error"] * 5 / 4,
        p_reset_error=scale * c["measure_2_error"] * 3 / 2,  # depolarizing -> flip prob = measure error
        p_meas_qubit_error=0.0,
        p_readout_flip=scale * c["measure_2_error"],
    )


def native_gates() -> NativeGateSetAndTimes:
    one_q = {g: TIMES["one_qubit_s"] for g in (H, X, Y, Z, S)}
    return NativeGateSetAndTimes(
        one_qubit_gates=one_q,
        two_qubit_gates={CZ: TIMES["cz_s"]},
        reset_gates={RZ: 1e-9},  # fused into the readout (measure_reset_2)
        measurement_gates={MZ: TIMES["measure_2_s"]},
    )


def memory_circuit(width: int, height: int, rounds: int, basis: PauliBasis, echo: bool,
                   scale: float = 1.0):
    code = RotatedPlanarCode(width=width, height=height)
    circuit = css_code_memory_circuit(code, num_rounds=rounds, logical_basis=basis)
    qpu = QPU(circuit.qubits, native_gates_and_times=native_gates(), noise_model=noise_model(echo, scale))
    return qpu.compile_and_add_noise_to_circuit(circuit)


def sample(circuit) -> dict:
    decoder, stim_circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(circuit)
    with mock.patch("sys.stderr"):
        r = run_decoding_on_circuit(stim_circuit, MAX_SHOTS, decoder, max_batch_size=100_000,
                                    target_rse=0.05, min_fails=MIN_FAILS)
    shots, fails = int(r["shots"]), int(r["fails"])
    return {"shots": shots, "fails": fails, "p_L": fails / shots}


def error_per_round(points: dict) -> float:
    """Fit ln(1 - 2 P_L(r)) = a + r ln(1 - 2 eps), as for a memory decaying per round."""
    rs = np.array(sorted(points), float)
    y = np.log(np.array([1 - 2 * points[r] for r in sorted(points)]))
    slope = np.polyfit(rs, y, 1)[0]
    return (1 - math.exp(slope)) / 2


def run_patch(width, height, basis, echo, scale=1.0):
    pts, raw = {}, {}
    for r in ROUNDS:
        m = sample(memory_circuit(width, height, r, basis, echo, scale))
        pts[r], raw[r] = m["p_L"], m
    return error_per_round(pts), raw


def main():
    t0 = time.time()
    out = {"source": "arXiv:2610.23840", "calibration": CALIBRATION, "times": TIMES,
           "rounds": ROUNDS, "decoder": "PyMatching (MWPM)", "measured": MEASURED, "runs": []}
    # Which side of RotatedPlanarCode protects which memory is checked
    # empirically: the protected distance is the one that lowers eps.
    for echo in (True, False):
        for basis in (PauliBasis.Z, PauliBasis.X):
            for protected in (3, 4, 5, 6):
                for orient in ("width", "height"):
                    w, h = (protected, 3) if orient == "width" else (3, protected)
                    if protected == 3 and orient == "height":
                        continue
                    eps, raw = run_patch(w, h, basis, echo)
                    row = {"echo": echo, "basis": basis.name, "width": w, "height": h,
                           "eps": eps, "raw": raw}
                    out["runs"].append(row)
                    print(f"echo={echo!s:5} {basis.name} width={w} height={h}  "
                          f"eps/round = {eps:.4%}   ({time.time() - t0:.0f} s)", flush=True)
    # Uniform-degradation scan: is there one k for which the model matches
    # both the measured eps at d=3 and the measured Lambda? (Z memory, echo.)
    out["scale_scan"] = []
    for scale in SCALES:
        row = {"scale": scale}
        for d in (3, 6):
            row[f"eps_d{d}"], _ = run_patch(d, 3, PauliBasis.Z, True, scale)
        row["lambda"] = (row["eps_d3"] / row["eps_d6"]) ** (2 / 3)
        out["scale_scan"].append(row)
        print(f"scale={scale:4.2f}  eps d=3 {row['eps_d3']:.4%}  d=6 {row['eps_d6']:.4%}  "
              f"Lambda {row['lambda']:.2f}   ({time.time() - t0:.0f} s)", flush=True)
    path = os.path.join(HERE, "ibm_nighthawk_2026_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
