"""Fit qregime's surface-code threshold model per noise model, with Deltakit.

For each noise model this script measures the logical error rate of a
rotated surface-code Z-memory experiment (d rounds, Deltakit's
RotatedPlanarCode, decoded with Deltakit's PyMatching decoder) over a grid
of distances and physical error rates, then fits

    p_L(d) = A * (p / p_th) ** ((d + 1) / 2)     (per patch, per d rounds)

and writes ``src/qregime/data/threshold_fit_<noise>.json``, which
``qregime.resources._load_fit(noise_model)`` reads.

Noise models (all from Deltakit; sd6 and si1000 as in arXiv:2108.10457, Table 2):
  sd6     standard circuit-level depolarizing: every 1q/2q gate, every idle
          qubit at every tick, reset and measurement fail with probability p.
          Theory baseline and the default of the low-level API.
  si1000  superconducting-inspired: 2q gates p, 1q gates and idles p/10,
          measurement flip 5p, reset 2p. Used for the superconducting
          profile, with p = the profile's two-qubit error rate.
  toy     Deltakit ToyNoise: 2q gates p, 1q gates and idles p/10,
          measurement flip p. Used for the trapped-ion profiles (long
          coherence: idling is much cheaper than a two-qubit gate) --
          a declared approximation, not an ion-specific error model.

Fitting only uses points well below threshold (p <= 0.8 p_th, refined
iteratively) and d >= 5 (d = 3 deviates from the asymptotic form; qregime
still evaluates d = 3 as a declared extrapolation).

    python research/qec/fit_threshold_deltakit.py [noise ...]   # measure + fit
    python research/qec/fit_threshold_deltakit.py --refit       # refit saved points

    pip install 'qregime[deltakit]'
    python research/qec/fit_threshold_deltakit.py          # ~10-20 min
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

import deltakit
from deltakit.circuit.gates import PauliBasis
from deltakit.decode import PyMatchingDecoder
from deltakit.decode.analysis import run_decoding_on_circuit
from deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit.explorer.qpu import ExhaustiveGateSet, QPU, SD6Noise, SI1000Noise, ToyNoise

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "qregime", "data")

DISTANCES = (3, 5, 7, 9, 11)
GRIDS = {
    "sd6": (SD6Noise, (0.0015, 0.002, 0.003, 0.004, 0.005, 0.006)),
    "si1000": (SI1000Noise, (0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003)),
    "toy": (ToyNoise, (0.001, 0.002, 0.003, 0.004, 0.005, 0.006)),
}
FIT_MIN_D = 5
MAX_SHOTS = 2_000_000
MIN_FAILS = 30
SEED_NOTE = "Deltakit/Stim sampling is unseeded; reruns agree within the reported stderr."


# SD6 as defined in the literature (Gidney et al. 2021, Table 2) has native CX.
# Deltakit's default native gate set compiles CX to H-CZ-H, which under SD6
# adds two depolarizing(p) 1q gates per CX and lowers the threshold from
# ~0.69% to ~0.47%; so SD6 is simulated with every gate native. SI1000 and
# ToyNoise keep the default (CZ-native, as on superconducting hardware); their
# 1q error is p/10, so compilation barely matters there.
NATIVE_CX = {"sd6"}


def memory_circuit(d: int, noise):
    code = RotatedPlanarCode(width=d, height=d)
    circuit = css_code_memory_circuit(code, num_rounds=d, logical_basis=PauliBasis.Z)
    if isinstance(noise, SD6Noise):
        qpu = QPU(circuit.qubits, native_gates_and_times=ExhaustiveGateSet(), noise_model=noise)
    else:
        qpu = QPU(circuit.qubits, noise_model=noise)
    return qpu.compile_and_add_noise_to_circuit(circuit)


def measure(d: int, noise) -> dict:
    return measure_circuit(memory_circuit(d, noise))


def measure_circuit(circuit) -> dict:
    """Sample and decode any noisy Deltakit memory circuit."""
    decoder, stim_circuit = PyMatchingDecoder.construct_decoder_and_stim_circuit(circuit)
    with mock.patch("sys.stderr"):
        r = run_decoding_on_circuit(stim_circuit, MAX_SHOTS, decoder, max_batch_size=100_000,
                                    target_rse=0.1, min_fails=MIN_FAILS)
    shots, fails = int(r["shots"]), int(r["fails"])
    p_l = fails / shots
    return {"shots": shots, "fails": fails, "p_L": p_l,
            "stderr": math.sqrt(max(p_l * (1 - p_l), 1e-300) / shots)}


def _wls(pts):
    k = np.array([(r["d"] + 1) / 2 for r in pts])
    y = np.array([math.log(r["p_L"]) - kk * math.log(r["p"]) for r, kk in zip(pts, k)])
    w = np.array([r["p_L"] / r["stderr"] for r in pts])
    X = np.column_stack([np.ones_like(k), -k])
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    resid = (y - X @ coef) * w
    dof = max(len(pts) - 2, 1)
    chi2 = float(resid @ resid / dof)
    cov = np.linalg.inv((X * w[:, None]).T @ (X * w[:, None])) * max(chi2, 1.0)
    return coef, cov, chi2


def fit(points: list[dict]) -> dict:
    usable = [r for r in points
              if r["fails"] >= MIN_FAILS and r["p_L"] < 0.3 and r["d"] >= FIT_MIN_D]
    p_th = max(r["p"] for r in usable) / 0.8
    for _ in range(10):  # keep only p <= 0.8 p_th; converges in 2-3 passes
        pts = [r for r in usable if r["p"] <= 0.8 * p_th]
        coef, cov, chi2 = _wls(pts)
        new_p_th = math.exp(coef[1])
        if abs(new_p_th - p_th) < 1e-7:
            break
        p_th = new_p_th
    log_a, log_pth = coef
    p_th = math.exp(log_pth)
    return {"A": math.exp(log_a), "p_th": p_th, "p_th_stderr": p_th * math.sqrt(cov[1, 1]),
            "A_stderr": math.exp(log_a) * math.sqrt(cov[0, 0]),
            "chi2_per_dof": chi2, "n_points_used": len(pts),
            "p_max_used_for_fit": max(r["p"] for r in pts), "fit_min_d": FIT_MIN_D}


def main(names, refit=False):
    for name in names:
        noise_cls, grid = GRIDS[name]
        path = os.path.join(OUT_DIR, f"threshold_fit_{name}.json")
        print(f"\n== {name} ==", flush=True)
        if refit:
            with open(path) as fh:
                old = json.load(fh)
            points, source = old["points"], old["source"]
        else:
            points = []
            for d in DISTANCES:
                for p in grid:
                    t = time.time()
                    m = measure(d, noise_cls(p=p))
                    points.append({"d": d, "p": p, **m})
                    print(f"  d={d:2d} p={p:.5f} p_L={m['p_L']:.3e} "
                          f"({m['fails']}/{m['shots']:,}, {time.time() - t:.0f}s)", flush=True)
            source = (f"Deltakit {deltakit.__version__}: RotatedPlanarCode Z memory, rounds=d, "
                      f"{noise_cls.__name__}, "
                      f"{'all gates native (CX)' if name in NATIVE_CX else 'default native gates (CZ)'}, "
                      f"PyMatchingDecoder")
        f = fit(points)
        f.update({
            "noise_model": name,
            "source": source,
            "distances": list(DISTANCES), "p_grid": list(grid),
            "note": SEED_NOTE, "points": points,
        })
        with open(path, "w") as fh:
            json.dump(f, fh, indent=1)
        print(f"  fit: A={f['A']:.4f}  p_th={f['p_th'] * 100:.3f}% +/- "
              f"{f['p_th_stderr'] * 100:.3f}  chi2/dof={f['chi2_per_dof']:.2f}  "
              f"({f['n_points_used']} points)  -> {os.path.normpath(path)}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--refit"]
    main(args or list(GRIDS), refit="--refit" in sys.argv)
