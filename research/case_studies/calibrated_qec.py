"""Are qregime's device calibrations realistic enough for QEC? Test against measured Lambda.

Surface-code memories are built with Deltakit, placed on the device's real
qubits with qregime's noise-aware layout, and given circuit-level noise
qubit by qubit from a qregime calibration. Decoding is MWPM (PyMatching).
The resulting Lambda is compared with the one measured on hardware:

  - Google Willow: Lambda_{3,5,7} = 2.14 +/- 0.02 (Willow spec sheet, Dec
    2024; neural-network decoder), with the `google_willow` calibration
    (Google's 2024-08-16 snapshot bundled with cirq-google; we do not know
    whether it is the chip used for that experiment);
  - IBM Nighthawk: Lambda = 1.68 [1.57, 1.81] on ibm_phoenix
    (arXiv:2610.23840, exact ML decoding, d = 3 and 6 with the other
    distance 3), with the `ibm_miami` calibration (another 120-qubit
    Nighthawk, 2026-04-17 snapshot; ibm_phoenix has no public snapshot).

For each device three noise models are compared:

  median     every qubit and coupler at the calibration's median (what a
             "median calibration" model does, as in ibm_nighthawk_2026.py);
  per-qubit  each physical qubit and coupler with its own errors, T1/T2
             and readout (the qregime calibration as is);
  coherent   (Willow only) per-qubit, plus Google's coherent fSim error per
             coupler, Pauli-twirled: its entanglement infidelity is added
             to the CZ depolarizing probability.

Noise, per TICK layer: depolarizing after each gate matched to the average
gate error (1q: 3/2 e, 2q: 5/4 e); a flip with the qubit's mean readout
error before each measurement and after each reset; Pauli-twirled T1/T2
noise for the time a qubit waits in the layer. Not modelled: leakage,
crosstalk, correlated errors, drift, dynamical decoupling.

    pip install "deltakit>=0.10" stim
    python research/case_studies/calibrated_qec.py      # writes calibrated_qec_results.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

# Deltakit must be imported before pymatching: the other order breaks
# Deltakit's code construction (two Stim builds in one process)
from deltakit.circuit.gates import CZ, H, MZ, RZ, S, X, Y, Z, PauliBasis
from deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit.explorer.qpu import QPU, NativeGateSetAndTimes, PhysicalNoise

import pymatching  # noqa: E402
import stim  # noqa: E402

from qregime import calibration as K

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS = (3, 5, 9, 13)
MIN_FAILS = 300
MAX_SHOTS = 2_000_000

DEVICES = {
    "google_willow": {
        "measured": {"lambda": 2.14, "lambda_err": 0.02, "decoder": "neural network",
                     "source": "Willow spec sheet (Dec 9, 2024): Lambda_{3,5,7} = 2.14 +/- 0.02"},
        # device specification bundled with cirq-google (picoseconds there)
        "measure_s": 600e-9, "reset_s": 160e-9,
        "patches": [(3, 3), (5, 5), (7, 7)],
        "variants": ("median", "per-qubit", "coherent"),
    },
    "ibm_miami": {
        "measured": {"lambda": 1.68, "lambda_ci": [1.57, 1.81], "decoder": "exact ML",
                     "device": "ibm_phoenix",
                     "source": "arXiv:2610.23840, Z memory, d = 3 and 6, other distance 3"},
        # FakeMiami target: measure 2.4 us; reset fused into the measurement
        "measure_s": 2.4e-6, "reset_s": 0.0,
        "patches": [(3, 3), (6, 3)],
        "variants": ("median", "per-qubit"),
    },
}


def _native_circuit(width: int, height: int, rounds: int) -> stim.Circuit:
    """Z-basis memory compiled by Deltakit to H, CZ, M, R, with its noise stripped."""
    circ = css_code_memory_circuit(RotatedPlanarCode(width=width, height=height),
                                   num_rounds=rounds, logical_basis=PauliBasis.Z)
    gates = NativeGateSetAndTimes(one_qubit_gates={g: 1e-8 for g in (H, X, Y, Z, S)},
                                  two_qubit_gates={CZ: 1e-8}, reset_gates={RZ: 1e-8},
                                  measurement_gates={MZ: 1e-8})
    zero = PhysicalNoise(t1=1e9, t2=1e9, p_1_qubit_gate_error=0, p_2_qubit_gate_error=0,
                         p_reset_error=0, p_meas_qubit_error=0, p_readout_flip=0)
    compiled = QPU(circ.qubits, native_gates_and_times=gates, noise_model=zero)
    text = str(compiled.compile_and_add_noise_to_circuit(circ).as_stim_circuit())
    flat = stim.Circuit(text).flattened()
    clean = stim.Circuit()
    for inst in flat:
        if inst.name not in NOISE_OPS:
            clean.append(inst)
    return clean


NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
             "X_ERROR", "Y_ERROR", "Z_ERROR"}


def _layers(circuit: stim.Circuit):
    layer = []
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            raise ValueError("unrolled circuits expected")
        if inst.name == "TICK":
            yield layer
            layer = []
        else:
            layer.append(inst)
    yield layer


class Noise:
    """Per-physical-qubit noise parameters from a calibration (or its medians)."""

    def __init__(self, cal: K.Calibration, variant: str):
        self.cal, self.variant = cal, variant
        med = lambda v: float(np.median([x for x in v if math.isfinite(x)]))  # noqa: E731
        self.median = {
            "sq": med(cal.sq_error), "ro": med([(a + b) / 2 for a, b in zip(cal.p01, cal.p10)]),
            "t1": med(cal.t1), "t2": med(cal.t2), "sq_t": med(cal.sq_duration),
            "tq": med([cal.two_qubit(a, b)[0] for a, b in cal.edges]),
            "tq_t": med([cal.two_qubit(a, b)[1] for a, b in cal.edges]),
        }

    def _get(self, key, per_qubit):
        return self.median[key] if self.variant == "median" else per_qubit

    def sq(self, p):
        return self._get("sq", self.cal.sq_error[p]), self._get("sq_t", self.cal.sq_duration[p])

    def tq(self, a, b):
        error, duration = self.cal.two_qubit(a, b)
        if self.variant == "median":
            return self.median["tq"], self.median["tq_t"]
        if self.variant == "coherent":
            u = self.cal.coherent_2q.get((min(a, b), max(a, b)))
            if u is not None:
                # Pauli-twirled coherent error: its process infidelity, as a
                # two-qubit Pauli probability on top of 5/4 e
                return error + 0.8 * (1 - abs(np.trace(u)) ** 2 / 16), duration
        return error, duration

    def ro(self, p):
        return self._get("ro", (self.cal.p01[p] + self.cal.p10[p]) / 2)

    def t1t2(self, p):
        t1, t2 = self._get("t1", self.cal.t1[p]), self._get("t2", self.cal.t2[p])
        return t1, min(t2, 2 * t1)


def noisy_circuit(native: stim.Circuit, phys: list, noise: Noise, measure_s: float,
                  reset_s: float) -> stim.Circuit:
    n = native.num_qubits
    out = stim.Circuit()
    for layer in _layers(native):
        busy, pre, post = {}, [], []
        for inst in layer:
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if inst.name in ("H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG"):
                for q in targets:
                    e, t = noise.sq(phys[q])
                    busy[q] = t
                    post.append(("DEPOLARIZE1", [q], 1.5 * e))
            elif inst.name == "CZ":
                for a, b in zip(targets[::2], targets[1::2]):
                    e, t = noise.tq(phys[a], phys[b])
                    busy[a] = busy[b] = t
                    post.append(("DEPOLARIZE2", [a, b], min(15 / 16, 1.25 * e)))
            elif inst.name == "M":
                for q in targets:
                    busy[q] = measure_s
                    pre.append(("X_ERROR", [q], noise.ro(phys[q])))
            elif inst.name == "R":
                for q in targets:
                    busy[q] = reset_s
                    post.append(("X_ERROR", [q], noise.ro(phys[q])))
            elif inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS",
                                   "QUBIT_COORDS", "I"):
                raise ValueError(f"unexpected instruction {inst.name}")
        for name, qs, p in pre:
            out.append(name, qs, p)
        for inst in layer:
            out.append(inst)
        for name, qs, p in post:
            out.append(name, qs, p)
        duration = max(busy.values(), default=0.0)
        if duration > 0:
            for q in range(n):
                idle = duration - busy.get(q, 0.0)
                if idle <= 0:
                    continue
                t1, t2 = noise.t1t2(phys[q])
                px = (1 - math.exp(-idle / t1)) / 4
                pz = max(0.0, (1 - math.exp(-idle / t2)) / 2 - px)
                out.append("PAULI_CHANNEL_1", [q], [px, px, pz])
        out.append("TICK")
    return out


def layout_for(native: stim.Circuit, cal: K.Calibration) -> list:
    pairs = Counter()
    for inst in native:
        if inst.name == "CZ":
            ts = [t.value for t in inst.targets_copy()]
            for a, b in zip(ts[::2], ts[1::2]):
                pairs[(min(a, b), max(a, b))] += 1
    layout, needs_routing = K._choose_layout(cal, native.num_qubits, dict(pairs))
    if needs_routing:
        raise RuntimeError(f"the code does not embed in {cal.name}'s coupling graph")
    return layout


def logical_error(circuit: stim.Circuit) -> float:
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    shots = fails = 0
    while fails < MIN_FAILS and shots < MAX_SHOTS:
        batch = 100_000
        det, obs = sampler.sample(batch, separate_observables=True)
        fails += int(np.sum(np.any(matcher.decode_batch(det) != obs, axis=1)))
        shots += batch
    return fails / shots, shots, fails


def error_per_round(points: dict) -> float:
    """Fit ln(1 - 2 P_L(r)) = a + r ln(1 - 2 eps)."""
    rs = np.array(sorted(points), float)
    y = np.log(np.array([1 - 2 * points[r] for r in sorted(points)]))
    return (1 - math.exp(np.polyfit(rs, y, 1)[0])) / 2


def lam(eps: dict) -> float:
    """Lambda from eps_d ~ A / Lambda**((d + 1) / 2), fitted over the distances."""
    ds = np.array(sorted(eps), float)
    slope = np.polyfit((ds + 1) / 2, np.log([eps[d] for d in sorted(eps)]), 1)[0]
    return float(math.exp(-slope))


def main(only=None):
    t0 = time.time()
    out = {"rounds": ROUNDS, "decoder": "PyMatching (MWPM)", "devices": {}}
    for name, spec in DEVICES.items():
        if only and name != only:
            continue
        cal = K.load(name)
        dev = {"measured": spec["measured"], "patches": {}, "lambda": {}}
        layouts = {}
        for width, height in spec["patches"]:
            native = _native_circuit(width, height, ROUNDS[0])
            layouts[(width, height)] = layout_for(native, cal)
        for variant in spec["variants"]:
            noise = Noise(cal, variant)
            eps = {}
            for width, height in spec["patches"]:
                pts, raw = {}, {}
                for r in ROUNDS:
                    native = _native_circuit(width, height, r)
                    circ = noisy_circuit(native, layouts[(width, height)], noise,
                                         spec["measure_s"], spec["reset_s"])
                    p, shots, fails = logical_error(circ)
                    pts[r], raw[r] = p, {"p_L": p, "shots": shots, "fails": fails}
                d = width
                eps[d] = error_per_round(pts)
                dev["patches"].setdefault(variant, {})[f"{width}x{height}"] = {
                    "eps_per_round": eps[d], "raw": raw,
                    "physical_qubits": layouts[(width, height)]}
                print(f"{name:14s} {variant:9s} {width}x{height}  eps/round {eps[d]:.4%}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
            dev["lambda"][variant] = lam(eps)
            print(f"{name:14s} {variant:9s} Lambda {dev['lambda'][variant]:.2f}  "
                  f"(measured {spec['measured']['lambda']})", flush=True)
        out["devices"][name] = dev
    path = os.path.join(HERE, "calibrated_qec_results.json" if not only
                        else f"calibrated_qec_{only}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
