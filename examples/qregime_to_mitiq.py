"""qregime decides, Mitiq mitigates: how does qregime's QEM boundary compare with ZNE?

qregime answers the question that comes before error mitigation: should this
circuit be mitigated at all, or does it need error correction? Mitiq answers
the next one: how to mitigate it. This example hands circuits from one to the
other and checks whether qregime's QEM boundary matches where Mitiq's
zero-noise extrapolation (ZNE) actually stops giving usable answers.

Circuits: 6-qubit layered circuits (RY rotations + CX brickwork) of growing
depth, run as mirrors U U^dagger, so the ideal output is |000000> and the
ideal success probability is exactly 1.

For each depth:
  - qregime.decide_regime on the IBM heavy-hex profile (p_2q = 0.32%):
    regime, and the unmitigated reliability from its noisy simulation;
  - Mitiq ZNE (scale factors 1, 2, 3, random gate folding), with two
    extrapolations: Richardson (Mitiq's common default) and exponential
    (ExpFactory, asymptote 1/2^n), which matches the exponential decay of
    pure depolarizing noise and is the fair best case for ZNE here. Same
    depolarizing noise, with a FINITE shot budget per
    executed circuit. With infinite shots ZNE always looks good; with real
    shots its statistical error grows as the raw signal shrinks. Each point
    is repeated to measure that spread.

    pip install -e ".[compare]"
    python examples/qregime_to_mitiq.py      # writes docs/qregime_to_mitiq.png and .json
"""
from __future__ import annotations

import json
import os
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mitiq import zne
from mitiq.zne.inference import ExpFactory, RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import qregime
from qregime import safe_qasm2_dumps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARDWARE = "superconductor_ibm"
N_QUBITS = 6
LAYERS = (1, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128)
SHOTS = 10_000          # per executed circuit (ZNE runs 3 circuits)
REPEATS = 20            # ZNE repetitions per depth, to measure the spread
QREGIME_SHOTS = 2000    # shots for qregime's own noisy simulation
SEED = 7

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots 1-3, validated; slot 3 is below 3:1 contrast, so every
# series also has its own marker and the numbers are printed and saved
RAW_COLOR = "#2a78d6"
ZNE_STYLE = {"richardson": ("#eb6834", "s", "Mitiq ZNE, Richardson"),
             "exponential": ("#1baf7a", "D", "Mitiq ZNE, exponential")}
QEM_BAND, NOT_QEM_BAND = "#eef4fc", "#fdf0ea"


def layered_mirror(layers: int, rng: np.random.Generator) -> QuantumCircuit:
    u = QuantumCircuit(N_QUBITS)
    for layer in range(layers):
        for q in range(N_QUBITS):
            u.ry(float(rng.uniform(0.1, 3.0)), q)
        for q in range(layer % 2, N_QUBITS - 1, 2):
            u.cx(q, q + 1)
    mirror = u.compose(u.inverse())
    return transpile(mirror, basis_gates=["u", "cx"], optimization_level=0)


def make_executor(p_2q: float, rng: np.random.Generator):
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=noise)

    def execute(circ: QuantumCircuit) -> float:
        """P(000000), estimated from SHOTS samples of the exact distribution."""
        c = circ.copy()
        c.save_probabilities()
        p0 = float(sim.run(c).result().data()["probabilities"][0])
        return rng.binomial(SHOTS, min(max(p0, 0.0), 1.0)) / SHOTS

    def exact(circ: QuantumCircuit) -> float:
        c = circ.copy()
        c.save_probabilities()
        return float(sim.run(c).result().data()["probabilities"][0])

    return execute, exact


def run() -> dict:
    rng = np.random.default_rng(SEED)
    hw = qregime.HARDWARE_PROFILES[HARDWARE]
    execute, exact = make_executor(hw.p_2q, rng)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for layers in LAYERS:
            qc = layered_mirror(layers, rng)
            path = os.path.join(tmp, f"mirror_{layers}.qasm")
            with open(path, "w") as f:
                f.write(safe_qasm2_dumps(qc))
            # same depolarizing model as the executor below, for a like-for-like check
            decision = qregime.decide_regime(path, hardware=HARDWARE, shots=QREGIME_SHOTS,
                                             use_calibration=False)
            factories = {"richardson": lambda: RichardsonFactory([1, 2, 3]),
                         "exponential": lambda: ExpFactory([1, 2, 3], asymptote=2 ** -N_QUBITS)}
            zne_rows = {}
            for name, make in factories.items():
                vals = np.array([zne.execute_with_zne(qc, execute, factory=make(),
                                                      scale_noise=fold_gates_at_random)
                                 for _ in range(REPEATS)])
                zne_rows[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                  "rms_error": float(np.sqrt(np.mean((vals - 1.0) ** 2)))}
            row = {
                "layers": layers,
                "cx": qc.count_ops().get("cx", 0),
                "qregime_regime": decision["regime"],
                "qregime_qem_reliability": decision["qem"]["reliability"],
                "qregime_qem_source": decision["qem"]["reliability_source"],
                "qregime_qem_meets_target": decision["qem"]["meets_target"],
                "raw_exact": exact(qc),
                "zne": zne_rows,
                "raw_error": 1.0 - exact(qc),
            }
            rows.append(row)
            print(f"layers={layers:3d} cx={row['cx']:4d}  qregime: {row['qregime_regime']:12s} "
                  f"reliability {row['qregime_qem_reliability']:.3f}   raw {row['raw_exact']:.3f}   "
                  + "   ".join(f"{k[:4]} {v['mean']:.3f}+/-{v['std']:.3f} (rms {v['rms_error']:.3f})"
                               for k, v in zne_rows.items()), flush=True)
    return {"hardware": HARDWARE, "p_2q": hw.p_2q, "n_qubits": N_QUBITS, "shots": SHOTS,
            "repeats": REPEATS, "reliability_target": 0.5, "rows": rows}


def plot(data: dict, path: str):
    rows = data["rows"]
    x = np.array([r["cx"] for r in rows])
    raw = np.array([r["raw_exact"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    # regime background: where qregime keeps QEM vs. where it does not
    edges = np.concatenate([[x[0] - 10], (x[1:] + x[:-1]) / 2, [x[-1] + 10]])
    for r, lo, hi in zip(rows, edges[:-1], edges[1:]):
        ax.axvspan(lo, hi, color=QEM_BAND if r["qregime_regime"] == "QEM" else NOT_QEM_BAND,
                   lw=0, zorder=0)
    for y in (0, 0.5, 1.0, 1.5):
        ax.axhline(y, color=GRID, lw=1, zorder=1)
    ax.axhline(1.0, color=INK_2, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(x[-1], 1.03, "ideal", color=INK_2, fontsize=8, ha="right")
    for name, (color, marker, label) in ZNE_STYLE.items():
        mean = np.array([r["zne"][name]["mean"] for r in rows])
        std = np.array([r["zne"][name]["std"] for r in rows])
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, mean, color=color, lw=2, marker=marker, ms=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3, label=f"{label} (±1 sd)")
    ax.plot(x, raw, color=RAW_COLOR, lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3, label="unmitigated")
    first_out = next((r for r in rows if r["qregime_regime"] != "QEM"), None)
    if first_out:
        ax.text(first_out["cx"], 1.42, f"qregime: {first_out['qregime_regime']}", color=INK_2,
                fontsize=8)
    ax.text(x[0], 1.42, "qregime: QEM", color=INK_2, fontsize=8)
    ax.set_ylim(-0.1, 1.55)
    ax.set_xlabel("CX gates in the mirror circuit", color=INK_2, fontsize=9)
    ax.set_ylabel("success probability P(000000)", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK_2,
              title=f"{data['shots']:,} shots per executed circuit, {data['repeats']} repeats",
              title_fontsize=8)
    ax.set_title(f"qregime's QEM boundary vs. what Mitiq ZNE recovers "
                 f"(depolarizing p_2q = {data['p_2q']:.2%})", loc="left", color=INK, fontsize=10,
                 pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    print(f"wrote {path}")


def main():
    data = run()
    with open(os.path.join(ROOT, "docs", "qregime_to_mitiq.json"), "w") as f:
        json.dump(data, f, indent=1)
    plot(data, os.path.join(ROOT, "docs", "qregime_to_mitiq.png"))


if __name__ == "__main__":
    main()
