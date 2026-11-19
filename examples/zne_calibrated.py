"""ZNE on a real device calibration: when does mitigation actually work?

examples/qregime_to_mitiq.py ran ZNE under pure depolarizing noise, the best
case for zero-noise extrapolation: there, exponential ZNE recovered the ideal
value far past qregime's QEM boundary. This example runs the same mirror
circuits through qregime.mitigation.simulate_zne, which uses the FakeTorino
calibration (per-qubit CZ and one-qubit errors, T1/T2 relaxation during
gates and idle times, readout errors) and a noise-aware layout.

For each depth it records the unmitigated success fraction, exponential and
Richardson ZNE with and without readout-error mitigation, their bias against
the ideal value 1, and the shots needed for +/-0.01 precision.

    pip install -e ".[realistic]"
    python examples/zne_calibrated.py      # writes docs/zne_calibrated.png and .json (~5 min)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile

import qregime
from qregime import safe_qasm2_dumps
from qregime.mitigation import simulate_zne

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HARDWARE = "superconductor_ibm"
LAYERS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
SEED = 7

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots 1-4, validated; slots 3-4 are below 3:1 contrast, so every
# series also has its own marker and the numbers are printed and saved
SERIES = {
    "unmitigated": ("#2a78d6", "o", "unmitigated"),
    "exp_rem": ("#eb6834", "s", "ZNE exponential + readout mitigation"),
    "rich_rem": ("#1baf7a", "D", "ZNE Richardson + readout mitigation"),
    "exp_raw": ("#eda100", "^", "ZNE exponential, no readout mitigation"),
}
QEM_BAND, NOT_QEM_BAND = "#eef4fc", "#fdf0ea"
N_QUBITS = 6


def mirror(layers: int, rng: np.random.Generator) -> QuantumCircuit:
    """Same construction as examples/qregime_to_mitiq.py (random angles differ)."""
    u = QuantumCircuit(N_QUBITS)
    for layer in range(layers):
        for q in range(N_QUBITS):
            u.ry(float(rng.uniform(0.1, 3.0)), q)
        for q in range(layer % 2, N_QUBITS - 1, 2):
            u.cx(q, q + 1)
    return transpile(u.compose(u.inverse()), basis_gates=["u", "cx"], optimization_level=0)


def run() -> dict:
    with open(os.path.join(ROOT, "docs", "qregime_to_mitiq.json")) as f:
        depolarizing = {r["layers"]: r for r in json.load(f)["rows"]}
    rng = np.random.default_rng(SEED)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for layers in LAYERS:
            qc = mirror(layers, rng)
            path = os.path.join(tmp, f"mirror_{layers}.qasm")
            with open(path, "w") as f:
                f.write(safe_qasm2_dumps(qc))
            decision = qregime.decide_regime(path, hardware=HARDWARE, shots=2000,
                                             simulate_mitigation=True)
            sim = decision["qem"]["mitigation"]["simulated"]
            raw, rem = sim["zne"], sim["zne_with_readout_mitigation"]
            dep = depolarizing.get(layers)
            row = {
                "layers": layers, "cx": qc.count_ops().get("cx", 0),
                "qregime_regime": decision["regime"],
                "qregime_qem_reliability": decision["qem"]["reliability"],
                "analytic_zne_shots": decision["qem"]["mitigation"]["total_shots"],
                "physical_qubits": sim["physical_qubits"],
                "scale_factors": sim["scale_factors"],
                "unmitigated": sim["unmitigated"],
                "unmitigated_after_readout_mitigation": rem["values"][0],
                "exp_raw": raw["exponential"], "rich_raw": raw["richardson"],
                "exp_rem": rem["exponential"], "rich_rem": rem["richardson"],
                "exp_rem_shots": rem["exponential_total_shots"],
                "rich_rem_shots": rem["richardson_total_shots"],
                "best_method": sim["best_method"], "best_abs_bias": sim["best_abs_bias"],
                "viable": sim["viable"],
                "depolarizing_exp_zne_mean": dep["zne"]["exponential"]["mean"] if dep else None,
            }
            rows.append(row)
            fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
            print(f"layers={layers:3d} cx={row['cx']:4d} {row['qregime_regime']:4s} "
                  f"raw {fmt(row['unmitigated'])}  exp {fmt(row['exp_raw'])}  "
                  f"exp+REM {fmt(row['exp_rem'])}  rich+REM {fmt(row['rich_rem'])}  "
                  f"best bias {row['best_abs_bias']:.3f} viable={row['viable']}  "
                  f"shots exp+REM {row['exp_rem_shots']}", flush=True)
    return {"hardware": HARDWARE, "device_model": sim["device_model"], "precision": sim["precision"],
            "not_modelled": sim["not_modelled"], "rows": rows}


def plot(data: dict, path: str):
    rows = data["rows"]
    x = np.array([r["cx"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = np.log(x)
    edges = np.exp(np.concatenate([[xs[0] - 0.25], (xs[1:] + xs[:-1]) / 2, [xs[-1] + 0.25]]))
    for r, lo, hi in zip(rows, edges[:-1], edges[1:]):
        ax.axvspan(lo, hi, color=QEM_BAND if r["qregime_regime"] == "QEM" else NOT_QEM_BAND,
                   lw=0, zorder=0)
    for y in (0, 0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color=GRID, lw=1, zorder=1)
    ax.axhspan(1 - data["precision"], 1 + data["precision"], color=INK_2, alpha=0.12, lw=0, zorder=1)
    ax.text(x[0], 1.035, f"ideal ± {data['precision']:g}", color=INK_2, fontsize=8)
    dep = [(r["cx"], r["depolarizing_exp_zne_mean"]) for r in rows if r["depolarizing_exp_zne_mean"]]
    if dep:
        ax.plot(*zip(*dep), color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=2,
                label="ZNE exponential, depolarizing model (previous example)")
    for key, (color, marker, label) in SERIES.items():
        pts = [(r["cx"], r[key]) for r in rows if r[key] is not None]
        ax.plot(*zip(*pts), color=color, lw=2, marker=marker, ms=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3, label=label)
    # exponential fits whose shot cost exceeds any realistic budget: the
    # values sit near the asymptote and the extrapolation is unstable
    for r in rows:
        if r["exp_rem_shots"] and r["exp_rem_shots"] > 1e8 and r["exp_rem"] > 0.5:
            ax.annotate(f"unstable fit:\n{r['exp_rem_shots']:.1e} shots", (r["cx"], r["exp_rem"]),
                        xytext=(-58, -38), textcoords="offset points", fontsize=7.5, color=INK_2,
                        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    first_out = next((r for r in rows if r["qregime_regime"] != "QEM"), None)
    ax.text(x[0], 1.13, "qregime: QEM", color=INK_2, fontsize=8)
    if first_out:
        ax.text(first_out["cx"], 1.13, f"qregime: {first_out['qregime_regime']}", color=INK_2,
                fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in x])
    ax.set_ylim(-0.03, 1.2)
    ax.set_xlabel("CX gates in the mirror circuit (log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("success probability (ideal = 1)", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.minorticks_off()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, labelcolor=INK_2)
    ax.set_title("ZNE on the FakeTorino calibration: bias, not shots, sets the limit",
                 loc="left", color=INK, fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    print(f"wrote {path}")


def main():
    out = os.path.join(ROOT, "docs", "zne_calibrated.json")
    if "--plot-only" in sys.argv:
        with open(out) as f:
            data = json.load(f)
    else:
        data = run()
        with open(out, "w") as f:
            json.dump(data, f, indent=1)
    plot(data, os.path.join(ROOT, "docs", "zne_calibrated.png"))


if __name__ == "__main__":
    main()
