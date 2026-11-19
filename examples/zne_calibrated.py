"""ZNE on real device calibrations: when does mitigation actually work?

examples/qregime_to_mitiq.py ran ZNE under pure depolarizing noise, the best
case for zero-noise extrapolation: there, exponential ZNE recovered the ideal
value far past qregime's QEM boundary. This example runs the same kind of
mirror circuits through qregime's calibrated simulator on two devices:

  - ibm_torino: IBM Torino (FakeTorino snapshot): per-qubit CZ and
    single-qubit errors, T1/T2 during gates and idle times, readout errors,
    heavy-hex coupling;
  - quantinuum_h2_2: Quantinuum H2-2 published parameters: all-to-all ZZ
    gates, memory error per layer, asymmetric readout.

For each depth it records qregime's decision (with and without the
calibration), the unmitigated success fraction, exponential and Richardson
ZNE with and without readout-error mitigation, their bias against the ideal
value 1, and the shots needed for +/-0.01 precision. No Qiskit needed.

    python examples/zne_calibrated.py      # writes docs/zne_calibrated.png and .json (~1 min)
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

import qregime
import qregime.mitigation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEVICES = {"superconductor_ibm": "IBM Torino (heavy-hex, CZ)",
           "trapped_ion_quantinuum_h2": "Quantinuum H2-2 (all-to-all, ZZ)"}
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


def mirror(layers: int, rng: np.random.Generator) -> tuple:
    """OpenQASM text of U U^dagger (RY layers + CX brickwork), same
    construction as examples/qregime_to_mitiq.py, and its CX count."""
    ops = []
    for layer in range(layers):
        ops += [("ry", float(rng.uniform(0.1, 3.0)), q) for q in range(N_QUBITS)]
        ops += [("cx", None, q) for q in range(layer % 2, N_QUBITS - 1, 2)]
    inverse = [(name, -angle if angle is not None else None, q) for name, angle, q in reversed(ops)]
    lines = [f"ry({angle!r}) q[{q}];" if name == "ry" else f"cx q[{q}],q[{q + 1}];"
             for name, angle, q in ops + inverse]
    text = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + f"qreg q[{N_QUBITS}];\n" + "\n".join(lines) + "\n"
    return text, 2 * sum(1 for name, _, _ in ops if name == "cx")


def run_device(hardware: str, circuits: list, depolarizing: dict) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for layers, (text, cx) in zip(LAYERS, circuits):
            path = os.path.join(tmp, f"mirror_{layers}.qasm")
            with open(path, "w") as f:
                f.write(text)
            decision = qregime.decide_regime(path, hardware=hardware, shots=2000,
                                             simulate_mitigation=True)
            sim = decision["qem"]["mitigation"]["simulated"]
            raw, rem = sim["zne"], sim["zne_with_readout_mitigation"]
            uncalibrated = qregime.decide_regime(path, hardware=hardware, use_calibration=False)
            dep = depolarizing.get(layers) if hardware == "superconductor_ibm" else None
            row = {
                "layers": layers, "cx": cx,
                "qregime_regime": decision["regime"],
                "qregime_qem_reliability": decision["qem"]["reliability"],
                "qregime_qem_source": decision["qem"]["reliability_source"],
                "qregime_regime_depolarizing": uncalibrated["regime"],
                "qregime_qem_reliability_depolarizing": uncalibrated["qem"]["reliability"],
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
            fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"{hardware:26s} cx={cx:4d} qregime {row['qregime_regime']:4s} "
                  f"(depolarizing: {row['qregime_regime_depolarizing']}, "
                  f"{row['qregime_qem_reliability_depolarizing']:.3f})  "
                  f"raw {fmt(row['unmitigated'])}  exp {fmt(row['exp_raw'])}  "
                  f"exp+REM {fmt(row['exp_rem'])}  rich+REM {fmt(row['rich_rem'])}  "
                  f"bias {row['best_abs_bias']:.3f} viable={row['viable']}  "
                  f"shots {row['exp_rem_shots']}", flush=True)
    return {"device_model": sim["device_model"], "label": DEVICES[hardware], "rows": rows}


def run() -> dict:
    with open(os.path.join(ROOT, "docs", "qregime_to_mitiq.json")) as f:
        depolarizing = {r["layers"]: r for r in json.load(f)["rows"]}
    rng = np.random.default_rng(SEED)
    circuits = [mirror(layers, rng) for layers in LAYERS]
    devices = {hw: run_device(hw, circuits, depolarizing) for hw in DEVICES}
    return {"precision": qregime.mitigation.DEFAULT_PRECISION,
            "not_modelled": "leakage, crosstalk, drift", "devices": devices}


def _panel(ax, device: dict, precision: float, show_legend: bool):
    rows = device["rows"]
    x = np.array([r["cx"] for r in rows], dtype=float)
    ax.set_facecolor(SURFACE)
    xs = np.log(x)
    edges = np.exp(np.concatenate([[xs[0] - 0.25], (xs[1:] + xs[:-1]) / 2, [xs[-1] + 0.25]]))
    for r, lo, hi in zip(rows, edges[:-1], edges[1:]):
        ax.axvspan(lo, hi, color=QEM_BAND if r["qregime_regime"] == "QEM" else NOT_QEM_BAND,
                   lw=0, zorder=0)
    for y in (0, 0.25, 0.5, 0.75, 1.0):
        ax.axhline(y, color=GRID, lw=1, zorder=1)
    ax.axhspan(1 - precision, 1 + precision, color=INK_2, alpha=0.12, lw=0, zorder=1)
    dep = [(r["cx"], r["depolarizing_exp_zne_mean"]) for r in rows if r["depolarizing_exp_zne_mean"]]
    if dep:
        ax.plot(*zip(*dep), color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=2,
                label="ZNE exponential, depolarizing model")
    for key, (color, marker, label) in SERIES.items():
        pts = [(r["cx"], r[key]) for r in rows if r[key] is not None]
        ax.plot(*zip(*pts), color=color, lw=2, marker=marker, ms=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.3, zorder=3, label=label)
    for r in rows:
        if r["exp_rem_shots"] and r["exp_rem_shots"] > 1e8 and r["exp_rem"] > 0.5:
            ax.annotate(f"unstable fit:\n{r['exp_rem_shots']:.0e} shots", (r["cx"], r["exp_rem"]),
                        xytext=(-62, -30), textcoords="offset points", fontsize=7, color=INK_2,
                        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    first_out = next((r for r in rows if r["qregime_regime"] != "QEM"), None)
    ax.text(x[0], 1.12, "qregime: QEM", color=INK_2, fontsize=7.5)
    if first_out:
        ax.text(first_out["cx"], 1.12, f"qregime: {first_out['qregime_regime']}", color=INK_2,
                fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks(x[::2])
    ax.set_xticklabels([f"{int(v)}" for v in x[::2]])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.2)
    ax.set_xlabel("CX gates in the mirror circuit (log scale)", color=INK_2, fontsize=8.5)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(device["label"], loc="left", color=INK, fontsize=9.5, pad=8)
    if show_legend:
        ax.legend(loc="lower left", fontsize=7, frameon=False, labelcolor=INK_2)


def plot(data: dict, path: str):
    devices = list(data["devices"].values())
    fig, axes = plt.subplots(1, len(devices), figsize=(11, 4.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for k, (ax, device) in enumerate(zip(axes, devices)):
        _panel(ax, device, data["precision"], show_legend=k == 0)
    axes[0].set_ylabel("success probability (ideal = 1)", color=INK_2, fontsize=8.5)
    fig.suptitle("ZNE on device calibrations: bias and readout, not only shots, set the limit "
                 f"(band: ideal ± {data['precision']:g})", x=0.01, ha="left", color=INK,
                 fontsize=10.5)
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
