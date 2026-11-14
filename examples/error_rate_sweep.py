"""What two-qubit error rate does a superconducting device need before
error correction helps the flagship circuit?

Sweeps p_2q under the superconducting-inspired SI1000 noise fit (p_th ~0.40%)
for the paper's flagship circuit (16 logical qubits, T-count 3,930) and plots
the physical qubits needed for full fault tolerance and for partial QEC at a
50% reliability target. At today's p_2q = 0.32% neither exists for any code
distance up to 51.

    python examples/error_rate_sweep.py            # writes docs/error_rate_sweep.png
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qregime
from qregime.partial_qec import sweep_partial_qec
from qregime.resources import _load_fit

N_LOGICAL, TCOUNT, NOISE, TODAY = 16, 3930, "si1000", 0.0032
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FTQC_COLOR, PQEC_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1-2, validated


def sweep(ps):
    ftqc, pqec = [], []
    for p in ps:
        e = qregime.estimate_ftqc(N_LOGICAL, TCOUNT, p, noise_model=NOISE)
        r = sweep_partial_qec(N_LOGICAL, TCOUNT, p, 0.5, 10**9, verbose=False,
                              noise_model=NOISE)
        ftqc.append(e["physicalQubits"] if e["feasible"] else np.nan)
        pqec.append(r["cost_qubits"] if r.get("feasible") else np.nan)
    return np.array(ftqc, float), np.array(pqec, float)


def plot(path):
    ps = np.round(np.arange(0.0001, 0.00351, 0.00005), 6)
    ftqc, pqec = sweep(ps)
    p_th = _load_fit(NOISE)["p_th"]
    x = ps * 100

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_yscale("log")
    for y in (1e3, 1e4, 1e5):
        ax.axhline(y, color=GRID, lw=1, zorder=0)
    ax.step(x, ftqc, where="mid", color=FTQC_COLOR, lw=2, zorder=3, label="full fault tolerance")
    ax.step(x, pqec, where="mid", color=PQEC_COLOR, lw=2, zorder=3,
            label="partial QEC, 50% reliability")

    ax.axhline(1121, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.text(0.415, 1121 * 1.12, "1,121 qubits (IBM Condor)", color=INK_2, fontsize=8, ha="right")
    for xv, label in ((TODAY * 100, "today\n0.32%"), (p_th * 100, f"threshold\n{p_th * 100:.2f}%")):
        ax.axvline(xv, color=INK_2, lw=1, ls=(0, (3, 3)))
        ax.text(xv - 0.005, 420, label, color=INK_2, fontsize=8, ha="right", va="bottom")

    # direct labels mid-series: FTQC above its line, partial QEC below its line
    ax.text(0.085, 2.6e4, "full fault tolerance", color=INK, fontsize=8, ha="right")
    ax.text(0.16, 6.5e3, "partial QEC (50%)", color=INK, fontsize=8)

    ax.set_xlim(0, 0.42)
    ax.set_ylim(300, 2e5)
    ax.set_xlabel("two-qubit error rate p_2q (%)", color=INK_2, fontsize=9)
    ax.set_ylabel("physical qubits needed", color=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_2)
    ax.set_title("Flagship circuit, SI1000 noise: qubits needed vs. two-qubit error rate",
                 loc="left", color=INK, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "docs", "error_rate_sweep.png")
    print("wrote", plot(out))
