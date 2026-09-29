"""How much hardware does a circuit need before error correction pays off?

Sweeps the physical-qubit budget for the paper's flagship circuit (16 logical
qubits, T-count 3,930, p_2q = 0.32%) and plots the best reliability reachable
at each budget. Proposition 3 of the paper guarantees this curve never goes down.

    python examples/budget_staircase.py            # writes docs/budget_staircase.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qregime

N_LOGICAL, TCOUNT, P_PHYS = 16, 3930, 0.0032
SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def sweep(budgets):
    out = []
    for b in budgets:
        r = qregime.best_within_budget(N_LOGICAL, TCOUNT, P_PHYS, b)
        out.append(r["reliability"] if r["realizable"] else 0.0)
    return out


def plot(path):
    budgets = list(range(1_000, 36_001, 100))
    rel = sweep(budgets)
    full = qregime.estimate_ftqc(N_LOGICAL, TCOUNT, P_PHYS)["physicalQubits"]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for q in (0.25, 0.5, 0.75):
        ax.axhline(q, color=GRID, lw=1, zorder=0)
    ax.step(budgets, rel, where="post", color=SERIES, lw=2, zorder=3)

    ax.axvline(1121, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.annotate("1,121 qubits\n(largest single device, IBM Condor)", xy=(1121, 0.93),
                xytext=(1800, 0.93), color=INK_2, fontsize=8, va="top")
    ax.axvline(full, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.annotate(f"full fault tolerance\n{full:,} qubits", xy=(full, 0.35),
                xytext=(full - 7600, 0.35), color=INK_2, fontsize=8, va="top")
    for label, y in (("low", 0.125), ("low-moderate", 0.375),
                     ("moderate", 0.625), ("high", 0.875)):
        ax.text(35_800, y, label, color=INK_2, fontsize=8, ha="right", va="center")

    ax.set_xlim(0, 36_000)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("physical-qubit budget", color=INK_2, fontsize=9)
    ax.set_ylabel("best reachable reliability", color=INK_2, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Flagship circuit: reliability reachable at each qubit budget",
                 loc="left", color=INK, fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "docs", "budget_staircase.png")
    print("wrote", plot(out))
