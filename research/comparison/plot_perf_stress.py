"""Plot the performance and stress test results.

Reads research/comparison/perf_stress_results.json (written by perf_stress.py)
and plots run time and peak memory against circuit size for the operations
that read the circuit. The analytic estimators (qregime.estimate_ftqc,
azure.counts, qrechem.formula) take under 2 ms at every size and are left to
the table in PERF_STRESS.md. A cross marks the first size at which a tool
timed out, ran out of memory or raised an error.

    python research/comparison/plot_perf_stress.py   # writes docs/perf_stress.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Categorical slots 1-4, validated (CVD and normal-vision separation pass).
# Slots 3-4 are below 3:1 contrast on the surface, so every series also has
# its own marker and the numbers are tabulated in PERF_STRESS.md.
SERIES = {
    "qregime.decide_regime": ("#2a78d6", "o", "qregime decide_regime"),
    "qregime.profile": ("#eb6834", "s", "qregime profile"),
    "azure.qasm": ("#1baf7a", "D", "Azure QRE (QASM input)"),
    "mitiq.fold": ("#eda100", "^", "Mitiq fold (scale 3)"),
}
FAMILY_TITLES = {"clifford_t": "Clifford+T circuits", "rotations": "arbitrary-rotation circuits"}


def load():
    with open(os.path.join(HERE, "perf_stress_results.json")) as f:
        return json.load(f)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="y", color=GRID, lw=1, zorder=0)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot(path):
    data = load()
    rows = data["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for col, family in enumerate(("clifford_t", "rotations")):
        for row, (key, ylabel) in enumerate((("op_s", "run time (s)"),
                                             ("peak_rss_mb", "peak memory (MB)"))):
            ax = axes[row][col]
            style(ax)
            for op, (color, marker, label) in SERIES.items():
                pts = [r for r in rows if r["op"] == op and r["family"] == family]
                ok = [r for r in pts if r["status"] == "ok"]
                bad = [r for r in pts if r["status"] != "ok"]
                xs = [r["gates"] for r in ok]
                ys = [max(r[key], 1e-3) for r in ok]
                ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=6, zorder=3,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
                for r in bad:
                    y = data["timeout_s"] if key == "op_s" else r["polled_peak_rss_mb"]
                    ax.plot([r["gates"]], [y], marker="x", ms=9, mew=2, color=color, zorder=4)
            if key == "op_s":
                ax.axhline(data["timeout_s"], color=INK_2, lw=1, ls=(0, (3, 3)))
                ax.text(1, data["timeout_s"] * 1.15, transform=ax.get_yaxis_transform(),
                        ha="right", s=f"{data['timeout_s']} s timeout", color=INK_2, fontsize=7)
            else:
                ax.axhline(data["memory_limit_mb"], color=INK_2, lw=1, ls=(0, (3, 3)))
                ax.text(1, data["memory_limit_mb"] * 1.1, transform=ax.get_yaxis_transform(),
                        ha="right", s=f"{data['memory_limit_mb']:,} MB limit", color=INK_2,
                        fontsize=7)
            if col == 0:
                ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
            if row == 1:
                ax.set_xlabel("gates in the circuit", color=INK_2, fontsize=9)
            if row == 0:
                ax.set_title(FAMILY_TITLES[family], loc="left", color=INK, fontsize=10, pad=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, marker="x", ls="none", ms=8, mew=2))
    labels.append("first failure (timeout)")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=5,
               fontsize=8, frameon=False, labelcolor=INK_2)
    fig.suptitle("Run time and peak memory vs. circuit size (one process per run)",
                 x=0.06, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    plot(os.path.join(ROOT, "docs", "perf_stress.png"))
