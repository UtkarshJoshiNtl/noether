#!/usr/bin/env python3
"""Figure 4 — mixed-precision Pareto front: accuracy vs throughput (E4).

x: ns/step at n = 2^20 (compute-bound throughput, from e4_bench.csv)
y: max |dH/H0| over the 7-day single-satellite run (from e4_<prec>_dt*.csv)

Three variants (fp64, fp32, fp32+kick) at dt in {5, 10, 20, 40} s form
vertical accuracy bands at fixed throughput costs; the CPU fp64 verlet
point is included for context.

Usage (from repo root):  python3 analysis/fig4_pareto.py
Output: paper/figs/fig4_pareto.pdf
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "paper", "results")
OUTDIR = os.path.join(ROOT, "paper", "figs")

# variant -> (bench detail, color, marker)
VARIANTS = [
    ("fp64", "gpu:verlet", "C0", "o"),
    ("fp32k", "gpu:verlet-fp32k", "C2", "D"),
    ("fp32", "gpu:verlet-fp32", "C3", "s"),
]
DTS = [5, 10, 20, 40]


def main():
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "lines.linewidth": 0.8,
        }
    )
    bench = np.genfromtxt(f"{RESULTS}/e4_bench.csv", delimiter=",", names=True,
                          dtype=None, encoding=None)
    ns_step = {r["detail"]: float(r["ns_per_call"]) for r in bench}

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for prec, detail, color, marker in VARIANTS:
        x = ns_step[detail]
        ys, dts = [], []
        for dt in DTS:
            d = np.loadtxt(f"{RESULTS}/e4_{prec}_dt{dt}.csv", delimiter=",", skiprows=1)
            ys.append(np.abs(d[:, 8]).max())
            dts.append(dt)
        ax.plot([x] * len(ys), ys, "-", color=color, lw=0.6, alpha=0.7)
        ax.loglog([x] * len(ys), ys, linestyle="None", marker=marker, color=color,
                  ms=4.5, label=f"{prec}")
        for xx, yy, dt in zip([x] * len(ys), ys, dts):
            ax.annotate(f"{dt}", (xx, yy), fontsize=5.5, color=color,
                        xytext=(3, 0), textcoords="offset points")

    # CPU fp64 verlet for context (same physics as e4 fp64 dt=10).
    cpu_ns = ns_step["cpu:verlet"]
    d = np.loadtxt(f"{RESULTS}/e1_verlet_dt10_7d.csv", delimiter=",", skiprows=1)
    cpu_dh = np.abs(d[:, 8]).max()
    ax.loglog([cpu_ns], [cpu_dh], "*", color="0.35", ms=9, label="cpu:verlet fp64")

    ax.text(0.03, 0.05, "labels: $\\Delta t$ [s]", transform=ax.transAxes, fontsize=6)
    ax.set_xlabel("throughput cost [ns/step]  ($n = 2^{20}$)")
    ax.set_ylabel(r"max $|\Delta H/H_0|$ over 7 d")
    ax.grid(True, which="major", alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(3e5, 3e7)

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "fig4_pareto.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig("/tmp/opencode/fig4_pareto.png", dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
