#!/usr/bin/env python3
"""Figure 3 — diagnostic noise floor: fp64 resolves the oscillation, fp32
does not (E3 data, first 24 h of the 7-day ensemble drift).

The fp64 reductions (fixed and atomic) track the CPU long-double reference
through a smooth bounded ~1e-7 oscillation. The fp32 accumulator quantizes
H to ULP steps of 6.486e-8 in dH/H0 — comparable to the signal itself —
so fixed/Kahan show a staircase and atomic pins a single value all run.

Usage (from repo root):  python3 analysis/fig3_noise.py
Output: paper/figs/fig3_noise.pdf
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "paper", "results")
OUTDIR = os.path.join(ROOT, "paper", "figs")

# (file stem, label, color, linestyle)  — Kahan FP32 is bit-identical to
# fixed FP32 (noted in the caption, kept out of the legend for width).
SERIES = [
    ("cpu", "CPU long-double ref.", "k", "-"),
    ("fixed_fp64", "fixed/fp64", "C0", "--"),
    ("atomic_fp64", "atomic/fp64", "C0", ":"),
    ("fixed_fp32", "fixed/fp32", "C2", "-"),
    ("atomic_fp32", "atomic/fp32", "C3", "-"),
]


def main():
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "lines.linewidth": 0.9,
        }
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.1))

    for stem, label, color, ls in SERIES:
        d = np.loadtxt(f"{RESULTS}/e3_{stem}.csv", delimiter=",", skiprows=1)
        mask = d[:, 0] <= 86400
        ax.plot(d[mask, 0] / 3600.0, d[mask, 2], ls, color=color, label=label)

    quantum = 0.0078125 / 120445.0  # ULP(120445) / |H0|
    for sgn in (-1, 1):
        ax.axhline(sgn * quantum, color="C2", lw=0.5, ls="-.", alpha=0.7)
    ax.text(0.3, quantum * 1.08, "fp32 ULP quantum", fontsize=6, color="C2",
            ha="left", va="bottom")

    ax.axhline(0, color="0.7", lw=0.5)
    ax.set_xlim(0, 24)
    ax.set_ylim(-1.3e-7, 1.3e-7)
    ax.set_xlabel("time [h]")
    ax.set_ylabel(r"$\Delta H/H_0$")
    ax.grid(True, which="major", alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper right", ncol=2, frameon=False)

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "fig3_noise.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig("/tmp/opencode/fig3_noise.png", dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
