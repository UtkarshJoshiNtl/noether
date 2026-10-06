#!/usr/bin/env python3
"""Figure 1 — Hamiltonian drift of RK4 vs Verlet vs Yoshida (E1 data).

Two panels (dt = 10 s and dt = 40 s): |dH/H0| over 7 days on a log axis.
Symplectic integrators stay in bounded oscillation bands; RK4 grows
secularly and overtakes Verlet at the larger step.

Usage (from repo root):  python3 analysis/fig1_drift.py
Output: paper/figs/fig1_drift.pdf
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "paper", "results")
OUTDIR = os.path.join(ROOT, "paper", "figs")

STYLE = {
    "rk4": ("RK4", "C3", "-"),
    "verlet": ("Verlet", "C0", "-"),
    "yoshida4": ("Yoshida-4", "C2", "-"),
}


def envelope(dh, t, days=7):
    return [np.abs(dh[(t >= k * 86400) & (t < (k + 1) * 86400)]).max() for k in range(days)]


def main():
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "lines.linewidth": 0.9,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True)

    for ax, dt in zip(axes, (10, 40)):
        for key, (label, color, ls) in STYLE.items():
            d = np.loadtxt(f"{RESULTS}/e1_{key}_dt{dt}_7d.csv", delimiter=",", skiprows=1)
            t = d[:, 0] / 86400.0
            dh = np.abs(d[:, 8])
            ok = dh > 0
            ax.semilogy(t[ok], dh[ok], ls, color=color, label=label)
        ax.set_ylim(1e-13, 3e-5)
        ax.grid(True, which="major", alpha=0.3, linewidth=0.4)
        ax.text(0.02, 0.06, rf"$\Delta t = {dt}\,$s", transform=ax.transAxes, fontsize=6.5)
    axes[0].legend(loc="lower right", ncol=3, frameon=False)
    axes[0].set_ylabel(r"$|\Delta H/H_0|$")
    axes[1].set_ylabel(r"$|\Delta H/H_0|$")
    axes[1].set_xlabel("time [days]")
    axes[1].set_xlim(0, 7)
    axes[1].set_xticks(range(8))

    # Daily-envelope growth annotation on the dt=40 panel.
    d = np.loadtxt(f"{RESULTS}/e1_rk4_dt40_7d.csv", delimiter=",", skiprows=1)
    env = envelope(d[:, 8], d[:, 0])
    print("rk4 dt40 daily envelope:", " ".join(f"{e:.1e}" for e in env))

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "fig1_drift.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig("/tmp/opencode/fig1_drift.png", dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
