#!/usr/bin/env python3
"""Figure 2 — reduction quality ladder, adversarial cliff, determinism (E2).

Three panels:
  (a) lcg input: relative error vs n — fp64 accumulators sit on the
      summation floor (1.2e-17 .. 3.6e-15 over all modes); fp32 spans
      1.8e-9 .. 7.4e-8 for fixed/Kahan and reaches 3.0e-6 for atomic
      (n=2^22)
  (b) adversarial input (alternating +/-1e8 + residue): fp32 collapses to a
      relative error of 1.0 (sum rounds to exactly 0) at every n, while fp64
      stays exact
  (c) bitwise determinism: distinct values in 100 repeats — fixed/Kahan are
      always 1, atomic varies with n (peaks at 42 FP64 / 47 FP32 at n=2^22)

Usage (from repo root):  python3 analysis/fig2_reduce.py
Output: paper/figs/fig2_reduce.pdf
"""

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-reproducible PDFs

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "paper", "results", "e2_reduce.csv")
OUTDIR = os.path.join(ROOT, "paper", "figs")

COLORS = {
    ("atomic", "fp64"): "C3",
    ("fixed", "fp64"): "C0",
    ("kahan", "fp64"): "C0",
    ("atomic", "fp32"): "C1",
    ("fixed", "fp32"): "C2",
    ("kahan", "fp32"): "C2",
}
MARKERS = {"atomic": "o", "fixed": "s", "kahan": "^"}


def lcg_values(nmax, seed=42):
    """Vector-prefix of the engine's LCG input (one sequential pass)."""
    v = np.empty(nmax)
    s = seed
    mul = 6364136223846793005
    add = 1442695040888963407
    for i in range(nmax):
        s = (s * mul + add) & 0xFFFFFFFFFFFFFFFF
        v[i] = (s >> 11) * (1.0 / 9007199254740992.0)
    return v


def adversarial_oracle(n):
    i = np.arange(n, dtype=np.int64)
    vals = np.where(i % 2 == 1, -(10**8), 10**8) + (i % 7)
    return int(vals.sum())


def series_points(rows, inp, mode, prec, ns, oracle_fn):
    pts = []
    for n in ns:
        rs = rows.get((inp, n, mode, prec))
        if rs is None:
            continue
        oracle = oracle_fn(n)
        v = float(rs[0]["value"])
        rel = abs(v - oracle) / abs(oracle)
        pts.append((n, max(rel, 1e-18)))
    return pts


def draw(ax, pts, mode, prec, label=None, **kw):
    if not pts:
        return
    xs, ys = zip(*pts)
    ax.loglog(xs, ys, MARKERS[mode], color=COLORS[(mode, prec)],
              mec=COLORS[(mode, prec)],
              mfc="white" if mode == "kahan" else COLORS[(mode, prec)],
              ms=3.5, label=label, **kw)


def main():
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "lines.linewidth": 0.8,
        }
    )
    raw = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding=None)
    rows = {}
    for r in raw:
        rows.setdefault((r["input"], int(r["n"]), r["mode"], r["prec"]), []).append(r)
    ns_all = sorted({int(r["n"]) for r in raw})

    # One sequential LCG pass, exact prefix sums at each requested n.
    lcg = lcg_values(max(ns_all))
    lcg_oracle = {n: float(np.longdouble(lcg[:n]).sum()) for n in ns_all}
    adv_oracle = {n: adversarial_oracle(n) for n in ns_all}

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.1))

    ax = axes[0]
    for mode in ("atomic", "fixed", "kahan"):
        for prec in ("fp64", "fp32"):
            pts = series_points(rows, "lcg", mode, prec, ns_all,
                                lambda n: lcg_oracle[n])
            draw(ax, pts, mode, prec, label=f"{mode}/{prec}")
    ax.set_title("(a) uniform input: accumulator accuracy", fontsize=6.5, loc="left")
    ax.set_ylim(1e-17, 1e-4)
    ax.set_ylabel("rel. error")
    ax.legend(ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8)

    ax = axes[1]
    for mode in ("atomic", "fixed", "kahan"):
        for prec in ("fp64", "fp32"):
            pts = series_points(rows, "adversarial", mode, prec, ns_all,
                                lambda n: adv_oracle[n])
            draw(ax, pts, mode, prec)
    ax.axhline(1.0, color="C1", lw=0.5, ls=":")
    ax.text(0.02, 0.45, "fp32: sum rounds to exactly 0 (rel. error 1)",
            transform=ax.transAxes, fontsize=6, color="C1")
    ax.text(0.02, 0.10, "fp64: exact for every n (rel. error 0)",
            transform=ax.transAxes, fontsize=6, color="C0")
    ax.set_title(r"(b) adversarial input $\pm10^8$ + residue: cliff",
                 fontsize=6.5, loc="left")
    ax.set_ylim(5e-19, 3.0)
    ax.set_ylabel("rel. error")
    ax.set_xlabel("$n$ (number of terms)")

    ax = axes[2]
    for mode in ("atomic", "fixed", "kahan"):
        for prec in ("fp64", "fp32"):
            pts = []
            for n in ns_all:
                rs = rows.get(("lcg", n, mode, prec))
                if rs is None:
                    continue
                pts.append((n, len({r["bits"] for r in rs})))
            if pts:
                xs, ys = zip(*pts)
                ax.semilogx(xs, ys, MARKERS[mode], color=COLORS[(mode, prec)],
                            mec=COLORS[(mode, prec)],
                            mfc="white" if mode == "kahan" else COLORS[(mode, prec)],
                            ms=3.5)
    ax.axhline(1, color="C0", lw=0.5, ls=":")
    ax.text(0.02, 0.14, "fixed / Kahan: 1 value in 100 repeats (deterministic)",
            transform=ax.transAxes, fontsize=6, color="C0")
    ax.set_title("(c) bitwise repeatability: distinct values in 100 repeats",
                 fontsize=6.5, loc="left")
    ax.set_ylim(0.5, None)
    ax.set_ylabel("distinct values")
    ax.set_xlabel("$n$ (number of terms)")

    for ax in axes:
        ax.grid(True, which="major", alpha=0.3, linewidth=0.4)
        ticks = [2**e for e in range(10, 25, 4)] + [2**24]
        ax.set_xticks(ticks)
        ax.set_xticklabels([r"$2^{" + str(e) + "}$" for e in range(10, 25, 4)] + [r"$2^{24}$"])

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "fig2_reduce.pdf")
    fig.tight_layout(h_pad=0.8)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig("/tmp/opencode/fig2_reduce.png", dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
