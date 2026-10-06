# noether — project notes

Durable reference (supersedes the old `plan.md`). README covers usage/repro;
`git log` covers phase history (P0–P6, branch `alpha`). Every number below
was verified against `paper/results/*.csv` with an exact-integer oracle.

## Design decisions

- Standalone C++20/CUDA binary; no Python in the build. Python exists only in
  `analysis/` (unlinked; consumes CSVs → figures).
- Conservative model ONLY: Kepler + J2..J4 (autonomous, closed-form Φ,
  force = −∇Φ enforced by a finite-difference gradient test, rel tol 1e-6).
  Dissipative forces would change ΔH by design — ΔH is the measured quantity.
- Standard EGM96 J3 sign (fixes a legacy Astrosis bug: old J3 = −1× standard;
  old tests missed it because two-body energy is sign-insensitive to static
  fields and RAAN tests are J2-dominated).
- Fixed-step only (warp uniformity); no fast-math anywhere. Yoshida:
  c2 = cbrt(2), w1 = 1/(2−c2), w0 = −c2/(2−c2).
- One templated reduction kernel `k_partial<Acc,Kahan>` (BLOCK=256);
  finalize = atomic (order nondeterministic) | fixed out-of-place merge tree
  (`k_halve`) | Kahan. fp32k = persistent per-particle Kahan comp arrays on
  the velocity kick only (zeroed at upload, not per step); energies always
  evaluated fp64 from fp32 state.
- Figure PDFs are byte-reproducible (`SOURCE_DATE_EPOCH=0` setdefault at the
  top of every fig script — do not remove).

## Measured facts (source: `paper/results/*.csv`)

- **E1** (7 d, dt=10, 51.6° orbit, 600 s cadence): ΔH/H₀ max — RK4 3.56e-9
  (secular; daily envelope 5.0e-10 → 3.6e-9), Verlet 1.04e-7 (bounded),
  Yoshida4 2.88e-11. At dt=40: RK4 3.64e-6 > Verlet 1.18e-6 (crossover;
  Yoshida4 7.38e-9).
- **E2** (9 600 rows): fixed/Kahan return exactly 1 distinct value in 100
  repeats at every n; atomic uniques (lcg, n=2^10..2^24) — fp64
  1,1,1,2,5,13,42,28; fp32 1,1,1,10,8,23,47,43. Spread *varies* with n
  (peaks at 2²², dips at 2²⁴ — not strictly monotone).
- **E2 accuracy** vs exact integer oracle (lcg, seed 42): fp64 rel err
  1.17e-17..3.57e-15 over all modes (fixed/kahan 4.3e-17..1.2e-16, atomic
  carries the range ends); fp32 fixed/kahan 1.77e-9..7.39e-8, atomic up to
  2.97e-6 at n=2²² (2.20e-6 at 2²⁴).
- **E2 adversarial** (±1e8 + i%7 residue): fp32 → exactly 0 (rel err 1.0)
  in every mode, size, and repeat; fp64 bit-exact in every row.
- **Timings** — two harnesses, do not mix:
  - Paper quotes `e4_bench.csv` reduce rows (warmup, best-of-5):
    0.309 / 0.342 / 0.405 ms (atomic/fixed/kahan, fp64, n=2²⁰) → 11% / 31%.
  - `e2_reduce.csv` `ms` column is per-repeat; medians there are ~0.33 / 0.38 /
    0.45 ms. Both are real; different methodology.
- **E3** (4096 bodies, 7 d): fp64 modes + CPU long-double reference share an
  identical range [-5.18e-8, +2.66e-8], 100% nonzero; fp32 fixed/kahan
  staircase ±6.486e-8 (= ULP(1.2e5)/|H₀| = 2⁻⁷/120445), 68.7% nonzero;
  atomic fp32 all-zero; kahan fp32 bit-identical to fixed fp32.
- **E4** (GPU Verlet, table in paper = CSV max |dh| exactly): fp32k beats
  fp32 at every dt (4.2× at dt=10); fp32 error *falls* with dt
  (2.80e-5 @ 5 s → 6.72e-6 @ 40 s — per-step rounding dominates truncation).
- **Bench** @ n=2²⁰, best of 5: cpu verlet 9.601 ms, gpu fp64 4.816 (1.99×),
  fp32 0.483 (9.98×), fp32k 0.724 (6.65× vs gpu, 13.3× vs cpu).

## Test-vs-paper number note

`tests/test_integrators.cpp` prints ΔH for its *equatorial* reference orbit
sampled every 600 steps (Verlet 1.8e-7); E1/paper numbers use the 51.6°
orbit sampled every 600 s (1.04e-7). Both are correct for their config. The
convergence ratios it prints (16.59 / 4.00 / 15.98) are the ones the paper
quotes.

## Repro status (P6, commit `178e3a0`)

Clean clone from `/tmp` verified: CPU-only CI sim warning-free, 4/4 ctest;
CUDA build 5/5; figure PDFs byte-identical when regenerated from committed
CSVs; full `run_experiments.sh` rerun → **37/39 CSVs byte-identical**
(`e4_bench` differs only in timing; `e2_reduce` differs in `ms` for
9597/9600 rows + 839 value/bits rows — all atomic, fixed/Kahan 0
differences), i.e. the rerun itself reproduces the determinism claim.
Note: repo-wide "44 CSVs" claims were wrong — the real count is **39**.

## Verification commands

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure        # 5/5 (4/4 CPU-only)
./scripts/run_experiments.sh                      # ~90 s, 39 CSVs
python3 analysis/fig{1_drift,2_reduce,3_noise,4_pareto}.py
cd paper && tectonic main.tex                     # 4 pp.
```
