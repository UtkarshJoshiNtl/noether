# noether — Master Plan

Research engine + 4-page paper: **"Preserving Hamiltonian Symmetry on Parallel
Hardware: Deterministic Reductions and Mixed-Precision Acceleration in GPU
Symplectic Integration"**

Branch: `alpha` (legacy Astrosis stays on `main`). Engine name: **noether**
(Emmy Noether — symmetry ↔ conservation laws, the paper's thesis).

## Decisions

- **100% standalone C++20/CUDA binary.** No pybind11, no Python runtime in the
  build. Python survives only in `analysis/` (unlinked, consumes CSVs → figures).
- **Conservative model only**: Kepler + J2..J4 (autonomous, static field).
  Drag/SRP/third-body excluded — they break energy conservation, the measured
  quantity. Engine is headless: config in (JSON-subset), CSV/binary out.
- **Force = −∇Φ derived from the closed-form geopotential**, enforced by a
  finite-difference gradient test (per harmonic). Standard EGM96 signs —
  this *fixes a J3 sign bug* present in legacy Astrosis (old J3 = −1× standard,
  verified by harmonic-isolated comparison; survived old tests because
  two-body energy is sign-insensitive to static fields and RAAN tests are
  J2-dominated).

## Phases

- [x] **P0 — Wipe & skeleton**: delete Python-era codebase (71 deletions),
  CMakeLists (C++20, CUDA auto-detect sm_75/80/86/89, OpenMP, CTest),
  `core/constants.hpp`, `main.cpp` usage stub, `.clang-format`, CI →
  cmake+ctest, pre-commit → clang-format + hygiene. Build green, committed.
- [x] **P1 — Host core** (committed `91d2bb4`; measured: order ratios
      RK4 16.59×, Verlet 4.00×, Yoshida 15.98×; 7-day ΔH/H₀: RK4 3.6e-9,
      Verlet 1.8e-7, Yoshida 4.9e-11)
  - `core/force.hpp`: `gravity_accel`, `potential` Φ, `hamiltonian` (½v²+Φ)
  - `core/integrators.hpp`: `rk4_step`, `verlet_step` (kick-drift-kick),
    `yoshida_step` (4th-order composition)
  - `core/reductions.hpp`: serial, Kahan, long-double oracle
  - `io/config.hpp` (JSON-subset parser, ~150 lines), `io/writer.hpp` (CSV)
  - `main.cpp`: real `drift` subcommand → per-sample `t,...,H,dH/H0` CSV
  - CTest: `test_force` (force == −∇Φ FD, per-harmonic), `test_integrators`
    (order: RK4≥14×, Verlet≥3.5×, Yoshida≥14×; 7-day bounded ΔH/H₀ vs RK4
    secular drift; bitwise determinism), `test_config`
  - Gate: `ctest` fully green → commit
- [x] **P2 — CUDA core** (committed; measured on RTX 2050:
      atomic shows natural order spread even single-stream (8–13 unique
      bit-patterns / 100 repeats; fp64 spread ~1e-15 rel, fp32 up to 1.6 abs);
      fixed/kahan bitwise-stable 100/100 and fp64 bit-exact vs long-double
      oracle on lcg input; GPU verlet 4.81 ms/step @ 1M vs CPU 9.97 ms/step
      (2.07×); kahan reduce cost 0.41 vs atomic 0.31 ms)
  - `core/cuda/device_mem.hpp` (RAII port), `core/cuda/timing.hpp` (cudaEvent)
  - `core/cuda/kernels.cu`: `k_verlet_soa` (+rk4/yoshida), `k_energy`,
    reductions `atomic | fixed | kahan` × `{fp64, fp32-accumulator}`
  - commands split out of main.cpp → `cmd/{drift,reduce,bench}.cpp`;
    `drift --backend gpu` with `--red-mode/--red-prec/--red-streams`;
    `reduce` = E2 harness (per-repeat CSV with raw bit patterns);
    `bench` = throughput table + CSV
  - CTest `test_cuda` (skips 77 w/o device): reduction == long-double oracle,
    fixed/kahan bitwise-stable over 100 repeats, atomic spread recorded,
    unity-sum exactness, GPU/CPU trajectory agreement, GPU 7-day ΔH/H₀ bound
- [x] **P3 — Experiments → CSVs in `paper/results/`** (44 files, ~90 s via
      `scripts/run_experiments.sh`; all measured on RTX 2050)
  - E1 `drift` (CPU, dt {10,20,40}, 24 h + 7 d, 18 files): 7 d @ dt=10
      ΔH/H₀ max — RK4 3.56e-9, Verlet 1.04e-7, Yoshida4 2.88e-11
  - E2 `reduce` (n=2^10..2^24 × {lcg, adversarial} × 6 modes × R=100,
      9600 rows): atomic/fp64 13 unique values vs fixed 1 at n=2^20;
      adversarial input → fp32 sums collapse to exactly 0 at every n
      (100% error) while fp64 stays exact
  - E3 (7 d, n=4096, 7 files): fp64 modes resolve the ~1e-7 symplectic
      oscillation; fp32 accumulator quantization floor (ULP(H≈1.2e5)=0.0078
      → dh quanta 6.5e-8 ≥ signal) buries it — fixed/kahan tick at quantum
      boundaries, atomic pins one value for the whole run
  - E4 accuracy (7 d, dt {5,10,20,40}, 12 files): fp32k beats fp32 at every
      dt (dt=10: 6.48e-6 vs 2.73e-5 = 4.2×); fp32 error *falls* with larger
      dt (per-step rounding accumulation dominates truncation — honest
      inversion of the classical dt rule); fp64 2.69e-8 (dt=5) → 1.18e-6
  - E4 throughput (`e4_bench.csv`, n=2^20): fp64 4.816 ms/step,
      fp32 0.483 ms (9.98×), fp32k 0.724 ms (6.65×)
  - Prerequisite landed first: force/integrators templated on scalar type
      (double path verified bit-identical vs pre-refactor references)
- [x] **P4 — Figures**: `analysis/fig{1..4}_*.py` (numpy/matplotlib, NOT in
      build; `analysis/requirements.txt`) → `paper/figs/*.pdf`
  - fig1_drift: 2 panels (dt=10/40), |ΔH/H₀| 7 d — RK4 secular growth
      (5.2e-7→3.6e-6 daily envelope at dt=40) vs bounded Verlet/Yoshida bands
  - fig2_reduce: 3 panels — (a) lcg accuracy ladder fp64 ~1e-16 vs fp32
      ~5e-8, (b) adversarial cliff (fp32 rel err 1.0 flat, fp64 exact),
      (c) determinism: atomic uniques 1→42 with n, fixed/Kahan pinned at 1
  - fig3_noise: first 24 h of E3 — fp64 triple tracks the CPU reference;
      fp32 staircase at the ±6.486e-8 ULP quantum, atomic/fp32 flat at 0
  - fig4_pareto: max|ΔH/H₀| vs ns/step — vertical accuracy bands per
      precision at fixed throughput + CPU verlet star
- [ ] **P5 — Paper**: `paper/main.tex` (IEEEtran 4 pp): Intro / Background /
  Kernels / Setup (RTX 2050, CUDA 12.9, g++ 14) / Results / Threats /
  Conclusion + artifact appendix (3-command repro). Every number traceable
  to a `paper/results/` CSV row.
- [ ] **P6 — Polish**: README final, CI final, clean-clone repro run
  (build → ctest → run_experiments → identical figures).

## Experimental risks

1. atomicAdd order may appear stable for fixed launch configs → E2 includes
   multi-stream/chunked perturbation to force order variation.
2. Kahan serial dependency → per-thread Kahan over strided elements + fixed
   merge; its cost is the E4 Pareto axis, measured not hidden.
3. RTX 2050 FP64 = 1/64 FP32 → strong mixed-precision upside; if fp32
   accuracy fails at dt=10 s, E4 dt-sweep finds the crossover honestly.

## Verification commands

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./scripts/run_experiments.sh        # P3+: regenerates paper/results/*.csv
python analysis/fig1_drift.py       # P4+: regenerates paper/figs/*
```
