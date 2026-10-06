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
- [ ] **P3 — Experiments → CSVs in `paper/results/`**
  - E1 `drift`: RK4 vs Verlet vs Yoshida, ΔH/H₀ vs t, dt sweep (24 h + 7 d)
  - E2 `reduce`: modes × N=2^10..2^24 × R=100 — bitwise-repeatability + spread
  - E3: 7-day Verlet with each diagnostic reduction mode → noise floor vs
    symplectic bounded oscillation
  - E4: fp64 / fp32 / fp32+Kahan kick — drift vs ns/step Pareto
  - `scripts/run_experiments.sh` regenerates everything
- [ ] **P4 — Figures**: `analysis/fig*.py` (numpy/matplotlib, NOT in build)
  → `paper/figs/*.pdf`
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
