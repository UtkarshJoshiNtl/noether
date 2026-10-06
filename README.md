# noether

A lean C++20/CUDA research engine for studying **Hamiltonian symmetry preservation on
parallel hardware**: symplectic integration, deterministic parallel reductions, and
mixed-precision acceleration on consumer GPUs.

Backs the 4-page paper in [`paper/`](paper/main.pdf): *"Preserving Hamiltonian Symmetry
on Parallel Hardware: Deterministic Reductions and Mixed-Precision Acceleration in GPU
Symplectic Integration."*

No Python runtime. No UI. No plotting. The engine computes and writes raw CSV data;
figures are produced externally from `paper/results/` by `analysis/`.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

- C++20, `-O3 -Wall -Wextra`, no fast-math (FP64 stays IEEE-conformant).
- CUDA is auto-detected (sm_75/80/86/89); force CPU-only with `-DNOETHER_USE_CUDA=OFF`.
- CI builds and tests the CPU-only configuration (`.github/workflows/ci.yml`).

## Usage

```sh
# 7-day energy drift, GPU, mixed-precision Verlet with Kahan kick compensation
./build/noether drift --backend gpu --precision fp32k --dt 10 --t 604800 --out drift.csv

# Reduction determinism: 100 repeats, n = 2^20, all modes x precisions
./build/noether reduce --n 1048576 --repeats 100 --input lcg --out reduce.csv

# Throughput (best of 5, include the OpenMP CPU reference)
./build/noether bench --n 1048576 --steps 200 --repeats 5 --cpu --out bench.csv
```

`drift` options: `--integrator rk4|verlet|yoshida4`, `--backend cpu|gpu`,
`--precision fp64|fp32|fp32k` (GPU Verlet only), `--red-mode atomic|fixed|kahan`,
`--red-prec fp64|fp32`, `--n/--alt/--inc/--sample`. See `noether <cmd> --help`.

## Reproduce the paper

```sh
./scripts/run_experiments.sh              # 44 CSVs -> paper/results/  (~90 s, GPU)
python3 analysis/fig1_drift.py            # -> paper/figs/fig1_drift.pdf
python3 analysis/fig2_reduce.py           # fig2_reduce / fig3_noise / fig4_pareto
python3 analysis/fig3_noise.py
python3 analysis/fig4_pareto.py
cd paper && tectonic main.tex             # -> paper/main.pdf (4 pp.)
```

Figure scripts need `matplotlib`/`numpy` (`analysis/requirements.txt`); they are not
part of the engine build. Every number in the paper is computed from
`paper/results/*.csv` — see the artifact appendix in `paper/main.tex`.

## Layout

```
core/        physics, integrators, reductions (header-only, C++20)
core/cuda/   CUDA kernels (symplectic steppers, atomic/fixed/Kahan reductions)
cmd/         drift / reduce / bench subcommands
io/          JSON-subset config parser, CSV writers
tests/       CTest suite (force == -grad(Phi), integrator order, CUDA parity)
analysis/    Python figure scripts (consumes CSVs; NOT part of the build)
scripts/     run_experiments.sh (regenerates every paper/results/*.csv)
paper/       main.tex + main.pdf, figures, raw experiment results
```

## License

MIT — see [LICENSE](LICENSE).
