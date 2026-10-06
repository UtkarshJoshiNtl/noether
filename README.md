# noether

A lean C++20/CUDA research engine for studying **Hamiltonian symmetry preservation on
parallel hardware**: symplectic integration, deterministic parallel reductions, and
mixed-precision acceleration on consumer GPUs.

No Python runtime. No UI. No plotting. The engine computes and writes raw CSV/binary
data; figures are produced externally from `paper/results/`.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

- CUDA is auto-detected (sm_75/80/86/89); force CPU-only with `-DNOETHER_USE_CUDA=OFF`.
- The artifact for the paper lives in `paper/`; reproduce all experiment data with
  `scripts/run_experiments.sh`.

## Layout

```
core/        physics, integrators, reductions (header-only, C++20)
core/cuda/   CUDA kernels (velocity-Verlet, deterministic reductions)
io/          JSON-subset config parser, CSV/binary writers
tests/       CTest suite
analysis/    Python figure scripts (consumes CSVs; NOT part of the build)
paper/       LaTeX source, figures, raw experiment results
```
