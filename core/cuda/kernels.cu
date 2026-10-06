// CUDA kernels for the noether engine.
//
// P2 populates this translation unit with:
//   k_verlet_soa          — velocity-Verlet over an SoA ensemble
//   k_energy              — per-particle Hamiltonian evaluation
//   k_reduce_atomic       — block-shared tree + atomicAdd (nondeterministic order)
//   k_reduce_fixed        — fixed-order deterministic reduction
//   k_reduce_kahan        — fixed-order + Kahan compensated summation
// Each reduction also ships an FP32-accumulator variant.

#include <cuda_runtime.h>
