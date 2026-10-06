#pragma once

// Marks functions callable from both host and device when compiling with nvcc.
// force.hpp / integrators.hpp are single-source physics: the same arithmetic
// runs in C++ tests and inside CUDA kernels.

#ifdef __CUDACC__
#define NOETHER_HD __host__ __device__
#else
#define NOETHER_HD
#endif
