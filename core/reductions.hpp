#pragma once

#include <cstddef>

namespace noether {

// Host-side summation references. The CUDA reduction kernels are validated
// against these: fixed/kahan must match the serial/Kahan results exactly,
// and the long-double accumulator serves as the high-precision oracle.

inline double sum_serial(const double* v, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        s += v[i];
    return s;
}

inline double sum_kahan(const double* v, std::size_t n) {
    double sum = 0.0;
    double comp = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double y = v[i] - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    return sum;
}

inline long double sum_longdouble(const double* v, std::size_t n) {
    long double s = 0.0L;
    for (std::size_t i = 0; i < n; ++i)
        s += static_cast<long double>(v[i]);
    return s;
}

}  // namespace noether
