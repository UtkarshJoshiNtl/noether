#pragma once

#include <cmath>
#include <numbers>
#include <vector>

#include "force.hpp"

namespace noether {

// Host-side ensemble construction shared by drift/bench (single source for
// circular-orbit generation and the deterministic low-discrepancy phase spread).

inline State circular_state(double alt_km, double inc_deg, double raan_rad, double u_rad) {
    const double a = R_EARTH + alt_km;
    const double vc = std::sqrt(MU_EARTH / a);
    const double inc = inc_deg * std::numbers::pi / 180.0;

    const double cr = std::cos(raan_rad), sr = std::sin(raan_rad);
    const double ci = std::cos(inc), si = std::sin(inc);
    const double cu = std::cos(u_rad), su = std::sin(u_rad);

    // r0 = (a,0,0), v0 = (0,vc,0) in the orbital plane; rotate Rz(raan)*Rx(inc)*Rz(u).
    auto rot = [&](double x0, double y0, double z0) {
        double x1 = cu * x0 - su * y0, y1 = su * x0 + cu * y0, z1 = z0;
        double x2 = x1, y2 = ci * y1 - si * z1, z2 = si * y1 + ci * z1;
        return Vec3{cr * x2 - sr * y2, sr * x2 + cr * y2, z2};
    };

    Vec3 r = rot(a, 0.0, 0.0);
    Vec3 v = rot(0.0, vc, 0.0);
    return State{r.x, r.y, r.z, v.x, v.y, v.z};
}

// Deterministic quasi-random phase spread (golden-ratio low-discrepancy sequence).
inline std::vector<State> make_ensemble(int n, double alt_km, double inc_deg) {
    constexpr double kGolden = 0.6180339887498949;
    constexpr double kSilver = 0.3247179572447458;
    std::vector<State> states(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        double fi = static_cast<double>(i);
        double raan = 2.0 * std::numbers::pi * std::fmod(fi * kGolden, 1.0);
        double u = 2.0 * std::numbers::pi * std::fmod(fi * kSilver, 1.0);
        states[static_cast<std::size_t>(i)] = circular_state(alt_km, inc_deg, raan, u);
    }
    return states;
}

inline double total_energy(const std::vector<State>& states) {
    long double h = 0.0L;
    for (const State& s : states)
        h += static_cast<long double>(hamiltonian(s));
    return static_cast<double>(h);
}

}  // namespace noether
