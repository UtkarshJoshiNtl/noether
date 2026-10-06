#pragma once

#include <cmath>

#include "constants.hpp"

namespace noether {

// Conservative Earth gravity: point mass + zonal harmonics J2..J4.
// gravity_accel() is the exact negative gradient of potential(); tests/test_force.cpp
// enforces this with central finite differences, per harmonic.
// Convention (standard EGM96): Phi = -(MU/r) * [1 - sum_n Jn (RE/r)^n Pn(z/r)].
// Note: legacy Astrosis shipped a sign-flipped J3 term; the gradient test makes
// that class of bug impossible to reintroduce.

struct Vec3 {
    double x, y, z;
};

struct State {
    double x, y, z, vx, vy, vz;
};

// Harmonic mask bits for gravity_accel / potential.
inline constexpr unsigned J2_BIT = 1;
inline constexpr unsigned J3_BIT = 2;
inline constexpr unsigned J4_BIT = 4;
inline constexpr unsigned ALL_HARMONICS = 7;

inline double potential(double x, double y, double z, unsigned mask = ALL_HARMONICS) {
    double r2 = x * x + y * y + z * z;
    if (r2 < 1.0)
        return 0.0;
    double r = std::sqrt(r2);
    double u = z / r;
    double u2 = u * u;
    double re_r = R_EARTH / r;
    double s2 = re_r * re_r;
    double s3 = s2 * re_r;
    double s4 = s2 * s2;

    double correction = 0.0;
    if (mask & J2_BIT) {
        correction += J2 * s2 * (0.5 * (3.0 * u2 - 1.0));
    }
    if (mask & J3_BIT) {
        correction += J3 * s3 * (0.5 * (5.0 * u2 * u - 3.0 * u));
    }
    if (mask & J4_BIT) {
        correction += J4 * s4 * ((35.0 * u2 * u2 - 30.0 * u2 + 3.0) / 8.0);
    }
    return -(MU_EARTH / r) * (1.0 - correction);
}

inline Vec3 gravity_accel(double x, double y, double z, unsigned mask = ALL_HARMONICS) {
    double r2 = x * x + y * y + z * z;
    double rm = std::sqrt(r2);
    double r3 = r2 * rm;
    double r5 = r3 * r2;
    double r7 = r5 * r2;

    Vec3 a{-MU_EARTH * x / r3, -MU_EARTH * y / r3, -MU_EARTH * z / r3};
    if (rm < 1.0)
        return a;

    double z2r2 = (z * z) / r2;

    if (mask & J2_BIT) {
        double j2f = 1.5 * J2 * MU_EARTH * R_EARTH * R_EARTH / r5;
        a.x += j2f * x * (5.0 * z2r2 - 1.0);
        a.y += j2f * y * (5.0 * z2r2 - 1.0);
        a.z += j2f * z * (5.0 * z2r2 - 3.0);
    }

    if (mask & J3_BIT) {
        double j3f = 2.5 * J3 * MU_EARTH * R_EARTH * R_EARTH * R_EARTH / r7;
        a.x += j3f * x * (7.0 * z2r2 * z - 3.0 * z);
        a.y += j3f * y * (7.0 * z2r2 * z - 3.0 * z);
        a.z += j3f * (7.0 * z2r2 * z * z - 6.0 * z * z + 0.6 * r2);
    }

    if (mask & J4_BIT) {
        double z4r4 = z2r2 * z2r2;
        double j4f = 0.625 * J4 * MU_EARTH * R_EARTH * R_EARTH * R_EARTH * R_EARTH / r7;
        a.x += j4f * x * (3.0 - 42.0 * z2r2 + 63.0 * z4r4);
        a.y += j4f * y * (3.0 - 42.0 * z2r2 + 63.0 * z4r4);
        a.z += j4f * z * (15.0 - 70.0 * z2r2 + 63.0 * z4r4);
    }

    return a;
}

// Total energy per unit mass: H = T + Phi (the Hamiltonian of the separable system).
inline double hamiltonian(const State& s) {
    return 0.5 * (s.vx * s.vx + s.vy * s.vy + s.vz * s.vz) + potential(s.x, s.y, s.z);
}

inline double hamiltonian(double x, double y, double z, double vx, double vy, double vz) {
    return 0.5 * (vx * vx + vy * vy + vz * vz) + potential(x, y, z);
}

}  // namespace noether
