#pragma once

#include <cmath>

#include "constants.hpp"

namespace noether {

// Conservative Earth gravity: point mass + zonal harmonics J2..J4.
// gravity_accel() is the exact negative gradient of potential(); tests enforce
// this with central finite differences, per harmonic.
// Convention (EGM96): Phi = -(MU/r) * [1 - sum_n Jn (RE/r)^n Pn(z/r)].
//
// Physics is templated on the scalar type: T=double is the reference path;
// T=float is available for reduced-precision studies.

template <typename T>
struct Vec3T {
    T x, y, z;
};

template <typename T>
struct StateT {
    T x, y, z, vx, vy, vz;
};

using Vec3 = Vec3T<double>;
using State = StateT<double>;

// Harmonic mask bits for gravity_accel / potential.
inline constexpr unsigned J2_BIT = 1;
inline constexpr unsigned J3_BIT = 2;
inline constexpr unsigned J4_BIT = 4;
inline constexpr unsigned ALL_HARMONICS = 7;

template <typename T>
inline T potential(T x, T y, T z, unsigned mask = ALL_HARMONICS) {
    T r2 = x * x + y * y + z * z;
    if (r2 < T(1))
        return T(0);
    T r = std::sqrt(r2);
    T u = z / r;
    T u2 = u * u;
    T re_r = T(R_EARTH) / r;
    T s2 = re_r * re_r;
    T s3 = s2 * re_r;
    T s4 = s2 * s2;

    T correction = T(0);
    if (mask & J2_BIT) {
        correction += T(J2) * s2 * (T(0.5) * (T(3) * u2 - T(1)));
    }
    if (mask & J3_BIT) {
        correction += T(J3) * s3 * (T(0.5) * (T(5) * u2 * u - T(3) * u));
    }
    if (mask & J4_BIT) {
        correction += T(J4) * s4 * ((T(35) * u2 * u2 - T(30) * u2 + T(3)) / T(8));
    }
    return -(T(MU_EARTH) / r) * (T(1) - correction);
}

template <typename T>
inline Vec3T<T> gravity_accel(T x, T y, T z, unsigned mask = ALL_HARMONICS) {
    T r2 = x * x + y * y + z * z;
    T rm = std::sqrt(r2);
    T r3 = r2 * rm;
    T r5 = r3 * r2;
    T r7 = r5 * r2;

    Vec3T<T> a{-T(MU_EARTH) * x / r3, -T(MU_EARTH) * y / r3, -T(MU_EARTH) * z / r3};
    if (rm < T(1))
        return a;

    T z2r2 = (z * z) / r2;

    if (mask & J2_BIT) {
        T j2f = T(1.5) * T(J2) * T(MU_EARTH) * T(R_EARTH) * T(R_EARTH) / r5;
        a.x += j2f * x * (T(5) * z2r2 - T(1));
        a.y += j2f * y * (T(5) * z2r2 - T(1));
        a.z += j2f * z * (T(5) * z2r2 - T(3));
    }

    if (mask & J3_BIT) {
        T j3f = T(2.5) * T(J3) * T(MU_EARTH) * T(R_EARTH) * T(R_EARTH) * T(R_EARTH) / r7;
        a.x += j3f * x * (T(7) * z2r2 * z - T(3) * z);
        a.y += j3f * y * (T(7) * z2r2 * z - T(3) * z);
        a.z += j3f * (T(7) * z2r2 * z * z - T(6) * z * z + T(0.6) * r2);
    }

    if (mask & J4_BIT) {
        T z4r4 = z2r2 * z2r2;
        T j4f =
            T(0.625) * T(J4) * T(MU_EARTH) * T(R_EARTH) * T(R_EARTH) * T(R_EARTH) * T(R_EARTH) / r7;
        a.x += j4f * x * (T(3) - T(42) * z2r2 + T(63) * z4r4);
        a.y += j4f * y * (T(3) - T(42) * z2r2 + T(63) * z4r4);
        a.z += j4f * z * (T(15) - T(70) * z2r2 + T(63) * z4r4);
    }

    return a;
}

// Total energy per unit mass: H = T + Phi (the Hamiltonian of the separable system).
template <typename T>
inline T hamiltonian(const StateT<T>& s) {
    return T(0.5) * (s.vx * s.vx + s.vy * s.vy + s.vz * s.vz) + potential(s.x, s.y, s.z);
}

template <typename T>
inline T hamiltonian(T x, T y, T z, T vx, T vy, T vz) {
    return T(0.5) * (vx * vx + vy * vy + vz * vz) + potential(x, y, z);
}

}  // namespace noether
