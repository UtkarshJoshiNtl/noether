#pragma once

#include <cmath>

#include "force.hpp"

namespace noether {

// All integrators operate on the separable Hamiltonian H = T(p) + Phi(q):
// the force depends on position only, so accelerations are position-only.
//
//   rk4       — classical 4th-order Runge-Kutta (non-symplectic baseline)
//   verlet    — velocity Verlet, kick-drift-kick (2nd-order symplectic)
//   yoshida4  — Yoshida (1990) triple jump of velocity Verlet (4th-order symplectic)
//
// Templated on scalar type: T=double is the reference path; T=float is
// available for reduced-precision studies.

template <typename T>
inline StateT<T> rk4_step(const StateT<T>& s, T dt) {
    Vec3T<T> a1 = gravity_accel(s.x, s.y, s.z);

    T x2 = s.x + T(0.5) * dt * s.vx;
    T y2 = s.y + T(0.5) * dt * s.vy;
    T z2 = s.z + T(0.5) * dt * s.vz;
    T vx2 = s.vx + T(0.5) * dt * a1.x;
    T vy2 = s.vy + T(0.5) * dt * a1.y;
    T vz2 = s.vz + T(0.5) * dt * a1.z;
    Vec3T<T> a2 = gravity_accel(x2, y2, z2);

    T x3 = s.x + T(0.5) * dt * vx2;
    T y3 = s.y + T(0.5) * dt * vy2;
    T z3 = s.z + T(0.5) * dt * vz2;
    T vx3 = s.vx + T(0.5) * dt * a2.x;
    T vy3 = s.vy + T(0.5) * dt * a2.y;
    T vz3 = s.vz + T(0.5) * dt * a2.z;
    Vec3T<T> a3 = gravity_accel(x3, y3, z3);

    T x4 = s.x + dt * vx3;
    T y4 = s.y + dt * vy3;
    T z4 = s.z + dt * vz3;
    T vx4 = s.vx + dt * a3.x;
    T vy4 = s.vy + dt * a3.y;
    T vz4 = s.vz + dt * a3.z;
    Vec3T<T> a4 = gravity_accel(x4, y4, z4);

    T k = dt / T(6);
    StateT<T> out;
    out.x = s.x + k * (s.vx + T(2) * vx2 + T(2) * vx3 + vx4);
    out.y = s.y + k * (s.vy + T(2) * vy2 + T(2) * vy3 + vy4);
    out.z = s.z + k * (s.vz + T(2) * vz2 + T(2) * vz3 + vz4);
    out.vx = s.vx + k * (a1.x + T(2) * a2.x + T(2) * a3.x + a4.x);
    out.vy = s.vy + k * (a1.y + T(2) * a2.y + T(2) * a3.y + a4.y);
    out.vz = s.vz + k * (a1.z + T(2) * a2.z + T(2) * a3.z + a4.z);
    return out;
}

template <typename T>
inline StateT<T> verlet_step(const StateT<T>& s, T dt) {
    Vec3T<T> a1 = gravity_accel(s.x, s.y, s.z);

    T hx = s.x + dt * (s.vx + T(0.5) * dt * a1.x);
    T hy = s.y + dt * (s.vy + T(0.5) * dt * a1.y);
    T hz = s.z + dt * (s.vz + T(0.5) * dt * a1.z);
    Vec3T<T> a2 = gravity_accel(hx, hy, hz);

    StateT<T> out;
    out.x = hx;
    out.y = hy;
    out.z = hz;
    out.vx = s.vx + T(0.5) * dt * (a1.x + a2.x);
    out.vy = s.vy + T(0.5) * dt * (a1.y + a2.y);
    out.vz = s.vz + T(0.5) * dt * (a1.z + a2.z);
    return out;
}

template <typename T>
inline StateT<T> yoshida_step(const StateT<T>& s, T dt) {
    const T c2 = static_cast<T>(std::cbrt(2.0));
    const T w1 = T(1) / (T(2) - c2);
    const T w0 = -c2 / (T(2) - c2);
    StateT<T> t = verlet_step(s, w1 * dt);
    t = verlet_step(t, w0 * dt);
    t = verlet_step(t, w1 * dt);
    return t;
}

}  // namespace noether
