#pragma once

#include <cmath>

#include "force.hpp"
#include "hd.hpp"

namespace noether {

// All integrators operate on the separable Hamiltonian H = T(p) + Phi(q):
// the force depends on position only, so accelerations are position-only.
//
//   rk4       — classical 4th-order Runge-Kutta (non-symplectic baseline)
//   verlet    — velocity Verlet, kick-drift-kick (2nd-order symplectic)
//   yoshida4  — Yoshida (1990) triple jump of velocity Verlet (4th-order symplectic)

NOETHER_HD inline State rk4_step(const State& s, double dt) {
    Vec3 a1 = gravity_accel(s.x, s.y, s.z);

    double x2 = s.x + 0.5 * dt * s.vx;
    double y2 = s.y + 0.5 * dt * s.vy;
    double z2 = s.z + 0.5 * dt * s.vz;
    double vx2 = s.vx + 0.5 * dt * a1.x;
    double vy2 = s.vy + 0.5 * dt * a1.y;
    double vz2 = s.vz + 0.5 * dt * a1.z;
    Vec3 a2 = gravity_accel(x2, y2, z2);

    double x3 = s.x + 0.5 * dt * vx2;
    double y3 = s.y + 0.5 * dt * vy2;
    double z3 = s.z + 0.5 * dt * vz2;
    double vx3 = s.vx + 0.5 * dt * a2.x;
    double vy3 = s.vy + 0.5 * dt * a2.y;
    double vz3 = s.vz + 0.5 * dt * a2.z;
    Vec3 a3 = gravity_accel(x3, y3, z3);

    double x4 = s.x + dt * vx3;
    double y4 = s.y + dt * vy3;
    double z4 = s.z + dt * vz3;
    double vx4 = s.vx + dt * a3.x;
    double vy4 = s.vy + dt * a3.y;
    double vz4 = s.vz + dt * a3.z;
    Vec3 a4 = gravity_accel(x4, y4, z4);

    double k = dt / 6.0;
    State out;
    out.x = s.x + k * (s.vx + 2.0 * vx2 + 2.0 * vx3 + vx4);
    out.y = s.y + k * (s.vy + 2.0 * vy2 + 2.0 * vy3 + vy4);
    out.z = s.z + k * (s.vz + 2.0 * vz2 + 2.0 * vz3 + vz4);
    out.vx = s.vx + k * (a1.x + 2.0 * a2.x + 2.0 * a3.x + a4.x);
    out.vy = s.vy + k * (a1.y + 2.0 * a2.y + 2.0 * a3.y + a4.y);
    out.vz = s.vz + k * (a1.z + 2.0 * a2.z + 2.0 * a3.z + a4.z);
    return out;
}

NOETHER_HD inline State verlet_step(const State& s, double dt) {
    Vec3 a1 = gravity_accel(s.x, s.y, s.z);

    double hx = s.x + dt * (s.vx + 0.5 * dt * a1.x);
    double hy = s.y + dt * (s.vy + 0.5 * dt * a1.y);
    double hz = s.z + dt * (s.vz + 0.5 * dt * a1.z);
    Vec3 a2 = gravity_accel(hx, hy, hz);

    State out;
    out.x = hx;
    out.y = hy;
    out.z = hz;
    out.vx = s.vx + 0.5 * dt * (a1.x + a2.x);
    out.vy = s.vy + 0.5 * dt * (a1.y + a2.y);
    out.vz = s.vz + 0.5 * dt * (a1.z + a2.z);
    return out;
}

NOETHER_HD inline State yoshida_step(const State& s, double dt) {
    const double c2 = std::cbrt(2.0);
    const double w1 = 1.0 / (2.0 - c2);
    const double w0 = -c2 / (2.0 - c2);
    State t = verlet_step(s, w1 * dt);
    t = verlet_step(t, w0 * dt);
    t = verlet_step(t, w1 * dt);
    return t;
}

}  // namespace noether
