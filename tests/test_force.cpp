#include <cmath>
#include <cstdio>

#include "core/force.hpp"

// Finite-difference gradient verification: gravity_accel must equal -grad(potential)
// per harmonic. This is the test that would have caught the legacy sign-flipped J3.

using namespace noether;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void check_close(double got, double ref, double tol, const char* what) {
    if (!(std::fabs(got - ref) <= tol)) {
        std::printf("FAIL: %s: got %.17g ref %.17g |diff| %.3e tol %.3e\n", what, got, ref,
                    std::fabs(got - ref), tol);
        ++failures;
    }
}

struct Point {
    double x, y, z;
};

const Point kPoints[] = {
    {6778.0, 0.0, 0.0},        // equatorial LEO
    {5000.0, 3000.0, 4000.0},  // generic
    {6600.0, -1500.0, 500.0},
    {-4000.0, 6000.0, -2500.0},
};

void test_gradient(unsigned mask, const char* mask_name) {
    const double h = 1.0;  // km
    for (const Point& p : kPoints) {
        double fdp =
            (potential(p.x + h, p.y, p.z, mask) - potential(p.x - h, p.y, p.z, mask)) / (2.0 * h);
        double fdy =
            (potential(p.x, p.y + h, p.z, mask) - potential(p.x, p.y - h, p.z, mask)) / (2.0 * h);
        double fdz =
            (potential(p.x, p.y, p.z + h, mask) - potential(p.x, p.y, p.z - h, mask)) / (2.0 * h);
        Vec3 a = gravity_accel(p.x, p.y, p.z, mask);

        // -grad(Phi) vs analytic force
        auto tol = [](double ref) { return 1e-6 * std::fabs(ref) + 1e-11; };
        char label[128];
        std::snprintf(label, sizeof(label), "%s @ (%.0f,%.0f,%.0f) x", mask_name, p.x, p.y, p.z);
        check_close(-fdp, a.x, tol(a.x), label);
        std::snprintf(label, sizeof(label), "%s @ (%.0f,%.0f,%.0f) y", mask_name, p.x, p.y, p.z);
        check_close(-fdy, a.y, tol(a.y), label);
        std::snprintf(label, sizeof(label), "%s @ (%.0f,%.0f,%.0f) z", mask_name, p.x, p.y, p.z);
        check_close(-fdz, a.z, tol(a.z), label);
    }
}

void test_point_mass_reference() {
    // Circular LEO speed: acceleration magnitude must be MU/r^2 exactly (no harmonics).
    double a = R_EARTH + 400.0;
    Vec3 g = gravity_accel(a, 0.0, 0.0, 0);  // mask 0: point mass only
    check_close(std::sqrt(g.x * g.x + g.y * g.y + g.z * g.z), MU_EARTH / (a * a), 1e-12,
                "point-mass |a| = mu/r^2");
    check(g.x < 0.0, "point-mass accel points inward");
}

void test_hamiltonian_consistency() {
    State s{6778.0, 100.0, 50.0, 1.0, 7.5, 0.2};
    double h = hamiltonian(s);
    check_close(h, 0.5 * (s.vx * s.vx + s.vy * s.vy + s.vz * s.vz) + potential(s.x, s.y, s.z), 0.0,
                "H = T + Phi");
}

void test_mask_monotone() {
    // Each harmonic changes the potential (guards against dead mask bits).
    double p0 = potential(6000.0, 1000.0, 3000.0, 0);
    check(std::fabs(potential(6000.0, 1000.0, 3000.0, J2_BIT) - p0) > 1e-6, "J2 bit active");
    check(std::fabs(potential(6000.0, 1000.0, 3000.0, J3_BIT) - p0) > 1e-8, "J3 bit active");
    check(std::fabs(potential(6000.0, 1000.0, 3000.0, J4_BIT) - p0) > 1e-8, "J4 bit active");
}

}  // namespace

int main() {
    test_gradient(J2_BIT, "J2");
    test_gradient(J3_BIT, "J3");
    test_gradient(J4_BIT, "J4");
    test_gradient(ALL_HARMONICS, "ALL");
    test_point_mass_reference();
    test_hamiltonian_consistency();
    test_mask_monotone();
    if (failures) {
        std::printf("test_force: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_force: OK\n");
    return 0;
}
