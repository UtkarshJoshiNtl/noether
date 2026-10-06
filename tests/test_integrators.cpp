#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/integrators.hpp"

// Integrator verification: convergence order, 7-day energy boundedness,
// and bitwise run-to-run determinism.

using namespace noether;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

State circular() {
    double a = R_EARTH + 400.0;
    double vc = std::sqrt(MU_EARTH / a);
    return State{a, 0.0, 0.0, 0.0, vc, 0.0};
}

using StepFn = State (*)(const State&, double);

State propagate(StepFn step, State s, double dt, int steps) {
    for (int i = 0; i < steps; ++i)
        s = step(s, dt);
    return s;
}

double dist6(const State& a, const State& b) {
    double d[6] = {a.x - b.x, a.y - b.y, a.z - b.z, a.vx - b.vx, a.vy - b.vy, a.vz - b.vz};
    double s = 0.0;
    for (double v : d)
        s += v * v;
    return std::sqrt(s);
}

void test_convergence(StepFn step, const char* name, double min_ratio) {
    // Self-convergence over T = 640 s (dt divides T exactly): all runs share
    // the same final epoch, so the difference ratios isolate truncation order.
    const double T = 640.0;
    const double dt = 80.0;
    State s1 = propagate(step, circular(), dt, static_cast<int>(T / dt));
    State s2 = propagate(step, circular(), dt / 2, static_cast<int>(T / (dt / 2)));
    State s3 = propagate(step, circular(), dt / 4, static_cast<int>(T / (dt / 4)));
    double e1 = dist6(s1, s2);
    double e2 = dist6(s2, s3);
    double ratio = e1 / e2;
    std::printf("%-8s convergence ratio = %.2f (order ~ %.2f)\n", name, ratio, std::log2(ratio));
    check(ratio >= min_ratio, (std::string(name) + " convergence order").c_str());
}

void test_energy_bounded(StepFn step, const char* name, double dt, double bound) {
    // 7 days, dt = 10 s: symplectic integrators must keep |dH/H0| bounded.
    State s = circular();
    double h0 = hamiltonian(s);
    const int steps = static_cast<int>(7 * 86400 / dt);
    double max_dh = 0.0;
    double last_dh = 0.0;
    for (int i = 0; i < steps; ++i) {
        s = step(s, dt);
        if (i % 600 == 0 || i == steps - 1) {  // sample every ~100 min
            last_dh = (hamiltonian(s) - h0) / std::fabs(h0);
            max_dh = std::max(max_dh, std::fabs(last_dh));
        }
    }
    std::printf("%-8s 7d dt=%.0f: max_dh_rel = %.3e final = %.3e\n", name, dt, max_dh, last_dh);
    check(max_dh <= bound, (std::string(name) + " 7-day energy boundedness").c_str());
    // Orbit must still be bound (no blow-up).
    double r = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
    check(r > R_EARTH + 200.0 && r < R_EARTH + 2000.0,
          (std::string(name) + " orbit still bound").c_str());
}

void test_determinism(StepFn step, const char* name) {
    State a = propagate(step, circular(), 7.0, 5000);
    State b = propagate(step, circular(), 7.0, 5000);
    check(std::memcmp(&a, &b, sizeof(State)) == 0,
          (std::string(name) + " bitwise deterministic").c_str());
}

}  // namespace

int main() {
    struct Case {
        StepFn step;
        const char* name;
        double conv_min_ratio;
        double energy_bound;
    };
    const Case cases[] = {
        {rk4_step, "rk4", 14.0, 1e-7},   // measured 16.59× / 3.6e-9 @ dt=10s / 7 d
        {verlet_step, "verlet", 3.5, 1e-6},  // measured 4.00× / 1.8e-7
        {yoshida_step, "yoshida4", 14.0, 1e-8},  // measured 15.98× / 4.9e-11
    };

    for (const Case& c : cases) {
        test_convergence(c.step, c.name, c.conv_min_ratio);
        test_energy_bounded(c.step, c.name, 10.0, c.energy_bound);
        test_determinism(c.step, c.name);
    }

    if (failures) {
        std::printf("test_integrators: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_integrators: OK\n");
    return 0;
}
