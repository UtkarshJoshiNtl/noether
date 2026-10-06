#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <string>
#include <vector>

#include "core/force.hpp"
#include "core/integrators.hpp"
#include "io/config.hpp"
#include "io/writer.hpp"

using namespace noether;

namespace {

void usage() {
    std::puts(
        "noether — symplectic orbital research engine\n"
        "\n"
        "Usage:\n"
        "  noether <command> [options]\n"
        "\n"
        "Commands:\n"
        "  drift    Propagate satellites and record Hamiltonian diagnostics\n"
        "  reduce   Parallel reduction determinism / precision experiment\n"
        "  bench    Kernel throughput benchmark\n"
        "\n"
        "Options for drift:\n"
        "  --integrator rk4|verlet|yoshida4   Integration scheme (default: verlet)\n"
        "  --dt <s>            Step size (default: 10)\n"
        "  --t <s>             Total duration (default: 86400)\n"
        "  --alt <km>          Circular altitude (default: 400)\n"
        "  --inc <deg>         Inclination (default: 51.6)\n"
        "  --n <count>         Ensemble size (default: 1)\n"
        "  --sample <s>        Output cadence (default: 600)\n"
        "  --out <path>        Output CSV (default: drift.csv)\n"
        "  --config <file.json>   Load parameters from a JSON config\n"
        "  --help              Show this message\n"
        "\n"
        "CLI options override values from --config.");
}

struct DriftParams {
    std::string integrator = "verlet";
    double dt = 10.0;
    double t = 86400.0;
    double alt = 400.0;
    double inc = 51.6;
    double sample = 600.0;
    int n = 1;
    std::string out = "drift.csv";
};

using StepFn = State (*)(const State&, double);

StepFn pick_integrator(const std::string& name) {
    if (name == "rk4")
        return rk4_step;
    if (name == "verlet")
        return verlet_step;
    if (name == "yoshida4")
        return yoshida_step;
    throw std::runtime_error("unknown integrator '" + name + "' (rk4|verlet|yoshida4)");
}

State circular_state(double alt_km, double inc_deg, double raan_rad, double u_rad) {
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
std::vector<State> make_ensemble(const DriftParams& p) {
    constexpr double kGolden = 0.6180339887498949;
    constexpr double kSilver = 0.3247179572447458;
    std::vector<State> states(static_cast<std::size_t>(p.n));
    for (int i = 0; i < p.n; ++i) {
        double fi = static_cast<double>(i);
        double raan = 2.0 * std::numbers::pi * std::fmod(fi * kGolden, 1.0);
        double u = 2.0 * std::numbers::pi * std::fmod(fi * kSilver, 1.0);
        states[static_cast<std::size_t>(i)] = circular_state(p.alt, p.inc, raan, u);
    }
    return states;
}

double total_energy(const std::vector<State>& states) {
    long double h = 0.0L;
    for (const State& s : states)
        h += static_cast<long double>(hamiltonian(s));
    return static_cast<double>(h);
}

int run_drift(const DriftParams& p) {
    StepFn step = pick_integrator(p.integrator);
    std::vector<State> states = make_ensemble(p);

    const int steps = static_cast<int>(std::llround(p.t / p.dt));
    const int every = std::max(1, static_cast<int>(std::llround(p.sample / p.dt)));
    if (steps < 1)
        throw std::runtime_error("drift: --t must be >= --dt");

    const double h0 = total_energy(states);
    const bool single = states.size() == 1;
    const int nsamp = steps / every + 1;  // t=0 plus every-th exact multiple

    std::vector<double> hvals(static_cast<std::size_t>(nsamp));
    std::vector<State> snaps(single ? static_cast<std::size_t>(nsamp) : 0);
    hvals[0] = h0;
    if (single)
        snaps[0] = states[0];

    // Parallelism is over particles inside each time step (time steps are
    // strictly sequential); samples are read after the implicit barrier, so
    // rows stay ordered and consistent.
    const bool par = states.size() >= 64;
    int k = 0;
    int j = 1;
    while (k < steps) {
        ++k;
#ifdef NOETHER_USE_OPENMP
#pragma omp parallel for schedule(static) if (par)
#endif
        for (std::size_t i = 0; i < states.size(); ++i)
            states[i] = step(states[i], p.dt);
        if (k % every == 0 && j < nsamp) {
            hvals[static_cast<std::size_t>(j)] = total_energy(states);
            if (single)
                snaps[static_cast<std::size_t>(j)] = states[0];
            ++j;
        }
    }

    CsvWriter csv(p.out, single ? "t,x,y,z,vx,vy,vz,h,dh_rel" : "t,h,dh_rel");
    double max_dh = 0.0;
    double last_dh = 0.0;
    for (int m = 0; m < nsamp; ++m) {
        double t = static_cast<double>(m * every) * p.dt;
        double dh = (hvals[static_cast<std::size_t>(m)] - h0) / std::fabs(h0);
        if (single) {
            const State& s = snaps[static_cast<std::size_t>(m)];
            csv.row({t, s.x, s.y, s.z, s.vx, s.vy, s.vz, hvals[static_cast<std::size_t>(m)], dh});
        } else {
            csv.row({t, hvals[static_cast<std::size_t>(m)], dh});
        }
        max_dh = std::max(max_dh, std::fabs(dh));
        last_dh = dh;
    }

    std::printf("command      = drift\n");
    std::printf("integrator   = %s\n", p.integrator.c_str());
    std::printf("backend      = cpu\n");
    std::printf("n            = %zu\n", states.size());
    std::printf("steps        = %d\n", steps);
    std::printf("dt           = %.17g\n", p.dt);
    std::printf("h0           = %.17g\n", h0);
    std::printf("max_dh_rel   = %.17g\n", max_dh);
    std::printf("final_dh_rel = %.17g\n", last_dh);
    std::printf("out          = %s\n", p.out.c_str());
    return 0;
}

void apply_config(const Config& cfg, DriftParams& p) {
    p.integrator = cfg.str("integrator", p.integrator);
    p.dt = cfg.num("dt", p.dt);
    p.t = cfg.num("t", p.t);
    p.alt = cfg.num("alt", p.alt);
    p.inc = cfg.num("inc", p.inc);
    p.sample = cfg.num("sample", p.sample);
    p.n = cfg.integer("n", p.n);
    p.out = cfg.str("out", p.out);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        usage();
        return 0;
    }

    try {
        if (cmd == "drift") {
            Config cfg;
            DriftParams p;
            for (int i = 2; i < argc; ++i) {
                std::string a = argv[i];
                auto next = [&](const char* name) -> std::string {
                    if (i + 1 >= argc)
                        throw std::runtime_error(std::string(name) + " needs a value");
                    return argv[++i];
                };
                if (a == "--config") {
                    cfg = load_config(next("--config"));
                    apply_config(cfg, p);
                } else if (a == "--integrator") {
                    p.integrator = next("--integrator");
                } else if (a == "--dt") {
                    p.dt = std::stod(next("--dt"));
                } else if (a == "--t") {
                    p.t = std::stod(next("--t"));
                } else if (a == "--alt") {
                    p.alt = std::stod(next("--alt"));
                } else if (a == "--inc") {
                    p.inc = std::stod(next("--inc"));
                } else if (a == "--sample") {
                    p.sample = std::stod(next("--sample"));
                } else if (a == "--n") {
                    p.n = std::stoi(next("--n"));
                } else if (a == "--out") {
                    p.out = next("--out");
                } else {
                    throw std::runtime_error("unknown option '" + a + "' (try --help)");
                }
            }
            if (p.n < 1)
                throw std::runtime_error("--n must be >= 1");
            return run_drift(p);
        }

        if (cmd == "reduce" || cmd == "bench") {
            std::fprintf(stderr, "noether: command '%s' lands in P2\n", cmd.c_str());
            return 2;
        }

        std::fprintf(stderr, "noether: unknown command '%s' (try --help)\n", cmd.c_str());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "noether: error: %s\n", e.what());
        return 1;
    }
}
