#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cmd/commands.hpp"
#include "core/ensemble.hpp"
#include "core/force.hpp"
#include "core/integrators.hpp"
#include "io/config.hpp"
#include "io/writer.hpp"

#ifdef NOETHER_USE_CUDA
#include "core/cuda/api.hpp"
#endif

using namespace noether;

namespace {

void usage_drift() {
    std::puts(
        "noether drift — propagate satellites and record Hamiltonian diagnostics\n"
        "\n"
        "Usage:\n"
        "  noether drift [options]\n"
        "\n"
        "Options:\n"
        "  --integrator rk4|verlet|yoshida4   Integration scheme (default: verlet)\n"
        "  --backend cpu|gpu       Execution backend (default: cpu)\n"
        "  --dt <s>                Step size (default: 10)\n"
        "  --t <s>                 Total duration (default: 86400)\n"
        "  --alt <km>              Circular altitude (default: 400)\n"
        "  --inc <deg>             Inclination (default: 51.6)\n"
        "  --n <count>             Ensemble size (default: 1)\n"
        "  --sample <s>            Output cadence (default: 600)\n"
        "  --out <path>            Output CSV (default: drift.csv)\n"
        "  --red-mode atomic|fixed|kahan     GPU energy reduction mode (default: fixed)\n"
        "  --red-prec fp64|fp32            GPU reduction accumulator (default: fp64)\n"
        "  --red-streams <k>       Atomic reduction streams (default: 1)\n"
        "  --precision fp64|fp32|fp32k     GPU state precision; fp32k adds Kahan\n"
        "                           kick compensation; verlet only (default: fp64)\n"
        "  --config <file.json>    Load parameters from a JSON config\n"
        "  --help                  Show this message\n"
        "\n"
        "CLI options override values from --config.\n"
        "CSV: single satellite -> t,x,y,z,vx,vy,vz,h,dh_rel ; ensemble -> t,h,dh_rel");
}

struct DriftParams {
    std::string integrator = "verlet";
    std::string backend = "cpu";
    double dt = 10.0;
    double t = 86400.0;
    double alt = 400.0;
    double inc = 51.6;
    double sample = 600.0;
    int n = 1;
    std::string out = "drift.csv";
    std::string red_mode = "fixed";
    std::string red_prec = "fp64";
    int red_streams = 1;
    std::string precision = "fp64";  // GPU state precision: fp64 | fp32 | fp32k
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

struct DriftSeries {
    std::vector<double> hvals;
    std::vector<State> snaps;
};

void validate_common(const DriftParams& p, int& steps, int& every, int& nsamp) {
    pick_integrator(p.integrator);  // validates name (also used by GPU path)
    steps = static_cast<int>(std::llround(p.t / p.dt));
    every = std::max(1, static_cast<int>(std::llround(p.sample / p.dt)));
    if (steps < 1)
        throw std::runtime_error("drift: --t must be >= --dt");
    nsamp = steps / every + 1;  // t=0 plus every-th exact multiple
}

DriftSeries compute_cpu(const DriftParams& p, int steps, int every, int nsamp) {
    StepFn step = pick_integrator(p.integrator);
    std::vector<State> states = make_ensemble(p.n, p.alt, p.inc);
    const bool single = states.size() == 1;

    DriftSeries out;
    out.hvals.assign(static_cast<std::size_t>(nsamp), 0.0);
    out.snaps.resize(single ? static_cast<std::size_t>(nsamp) : 0);
    out.hvals[0] = total_energy(states);
    if (single)
        out.snaps[0] = states[0];

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
            out.hvals[static_cast<std::size_t>(j)] = total_energy(states);
            if (single)
                out.snaps[static_cast<std::size_t>(j)] = states[0];
            ++j;
        }
    }
    return out;
}

#ifdef NOETHER_USE_CUDA
DriftSeries compute_gpu(const DriftParams& p, int steps, int every, int nsamp) {
    if (!cuda::available())
        throw std::runtime_error("drift: no CUDA device (use --backend cpu)");
    cuda::ReduceMode mode;
    cuda::ReducePrec prec;
    if (!cuda::parse_reduce_mode(p.red_mode, mode))
        throw std::runtime_error("drift: unknown --red-mode (atomic|fixed|kahan)");
    if (!cuda::parse_reduce_prec(p.red_prec, prec))
        throw std::runtime_error("drift: unknown --red-prec (fp64|fp32)");

    std::vector<State> states = make_ensemble(p.n, p.alt, p.inc);
    const bool single = states.size() == 1;
    cuda::DeviceEnsemble ens(states);

    DriftSeries out;
    out.hvals.assign(static_cast<std::size_t>(nsamp), 0.0);
    out.snaps.resize(single ? static_cast<std::size_t>(nsamp) : 0);
    out.hvals[0] = cuda::energy_sum(ens, mode, prec, p.red_streams);
    if (single) {
        std::vector<State> one(1);
        ens.download(one);
        out.snaps[0] = one[0];
    }

    auto step_one = [&]() {
        if (p.integrator == "rk4")
            cuda::step_rk4(ens, p.dt);
        else if (p.integrator == "yoshida4")
            cuda::step_yoshida(ens, p.dt);
        else
            cuda::step_verlet(ens, p.dt);
    };

    int k = 0;
    int j = 1;
    while (k < steps) {
        ++k;
        step_one();
        if (k % every == 0 && j < nsamp) {
            out.hvals[static_cast<std::size_t>(j)] =
                cuda::energy_sum(ens, mode, prec, p.red_streams);
            if (single) {
                std::vector<State> one(1);
                ens.download(one);
                out.snaps[static_cast<std::size_t>(j)] = one[0];
            }
            ++j;
        }
    }
    return out;
}

// fp32 state path (E4): same loop, F32Ensemble, verlet-only kernels.
DriftSeries compute_gpu_f32(const DriftParams& p, int steps, int every, int nsamp) {
    if (!cuda::available())
        throw std::runtime_error("drift: no CUDA device (use --backend cpu)");
    cuda::ReduceMode mode;
    cuda::ReducePrec prec;
    if (!cuda::parse_reduce_mode(p.red_mode, mode))
        throw std::runtime_error("drift: unknown --red-mode (atomic|fixed|kahan)");
    if (!cuda::parse_reduce_prec(p.red_prec, prec))
        throw std::runtime_error("drift: unknown --red-prec (fp64|fp32)");
    const bool kahan_kick = p.precision == "fp32k";

    std::vector<State> states = make_ensemble(p.n, p.alt, p.inc);
    const bool single = states.size() == 1;
    cuda::F32Ensemble ens(states);

    DriftSeries out;
    out.hvals.assign(static_cast<std::size_t>(nsamp), 0.0);
    out.snaps.resize(single ? static_cast<std::size_t>(nsamp) : 0);
    out.hvals[0] = cuda::energy_sum(ens, mode, prec, p.red_streams);
    if (single) {
        std::vector<State> one(1);
        ens.download(one);
        out.snaps[0] = one[0];
    }

    int k = 0;
    int j = 1;
    while (k < steps) {
        ++k;
        if (kahan_kick)
            cuda::step_verlet_f32k(ens, p.dt);
        else
            cuda::step_verlet_f32(ens, p.dt);
        if (k % every == 0 && j < nsamp) {
            out.hvals[static_cast<std::size_t>(j)] =
                cuda::energy_sum(ens, mode, prec, p.red_streams);
            if (single) {
                std::vector<State> one(1);
                ens.download(one);
                out.snaps[static_cast<std::size_t>(j)] = one[0];
            }
            ++j;
        }
    }
    return out;
}
#endif

int run_drift(const DriftParams& p) {
    int steps = 0, every = 0, nsamp = 0;
    validate_common(p, steps, every, nsamp);

    if (p.backend != "cpu" && p.backend != "gpu")
        throw std::runtime_error("drift: unknown --backend (cpu|gpu)");
    if (p.precision != "fp64" && p.precision != "fp32" && p.precision != "fp32k")
        throw std::runtime_error("drift: unknown --precision (fp64|fp32|fp32k)");
    if (p.precision != "fp64") {
        if (p.backend != "gpu")
            throw std::runtime_error("drift: --precision fp32/fp32k requires --backend gpu");
        if (p.integrator != "verlet")
            throw std::runtime_error("drift: --precision fp32/fp32k supports only verlet");
    }
    const bool gpu = p.backend == "gpu";

    DriftSeries series;
    if (gpu) {
#ifdef NOETHER_USE_CUDA
        series = p.precision == "fp64" ? compute_gpu(p, steps, every, nsamp)
                                       : compute_gpu_f32(p, steps, every, nsamp);
#else
        throw std::runtime_error("drift: built without CUDA (use --backend cpu)");
#endif
    } else {
        series = compute_cpu(p, steps, every, nsamp);
    }

    const bool single = p.n == 1;
    const double h0 = series.hvals[0];
    CsvWriter csv(p.out, single ? "t,x,y,z,vx,vy,vz,h,dh_rel" : "t,h,dh_rel");
    double max_dh = 0.0;
    double last_dh = 0.0;
    for (int m = 0; m < nsamp; ++m) {
        double t = static_cast<double>(m * every) * p.dt;
        double h = series.hvals[static_cast<std::size_t>(m)];
        double dh = (h - h0) / std::fabs(h0);
        if (single) {
            const State& s = series.snaps[static_cast<std::size_t>(m)];
            csv.row({t, s.x, s.y, s.z, s.vx, s.vy, s.vz, h, dh});
        } else {
            csv.row({t, h, dh});
        }
        max_dh = std::max(max_dh, std::fabs(dh));
        last_dh = dh;
    }

    std::printf("command      = drift\n");
    std::printf("integrator   = %s\n", p.integrator.c_str());
    std::printf("backend      = %s\n", p.backend.c_str());
    if (gpu) {
        std::printf("red_mode     = %s\n", p.red_mode.c_str());
        std::printf("red_prec     = %s\n", p.red_prec.c_str());
        std::printf("red_streams  = %d\n", p.red_streams);
        std::printf("precision    = %s\n", p.precision.c_str());
    }
    std::printf("n            = %d\n", p.n);
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
    p.backend = cfg.str("backend", p.backend);
    p.dt = cfg.num("dt", p.dt);
    p.t = cfg.num("t", p.t);
    p.alt = cfg.num("alt", p.alt);
    p.inc = cfg.num("inc", p.inc);
    p.sample = cfg.num("sample", p.sample);
    p.n = cfg.integer("n", p.n);
    p.out = cfg.str("out", p.out);
    p.red_mode = cfg.str("red_mode", p.red_mode);
    p.red_prec = cfg.str("red_prec", p.red_prec);
    p.red_streams = cfg.integer("red_streams", p.red_streams);
    p.precision = cfg.str("precision", p.precision);
}

}  // namespace

int cmd_drift(int argc, char** argv) {
    Config cfg;
    DriftParams p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string(name) + " needs a value");
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            usage_drift();
            return 0;
        } else if (a == "--config") {
            cfg = load_config(next("--config"));
            apply_config(cfg, p);
        } else if (a == "--integrator") {
            p.integrator = next("--integrator");
        } else if (a == "--backend") {
            p.backend = next("--backend");
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
        } else if (a == "--red-mode") {
            p.red_mode = next("--red-mode");
        } else if (a == "--red-prec") {
            p.red_prec = next("--red-prec");
        } else if (a == "--red-streams") {
            p.red_streams = std::stoi(next("--red-streams"));
        } else if (a == "--precision") {
            p.precision = next("--precision");
        } else {
            throw std::runtime_error("drift: unknown option '" + a + "' (try --help)");
        }
    }
    if (p.n < 1)
        throw std::runtime_error("drift: --n must be >= 1");
    return run_drift(p);
}
