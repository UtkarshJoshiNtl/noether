#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "cmd/commands.hpp"
#include "core/ensemble.hpp"
#include "core/force.hpp"
#include "core/integrators.hpp"
#include "io/writer.hpp"

#ifdef NOETHER_USE_CUDA
#include "core/cuda/api.hpp"
#include "core/cuda/timing.hpp"
#endif

using namespace noether;

namespace {

void usage_bench() {
    std::puts(
        "noether bench — kernel throughput benchmark\n"
        "\n"
        "Usage:\n"
        "  noether bench [options]\n"
        "\n"
        "Options:\n"
        "  --n <count>       Ensemble size (default: 1048576)\n"
        "  --steps <s>       Steps per timing run (default: 100)\n"
        "  --repeats <r>     Timing repeats, best reported (default: 5)\n"
        "  --cpu             Also measure the CPU (OpenMP) verlet loop\n"
        "  --out <path>      CSV output (optional)\n"
        "  --help            Show this message\n"
        "\n"
        "CSV columns: kind,n,detail,ns_per_call,calls");
}

struct Row {
    std::string kind;
    std::size_t n;
    std::string detail;
    double ns_per_call;
    double calls;
};

void print_row(const Row& r) {
    std::printf("%-7s %-10zu %-22s %12.1f %8.0f\n", r.kind.c_str(), r.n, r.detail.c_str(),
                r.ns_per_call, r.calls);
}

}  // namespace

int cmd_bench(int argc, char** argv) {
    std::size_t n = 1 << 20;
    int steps = 100;
    int repeats = 5;
    bool with_cpu = false;
    std::string out;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string(name) + " needs a value");
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            usage_bench();
            return 0;
        } else if (a == "--n") {
            n = static_cast<std::size_t>(std::stoll(next("--n")));
        } else if (a == "--steps") {
            steps = std::stoi(next("--steps"));
        } else if (a == "--repeats") {
            repeats = std::stoi(next("--repeats"));
        } else if (a == "--cpu") {
            with_cpu = true;
        } else if (a == "--out") {
            out = next("--out");
        } else {
            throw std::runtime_error("bench: unknown option '" + a + "' (try --help)");
        }
    }
    if (n < 1)
        throw std::runtime_error("bench: --n must be >= 1");
    if (steps < 1 || repeats < 1)
        throw std::runtime_error("bench: --steps and --repeats must be >= 1");

    std::vector<Row> rows;

    // ── CPU verlet (opt-in: E4 Pareto baseline) ────────────────────────────────
    if (with_cpu) {
        std::vector<State> states = make_ensemble(static_cast<int>(n), 400.0, 51.6);
        const bool par = states.size() >= 64;
        double best_ns = 1e300;
        for (int r = 0; r < repeats; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < steps; ++s) {
#ifdef NOETHER_USE_OPENMP
#pragma omp parallel for schedule(static) if (par)
#endif
                for (std::size_t i = 0; i < states.size(); ++i)
                    states[i] = verlet_step(states[i], 10.0);
            }
            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / steps;
            best_ns = std::min(best_ns, ns);
        }
        rows.push_back({"step", n, "cpu:verlet", best_ns, static_cast<double>(steps)});
    }

#ifdef NOETHER_USE_CUDA
    if (!cuda::available())
        throw std::runtime_error("bench: no CUDA device");

    std::vector<State> states = make_ensemble(static_cast<int>(n), 400.0, 51.6);
    cuda::DeviceEnsemble ens(states);

    struct StepCase {
        const char* detail;
        void (*fn)(cuda::DeviceEnsemble&, double);
    };
    const StepCase step_cases[] = {
        {"gpu:verlet", cuda::step_verlet},
        {"gpu:rk4", cuda::step_rk4},
        {"gpu:yoshida4", cuda::step_yoshida},
    };

    for (const StepCase& c : step_cases) {
        // Warmup + best-of timing (cudaEvent, GPU-side only).
        c.fn(ens, 10.0);
        float best_ms = 1e30f;
        for (int r = 0; r < repeats; ++r) {
            cuda::GpuTimer timer;
            timer.start();
            for (int s = 0; s < steps; ++s)
                c.fn(ens, 10.0);
            best_ms = std::min(best_ms, timer.stop_ms());
        }
        double ns = static_cast<double>(best_ms) * 1e6 / steps;
        rows.push_back({"step", n, c.detail, ns, static_cast<double>(steps)});
    }

    struct RedCase {
        std::string detail;
        cuda::ReduceMode mode;
        cuda::ReducePrec prec;
    };
    std::vector<RedCase> red_cases = {
        {"gpu:atomic/fp64", cuda::ReduceMode::Atomic, cuda::ReducePrec::Fp64},
        {"gpu:fixed/fp64", cuda::ReduceMode::Fixed, cuda::ReducePrec::Fp64},
        {"gpu:kahan/fp64", cuda::ReduceMode::Kahan, cuda::ReducePrec::Fp64},
        {"gpu:atomic/fp32", cuda::ReduceMode::Atomic, cuda::ReducePrec::Fp32Acc},
        {"gpu:fixed/fp32", cuda::ReduceMode::Fixed, cuda::ReducePrec::Fp32Acc},
        {"gpu:kahan/fp32", cuda::ReduceMode::Kahan, cuda::ReducePrec::Fp32Acc},
    };

    for (const RedCase& c : red_cases) {
        cuda::reduce(ens.energy(), n, c.mode, c.prec, 1);  // warmup
        double total_ms = 0.0;
        int calls = 100;
        for (int r = 0; r < calls; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            cuda::reduce(ens.energy(), n, c.mode, c.prec, 1);
            auto t1 = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        double ns = total_ms * 1e6 / calls;
        rows.push_back({"reduce", n, c.detail, ns, static_cast<double>(calls)});
    }
#else
    (void)steps;
    (void)repeats;
#endif

    if (rows.empty()) {
        std::fprintf(stderr, "bench: nothing to measure (no CUDA in this build; pass --cpu)\n");
        return 2;
    }

    std::printf("%-7s %-10s %-22s %12s %8s\n", "kind", "n", "detail", "ns/call", "calls");
    for (const Row& r : rows)
        print_row(r);

    if (!out.empty()) {
        CsvWriter csv(out, "kind,n,detail,ns_per_call,calls");
        for (const Row& r : rows) {
            char line[256];
            std::snprintf(line, sizeof(line), "%s,%zu,%s,%.6f,%.0f", r.kind.c_str(), r.n,
                          r.detail.c_str(), r.ns_per_call, r.calls);
            csv.raw_row(line);
        }
        std::printf("out        = %s\n", out.c_str());
    }
    return 0;
}
