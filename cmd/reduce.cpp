#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "cmd/commands.hpp"
#include "core/reductions.hpp"
#include "io/writer.hpp"

#ifdef NOETHER_USE_CUDA
#include "core/cuda/api.hpp"
#endif

using namespace noether;

namespace {

void usage_reduce() {
    std::puts(
        "noether reduce — parallel reduction determinism / precision experiment\n"
        "\n"
        "Usage:\n"
        "  noether reduce [options]\n"
        "\n"
        "Options:\n"
        "  --n <count>         Input length (default: 1048576)\n"
        "  --repeats <r>       Repeats per configuration (default: 100)\n"
        "  --input lcg|adversarial|unity   Input generator (default: lcg)\n"
        "  --modes a,b,c       Subset of atomic,fixed,kahan (default: all)\n"
        "  --precs a,b         Subset of fp64,fp32 (default: both)\n"
        "  --streams <k>       Atomic streams, 1..8 (default: 1)\n"
        "  --seed <s>          LCG seed (default: 42)\n"
        "  --out <path>        Per-repeat CSV (optional)\n"
        "  --help              Show this message\n"
        "\n"
        "CSV columns: input,n,mode,prec,streams,repeat,ms,value,bits");
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty())
                out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

std::vector<double> make_input(const std::string& kind, std::size_t n, std::uint64_t seed) {
    std::vector<double> v(n);
    if (kind == "lcg") {
        std::uint64_t s = seed;
        for (std::size_t i = 0; i < n; ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            std::uint64_t u = s >> 11;  // top 53 bits
            v[i] = static_cast<double>(u) * (1.0 / 9007199254740992.0);
        }
    } else if (kind == "adversarial") {
        // Large alternating magnitudes with a tiny residue: catastrophic
        // cancellation — separates accumulator quality sharply.
        for (std::size_t i = 0; i < n; ++i)
            v[i] = (i % 2 ? -1e8 : 1e8) + static_cast<double>(i % 7);
    } else if (kind == "unity") {
        for (std::size_t i = 0; i < n; ++i)
            v[i] = 1.0;
    } else {
        throw std::runtime_error("reduce: unknown --input (lcg|adversarial|unity)");
    }
    return v;
}

}  // namespace

int cmd_reduce(int argc, char** argv) {
#ifndef NOETHER_USE_CUDA
    (void)argc;
    (void)argv;
    std::fprintf(stderr, "noether reduce: built without CUDA\n");
    return 2;
#else
    std::size_t n = 1 << 20;
    int repeats = 100;
    std::string input_kind = "lcg";
    std::string modes_arg = "atomic,fixed,kahan";
    std::string precs_arg = "fp64,fp32";
    int streams = 1;
    std::uint64_t seed = 42;
    std::string out;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string(name) + " needs a value");
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            usage_reduce();
            return 0;
        } else if (a == "--n") {
            n = static_cast<std::size_t>(std::stoll(next("--n")));
        } else if (a == "--repeats") {
            repeats = std::stoi(next("--repeats"));
        } else if (a == "--input") {
            input_kind = next("--input");
        } else if (a == "--modes") {
            modes_arg = next("--modes");
        } else if (a == "--precs") {
            precs_arg = next("--precs");
        } else if (a == "--streams") {
            streams = std::stoi(next("--streams"));
        } else if (a == "--seed") {
            seed = static_cast<std::uint64_t>(std::stoull(next("--seed")));
        } else if (a == "--out") {
            out = next("--out");
        } else {
            throw std::runtime_error("reduce: unknown option '" + a + "' (try --help)");
        }
    }
    if (n < 1)
        throw std::runtime_error("reduce: --n must be >= 1");
    if (repeats < 1)
        throw std::runtime_error("reduce: --repeats must be >= 1");
    if (streams < 1 || streams > 8)
        throw std::runtime_error("reduce: --streams must be in 1..8");

    if (!cuda::available())
        throw std::runtime_error("reduce: no CUDA device");

    std::vector<double> input = make_input(input_kind, n, seed);
    const long double oracle_ld = sum_longdouble(input.data(), n);
    const double oracle_kahan = sum_kahan(input.data(), n);

    cuda::DeviceMem<double> d_in(n);
    d_in.upload(input);

    std::unique_ptr<CsvWriter> csv;
    if (!out.empty())
        csv = std::make_unique<CsvWriter>(out, "input,n,mode,prec,streams,repeat,ms,value,bits");

    std::printf("command    = reduce\n");
    std::printf("n          = %zu\n", n);
    std::printf("repeats    = %d\n", repeats);
    std::printf("input      = %s\n", input_kind.c_str());
    std::printf("oracle_ld  = %.17g\n", static_cast<double>(oracle_ld));
    std::printf("oracle_kahan = %.17g\n", oracle_kahan);
    std::printf("\n%-8s %-6s %-9s %-9s %-13s %-11s %s\n", "mode", "prec", "unique", "spread",
                "rel_err", "mean_ms", "streams");

    for (const std::string& ms : split_csv(modes_arg)) {
        cuda::ReduceMode mode;
        if (!cuda::parse_reduce_mode(ms, mode))
            throw std::runtime_error("reduce: unknown mode '" + ms + "'");
        for (const std::string& ps : split_csv(precs_arg)) {
            cuda::ReducePrec prec;
            if (!cuda::parse_reduce_prec(ps, prec))
                throw std::runtime_error("reduce: unknown prec '" + ps + "'");

            std::vector<double> values;
            std::vector<std::uint64_t> bits;
            double total_ms = 0.0;
            values.reserve(static_cast<std::size_t>(repeats));
            bits.reserve(static_cast<std::size_t>(repeats));

            // One untimed warmup (JIT/allocator effects) for fair timings.
            cuda::reduce(d_in.data(), n, mode, prec, streams);

            for (int r = 0; r < repeats; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                double v = cuda::reduce(d_in.data(), n, mode, prec, streams);
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_ms += ms;
                values.push_back(v);
                bits.push_back(std::bit_cast<std::uint64_t>(v));
                if (csv) {
                    char line[256];
                    std::snprintf(line, sizeof(line), "%s,%zu,%s,%s,%d,%d,%.6f,%.17g,%016llx",
                                  input_kind.c_str(), n, cuda::to_string(mode),
                                  cuda::to_string(prec), streams, r, ms, v,
                                  static_cast<unsigned long long>(bits.back()));
                    csv->raw_row(line);
                }
            }

            std::uint64_t unique = 0;
            {
                std::vector<std::uint64_t> u = bits;
                std::sort(u.begin(), u.end());
                u.erase(std::unique(u.begin(), u.end()), u.end());
                unique = u.size();
            }
            double vmin = *std::min_element(values.begin(), values.end());
            double vmax = *std::max_element(values.begin(), values.end());
            double spread = vmax - vmin;
            double rel_err = std::fabs(values[0] - static_cast<double>(oracle_ld)) /
                             std::fabs(static_cast<double>(oracle_ld));
            double mean_ms = total_ms / repeats;
            std::printf("%-8s %-6s %llu/%-6d %-13.3e %-11.3e %.4f %d\n", cuda::to_string(mode),
                        cuda::to_string(prec), static_cast<unsigned long long>(unique), repeats,
                        spread, rel_err, mean_ms, streams);
        }
    }
    if (csv)
        std::printf("out        = %s\n", out.c_str());
    return 0;
#endif
}
