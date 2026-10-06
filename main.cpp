#include <cstdio>
#include <string>

#include "cmd/commands.hpp"

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
        "Run 'noether <command> --help' for command-specific options.");
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
        if (cmd == "drift")
            return cmd_drift(argc - 1, argv + 1);
        if (cmd == "reduce")
            return cmd_reduce(argc - 1, argv + 1);
        if (cmd == "bench")
            return cmd_bench(argc - 1, argv + 1);
        std::fprintf(stderr, "noether: unknown command '%s' (try --help)\n", cmd.c_str());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "noether: error: %s\n", e.what());
        return 1;
    }
}
