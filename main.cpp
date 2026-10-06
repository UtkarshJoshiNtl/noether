#include <cstdio>
#include <cstring>

static void usage() {
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
        "Common options:\n"
        "  --config <file.json>   Load parameters from a JSON config file\n"
        "  --out <path>           Output path (CSV)\n"
        "  --help                 Show this message");
}

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        usage();
        return argc < 2 ? 1 : 0;
    }
    std::fprintf(stderr, "noether: command '%s' not implemented yet\n", argv[1]);
    return 1;
}
