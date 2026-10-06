#include <cstdio>

#include "core/constants.hpp"

using namespace noether;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::printf("FAIL: %s\n", what);
            ++failures;
        }
    };
    check(MU_EARTH > 3.9e5 && MU_EARTH < 4.0e5, "MU_EARTH in range");
    check(R_EARTH > 6378.0 && R_EARTH < 6379.0, "R_EARTH in range");
    check(J2 > 1.0e-3 && J2 < 1.1e-3, "J2 in range");
    check(J3 < 0.0, "J3 negative");
    check(J4 < 0.0, "J4 negative");
    if (failures)
        return 1;
    std::printf("test_smoke: OK\n");
    return 0;
}
