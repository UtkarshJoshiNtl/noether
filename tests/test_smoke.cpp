#include <cassert>

#include "core/constants.hpp"

int main() {
    using namespace noether;
    assert(MU_EARTH > 3.9e5 && MU_EARTH < 4.0e5);
    assert(R_EARTH > 6378.0 && R_EARTH < 6379.0);
    assert(J2 > 1.0e-3 && J2 < 1.1e-3);
    assert(J3 < 0.0);
    assert(J4 < 0.0);
    return 0;
}
