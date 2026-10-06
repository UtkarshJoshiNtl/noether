#pragma once

// Single source of truth for all physical constants (WGS-84 / EGM96 values).
// All quantities are in the ECI frame with km, km/s units.

namespace noether {

inline constexpr double MU_EARTH = 398600.4418;  // Earth gravitational parameter [km^3/s^2]
inline constexpr double R_EARTH = 6378.137;      // Earth equatorial radius [km]
inline constexpr double J2 = 1.08263e-3;         // Second zonal harmonic (oblateness)
inline constexpr double J3 = -2.53266e-6;         // Third zonal harmonic
inline constexpr double J4 = -1.61990e-6;         // Fourth zonal harmonic

}  // namespace noether
